#!/usr/bin/env python3
"""
Construction Drawing Bid-Item Audit Pipeline (v3)

Major changes from v2:

1. Stronger anti-hallucination audit logic.
2. Original bid-item metadata leakage reduced.
3. Region must be independently identified before association is judged.
4. DIRECT / INDIRECT / NONE association classification.
5. Only DIRECT association can produce RIGHT.
6. Specification/general-note/room-label/grid/dimension regions are
   not treated as RIGHT merely because they are related to the trade.
7. Added local-context crop between tight crop and whole-sheet locator.
8. Frequency is used for review/analysis, not primary LLM judgment.
9. Added deterministic OCR hints for specification text.
10. Expanded structured output for auditing model behavior.
11. Existing checkpoint/resume behavior retained.

Pipeline:

    Phase 1   : results.json -> flattened region table
    Phase 2   : PDF -> tight/local/locator crops
    Phase 2.5 : reference sheets -> legend lookup
    Phase 2.6 : frequency pre-pass
    Phase 2.7 : deterministic OCR hints
    Phase 3   : Azure OpenAI multimodal validation
    Phase 4   : corrected results.json + audit CSV + review queue

Usage:

    python drawing_audit.py legend-extract \
        --pdf drawings.pdf \
        --results results.json

    python drawing_audit.py run \
        --results results.json \
        --pdf drawings.pdf \
        --legend legend.json

Test:

    python drawing_audit.py run \
        --results results.json \
        --pdf drawings.pdf \
        --legend legend.json \
        --max-regions 50 \
        --reset-checkpoint

Required .env:

    AZURE_OPENAI_ENDPOINT=...
    AZURE_OPENAI_API_KEY=...
    AZURE_OPENAI_DEPLOYMENT=gpt-5-nano
"""

import argparse
import base64
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_W = 3024
TARGET_H = 2160

# ------------------------------------------------------------
# Image crops
# ------------------------------------------------------------

# Tight-ish crop around region.
MEDIUM_SCALE = 2.5
MIN_MEDIUM_W = 400
MIN_MEDIUM_H = 250

# Larger local-context crop.
LOCAL_SCALE = 5.0
MIN_LOCAL_W = 1000
MIN_LOCAL_H = 700

# Whole-sheet locator.
LOCATOR_MAX_DIM = 1000
LOCATOR_BOX_COLOR = "red"
LOCATOR_BOX_WIDTH = 10

# ------------------------------------------------------------
# Frequency analysis
# ------------------------------------------------------------

FREQUENCY_FLAG_THRESHOLD = 15

# ------------------------------------------------------------
# Review
# ------------------------------------------------------------

REVIEW_QUEUE_CONFIDENCE = {"LOW", "MEDIUM"}

# ------------------------------------------------------------
# LLM
# ------------------------------------------------------------

REQUEST_DELAY = 2
MAX_RETRIES = 3
RETRY_DELAY = 3
MAX_OUTPUT_TOKENS = 4000

# ------------------------------------------------------------
# Reference sheets
# ------------------------------------------------------------

REFERENCE_SHEETS = {
    "G-002.3",
    "G-501",
    "G-511",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_value(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def safe_text(value, max_chars=3000):
    value = clean_value(value)

    if len(value) > max_chars:
        return value[:max_chars] + "... [TRUNCATED]"

    return value


def parse_quad(value):
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = json.loads(str(value))

    values = [float(v) for v in values]

    if len(values) != 8:
        raise ValueError(
            f"quad_px must contain 8 numbers: {value}"
        )

    return values


def quad_to_bbox(quad):
    xs = quad[0::2]
    ys = quad[1::2]

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    )


def json_safe(value):
    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, tuple):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


# ============================================================
# PHASE 1 — LOAD AND FLATTEN results.json
# ============================================================

def phase1_extract(results_path, output_dir):

    print("\n" + "=" * 70)
    print("PHASE 1 — LOADING results.json")
    print("=" * 70)

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    drawings = data.get(
        "drawings_processing_results",
        []
    )

    if not isinstance(drawings, list):
        raise ValueError(
            "'drawings_processing_results' must be a list."
        )

    print(f"Number of drawings: {len(drawings)}")

    rows = []
    locators = []

    for drawing_index, drawing in enumerate(drawings):

        document_id = drawing.get("document_id")
        page_number = drawing.get("page_number")
        sheet_name = drawing.get("sheet_name")
        sheet_number = drawing.get("sheet_number")
        discipline = drawing.get("discipline")

        bid_items = drawing.get(
            "bid_items",
            {}
        )

        if not isinstance(bid_items, dict):
            continue

        for scope_key, items in bid_items.items():

            if not isinstance(items, list):
                continue

            for item_index, item in enumerate(items):

                regions = item.get(
                    "regions",
                    []
                )

                if not isinstance(regions, list):
                    continue

                for region_index, region in enumerate(regions):

                    row_index = len(rows)

                    rows.append({

                        "row_index": row_index,

                        "document_id": document_id,

                        "page_number": page_number,

                        "sheet_number": sheet_number,

                        "sheet_name": sheet_name,

                        "discipline": discipline,

                        "scope": item.get(
                            "scope",
                            scope_key
                        ),

                        "scope_key": scope_key,

                        "bid_item": item.get(
                            "bid_item"
                        ),

                        "subheading": item.get(
                            "subheading"
                        ),

                        "region_index": region_index,

                        "text": region.get(
                            "text",
                            ""
                        ),

                        "quad_px": region.get(
                            "quad_px"
                        ),

                        # These are retained in phase 1 for
                        # traceability, but intentionally NOT
                        # supplied to the primary audit prompt.
                        "specifications": item.get(
                            "specifications",
                            []
                        ),

                        "references": item.get(
                            "references"
                        ),

                        "drawing_reference_detail":
                            item.get(
                                "drawing_reference_detail",
                                ""
                            ),

                        "notes": item.get(
                            "notes",
                            ""
                        ),

                        "is_reference_sheet":
                            sheet_number
                            in REFERENCE_SHEETS,
                    })

                    locators.append({

                        "row_index": row_index,

                        "drawing_index":
                            drawing_index,

                        "scope_key":
                            scope_key,

                        "item_index":
                            item_index,

                        "region_index":
                            region_index,
                    })

    regions_df = pd.DataFrame(rows)

    phase1_csv = (
        output_dir /
        "phase1_regions.csv"
    )

    regions_df.to_csv(
        phase1_csv,
        index=False
    )

    print(
        f"Total regions: {len(regions_df)}"
    )

    n_ref = int(
        regions_df[
            "is_reference_sheet"
        ].sum()
    )

    print(
        "Regions on reference sheets "
        f"(excluded from judging): {n_ref}"
    )

    print(
        "Regions to judge: "
        f"{len(regions_df) - n_ref}"
    )

    print(
        f"Phase 1 CSV: {phase1_csv}"
    )

    return (
        data,
        regions_df,
        locators
    )


# ============================================================
# PHASE 2 — PDF RENDERING + CROPS
# ============================================================

def find_pdf_page_index(
    doc,
    sheet_number,
    fallback_page_number
):

    """
    Prefer page_number because results.json was generated with that
    page numbering.

    Fall back to searching page text when page_number is invalid.
    """

    candidate = int(
        fallback_page_number
    ) - 1

    if 0 <= candidate < len(doc):
        return candidate

    sheet = clean_value(
        sheet_number
    )

    if sheet:

        for i in range(len(doc)):

            page = doc[i]

            text = (
                page
                .get_textpage()
                .get_text_range()
            )

            if sheet in text:
                return i

    raise ValueError(
        f"Could not locate page for "
        f"sheet={sheet_number!r} "
        f"page_number={fallback_page_number!r}"
    )


def render_page(
    doc,
    page_index,
    cache
):

    if page_index in cache:
        return cache[page_index]

    page = doc[page_index]

    rotation = page.get_rotation()

    bitmap = page.render(
        scale=1.0
    )

    img = (
        bitmap
        .to_pil()
        .convert("RGB")
    )

    if img.size != (
        TARGET_W,
        TARGET_H
    ):

        img = img.resize(
            (
                TARGET_W,
                TARGET_H
            ),
            Image.Resampling.LANCZOS
        )

    cache[page_index] = (
        img,
        rotation
    )

    return img, rotation


def crop_box(
    cx,
    cy,
    width,
    height
):

    x1 = max(
        0,
        cx - width / 2
    )

    y1 = max(
        0,
        cy - height / 2
    )

    x2 = min(
        TARGET_W,
        cx + width / 2
    )

    y2 = min(
        TARGET_H,
        cy + height / 2
    )

    return (
        int(x1),
        int(y1),
        int(x2),
        int(y2)
    )


def make_medium_crop(
    image,
    bbox
):

    x1, y1, x2, y2 = bbox

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    cx = (
        x1 + x2
    ) / 2

    cy = (
        y1 + y2
    ) / 2

    medium_w = max(
        bbox_w * MEDIUM_SCALE,
        MIN_MEDIUM_W
    )

    medium_h = max(
        bbox_h * MEDIUM_SCALE,
        MIN_MEDIUM_H
    )

    return image.crop(
        crop_box(
            cx,
            cy,
            medium_w,
            medium_h
        )
    )


def make_local_context_crop(
    image,
    bbox
):

    """
    Larger context around the target region.

    This is important because construction drawing meaning often comes
    from nearby room labels, leaders, tags, dimensions, or symbols.
    """

    x1, y1, x2, y2 = bbox

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    cx = (
        x1 + x2
    ) / 2

    cy = (
        y1 + y2
    ) / 2

    local_w = max(
        bbox_w * LOCAL_SCALE,
        MIN_LOCAL_W
    )

    local_h = max(
        bbox_h * LOCAL_SCALE,
        MIN_LOCAL_H
    )

    return image.crop(
        crop_box(
            cx,
            cy,
            local_w,
            local_h
        )
    )


def make_locator_thumbnail(
    image,
    bbox
):

    """
    Whole-sheet thumbnail with the region boxed in red.
    """

    annotated = image.copy()

    draw = ImageDraw.Draw(
        annotated
    )

    x1, y1, x2, y2 = bbox

    pad = 15

    draw.rectangle(
        (
            x1 - pad,
            y1 - pad,
            x2 + pad,
            y2 + pad
        ),
        outline=LOCATOR_BOX_COLOR,
        width=LOCATOR_BOX_WIDTH
    )

    annotated.thumbnail(
        (
            LOCATOR_MAX_DIM,
            LOCATOR_MAX_DIM
        ),
        Image.Resampling.LANCZOS
    )

    return annotated


def phase2_create_crops(
    regions_df,
    pdf_path,
    output_dir,
    skip_existing=True
):

    print("\n" + "=" * 70)
    print("PHASE 2 — RENDERING PAGES + CREATING CROPS")
    print("=" * 70)

    medium_dir = (
        output_dir /
        "crops" /
        "medium"
    )

    local_dir = (
        output_dir /
        "crops" /
        "local"
    )

    locator_dir = (
        output_dir /
        "crops" /
        "locator"
    )

    medium_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    local_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    locator_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    doc = pdfium.PdfDocument(
        str(pdf_path)
    )

    print(
        f"PDF pages: {len(doc)}"
    )

    page_cache = {}

    output_rows = []

    to_judge = (
        regions_df[
            ~regions_df[
                "is_reference_sheet"
            ]
        ]
        .copy()
    )

    total = len(to_judge)

    for counter, (_, row) in enumerate(
        to_judge.iterrows(),
        start=1
    ):

        region_id = (
            f"region_"
            f"{int(row['row_index']):06d}"
        )

        medium_path = (
            medium_dir /
            f"{region_id}.png"
        )

        local_path = (
            local_dir /
            f"{region_id}.png"
        )

        locator_path = (
            locator_dir /
            f"{region_id}.png"
        )

        result = row.to_dict()

        result["region_id"] = (
            region_id
        )

        result["medium_path"] = (
            str(medium_path)
        )

        result["local_path"] = (
            str(local_path)
        )

        result["locator_path"] = (
            str(locator_path)
        )

        if (
            skip_existing
            and medium_path.exists()
            and local_path.exists()
            and locator_path.exists()
        ):

            result[
                "coordinate_valid"
            ] = "CACHED"

            output_rows.append(
                result
            )

            if counter % 200 == 0:
                print(
                    f"  {counter}/{total} "
                    "(cached)"
                )

            continue

        try:

            quad = parse_quad(
                row["quad_px"]
            )

            x1, y1, x2, y2 = (
                quad_to_bbox(quad)
            )

            coordinate_valid = (
                x1 >= 0
                and y1 >= 0
                and x2 <= TARGET_W
                and y2 <= TARGET_H
                and x2 > x1
                and y2 > y1
            )

            page_index = (
                find_pdf_page_index(
                    doc,
                    row["sheet_number"],
                    row["page_number"]
                )
            )

            image, rotation = (
                render_page(
                    doc,
                    page_index,
                    page_cache
                )
            )

            medium = (
                make_medium_crop(
                    image,
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )
            )

            medium.save(
                medium_path
            )

            local = (
                make_local_context_crop(
                    image,
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )
            )

            local.save(
                local_path
            )

            locator = (
                make_locator_thumbnail(
                    image,
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )
            )

            locator.save(
                locator_path
            )

            result.update({

                "pdf_page_index":
                    page_index,

                "pdf_rotation":
                    rotation,

                "bbox_x1":
                    x1,

                "bbox_y1":
                    y1,

                "bbox_x2":
                    x2,

                "bbox_y2":
                    y2,

                "coordinate_valid":
                    (
                        "VALID"
                        if coordinate_valid
                        else "OUT_OF_BOUNDS"
                    ),
            })

        except Exception as e:

            print(
                f"ERROR {region_id}: {e}"
            )

            result[
                "coordinate_valid"
            ] = "ERROR"

            result[
                "phase2_error"
            ] = str(e)

        output_rows.append(
            result
        )

        if counter % 100 == 0:
            print(
                f"  {counter}/{total}"
            )

    doc.close()

    phase2_df = pd.DataFrame(
        output_rows
    )

    phase2_csv = (
        output_dir /
        "phase2_regions.csv"
    )

    phase2_df.to_csv(
        phase2_csv,
        index=False
    )

    n_errors = (
        phase2_df[
            "coordinate_valid"
        ] == "ERROR"
    ).sum()

    print(
        f"Phase 2 complete. "
        f"Errors: {n_errors}"
    )

    print(
        f"Phase 2 CSV: {phase2_csv}"
    )

    return phase2_df


# ============================================================
# PHASE 2.5 — LEGEND EXTRACTION
# ============================================================

def legend_extract_command(
    pdf_path,
    results_path,
    output_path
):

    """
    Extract raw text from reference sheets and render them for
    manual verification.
    """

    with open(
        results_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    drawings = data.get(
        "drawings_processing_results",
        []
    )

    doc = pdfium.PdfDocument(
        str(pdf_path)
    )

    dump = {}

    for drawing in drawings:

        sheet_number = (
            drawing.get(
                "sheet_number"
            )
        )

        if (
            sheet_number
            not in REFERENCE_SHEETS
        ):
            continue

        page_number = (
            drawing.get(
                "page_number"
            )
        )

        page_index = (
            int(page_number) - 1
        )

        page = doc[
            page_index
        ]

        textpage = (
            page.get_textpage()
        )

        n_chars = (
            textpage.count_chars()
        )

        raw_text = (
            textpage.get_text_range(
                0,
                n_chars
            )
        )

        bitmap = page.render(
            scale=1.0
        )

        img = (
            bitmap
            .to_pil()
            .convert("RGB")
        )

        img_path = (
            Path(output_path).parent /
            f"legend_"
            f"{sheet_number.replace('.', '_')}.png"
        )

        img.save(
            img_path
        )

        dump[sheet_number] = {

            "page_number":
                page_number,

            "sheet_name":
                drawing.get(
                    "sheet_name"
                ),

            "raw_text":
                raw_text,

            "rendered_image":
                str(img_path),
        }

        print(
            f"Dumped {sheet_number} "
            f"-> {img_path}"
        )

    doc.close()

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            dump,
            f,
            indent=2
        )

    print(
        "\nRaw legend text + rendered images "
        f"saved to: {output_path}"
    )

    print(
        "\nBuild legend.json manually and "
        "verify it visually before trusting it."
    )


def load_legend(path):

    if (
        path is None
        or not Path(path).exists()
    ):

        print(
            "No legend.json provided/found - "
            "proceeding without legend context."
        )

        return {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        legend = json.load(f)

    print(
        f"Loaded {len(legend)} "
        f"legend entries from {path}"
    )

    return legend


def find_legend_match(
    text,
    legend
):

    if not legend:
        return None

    cleaned = (
        clean_value(text)
        .upper()
    )

    if cleaned in legend:
        return legend[cleaned]

    stripped = re.sub(
        r"[^A-Z0-9]",
        "",
        cleaned
    )

    if not stripped:
        return None

    for key, entry in legend.items():

        normalized_key = re.sub(
            r"[^A-Z0-9]",
            "",
            key.upper()
        )

        if normalized_key == stripped:
            return entry

    return None


# ============================================================
# PHASE 2.6 — FREQUENCY PRE-PASS
# ============================================================

def phase2_6_frequency_scan(
    regions_df,
    output_dir
):

    print("\n" + "=" * 70)
    print("PHASE 2.6 — FREQUENCY PRE-PASS")
    print("=" * 70)

    to_judge = (
        regions_df[
            ~regions_df[
                "is_reference_sheet"
            ]
        ]
        .copy()
    )

    texts = (
        to_judge[
            "text"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    counts = Counter(
        texts
    )

    to_judge[
        "text_frequency"
    ] = texts.map(
        counts
    )

    to_judge[
        "frequency_flag"
    ] = (
        to_judge[
            "text_frequency"
        ]
        >= FREQUENCY_FLAG_THRESHOLD
    )

    flagged = to_judge[
        to_judge[
            "frequency_flag"
        ]
    ]

    print(
        "Regions flagged by repetition "
        f"(>= {FREQUENCY_FLAG_THRESHOLD}x): "
        f"{len(flagged)}"
    )

    if len(flagged) > 0:

        summary = (
            flagged
            .groupby("text")
            .size()
            .sort_values(
                ascending=False
            )
            .head(20)
        )

        print(
            "\nTop repeated tags:"
        )

        print(
            summary.to_string()
        )

    freq_csv = (
        output_dir /
        "phase2_6_frequency_flags.csv"
    )

    flagged[
        [
            "row_index",
            "sheet_number",
            "scope",
            "bid_item",
            "text",
            "text_frequency"
        ]
    ].to_csv(
        freq_csv,
        index=False
    )

    print(
        f"Frequency flags CSV: {freq_csv}"
    )

    return to_judge


# ============================================================
# PHASE 2.7 — DETERMINISTIC OCR HINTS
# ============================================================

def detect_ocr_hints(text):

    """
    These are ONLY hints.

    They must NEVER directly determine RIGHT/WRONG.

    Their purpose is to help the LLM recognize that OCR may represent
    specification text, notes, dimensions, etc.
    """

    text = clean_value(
        text
    )

    upper = text.upper()

    hints = []

    # --------------------------------------------------------
    # Specification section patterns
    # --------------------------------------------------------

    if re.search(
        r"\b\d{2}\s*[-.]?\s*\d{2}\s*[-.]?\s*\d{2}\b",
        upper
    ):

        hints.append(
            "POSSIBLE_SPECIFICATION_SECTION"
        )

    if re.search(
        r"\bSECTION\s+\d{2}",
        upper
    ):

        hints.append(
            "POSSIBLE_SPECIFICATION_SECTION"
        )

    # --------------------------------------------------------
    # Specification terminology
    # --------------------------------------------------------

    specification_words = [
        "MATERIALS",
        "EXECUTION",
        "SUBMITTALS",
        "PRODUCTS",
        "INSTALLATION",
        "QUALITY ASSURANCE",
        "GENERAL",
        "PATCHING COMPOUND",
        "MANUFACTURERS",
    ]

    for word in specification_words:

        if word in upper:

            hints.append(
                "POSSIBLE_SPECIFICATION_TEXT"
            )

            break

    # --------------------------------------------------------
    # Dimension hints
    # --------------------------------------------------------

    if re.search(
        r"\b\d+['\"][- ]?\d*",
        upper
    ):

        hints.append(
            "POSSIBLE_DIMENSION"
        )

    # --------------------------------------------------------
    # Grid-like short text
    # --------------------------------------------------------

    compact = re.sub(
        r"[^A-Z0-9]",
        "",
        upper
    )

    if (
        1 <= len(compact) <= 3
        and compact
    ):

        hints.append(
            "SHORT_TAG"
        )

    return sorted(
        set(hints)
    )


def phase2_7_add_ocr_hints(
    regions_df,
    output_dir
):

    print("\n" + "=" * 70)
    print("PHASE 2.7 — OCR HINT DETECTION")
    print("=" * 70)

    regions_df = regions_df.copy()

    regions_df[
        "ocr_hints"
    ] = (
        regions_df[
            "text"
        ]
        .apply(
            detect_ocr_hints
        )
        .apply(
            json.dumps
        )
    )

    output_path = (
        output_dir /
        "phase2_7_ocr_hints.csv"
    )

    regions_df[
        [
            "row_index",
            "sheet_number",
            "bid_item",
            "text",
            "ocr_hints"
        ]
    ].to_csv(
        output_path,
        index=False
    )

    print(
        f"OCR hint CSV: {output_path}"
    )

    return regions_df


# ============================================================
# PHASE 3 — LLM PROMPT
# ============================================================

def build_llm_prompt(
    row,
    legend_entry
):

    """
    IMPORTANT:

    This prompt deliberately does NOT provide:

        scope
        specifications
        references
        drawing_reference_detail
        notes

    Those fields originate from the original association and can leak
    the answer.

    The LLM must first identify the region independently.
    """

    legend_block = ""

    if legend_entry:

        legend_block = f"""

============================================================
PROJECT LEGEND INFORMATION
============================================================

The OCR text appears to match a project legend entry.

Source sheet:
{safe_text(
    legend_entry.get(
        "source_sheet",
        "?"
    )
)}

Meaning:
{safe_text(
    legend_entry.get(
        "meaning",
        ""
    )
)}

Associated package, if explicitly defined:
{safe_text(
    legend_entry.get(
        "package",
        "not specified"
    )
)}

IMPORTANT:
Use this legend information only to interpret the visible code.

A legend definition does NOT by itself prove that this specific
highlighted region represents the original bid item.

The visual location and drawing relationship must still establish
the association.
"""

    ocr_hints = clean_value(
        row.get(
            "ocr_hints",
            ""
        )
    )

    ocr_hint_block = ""

    if ocr_hints:

        ocr_hint_block = f"""

============================================================
AUTOMATIC OCR HINTS
============================================================

The preprocessing system detected:

{ocr_hints}

These are ONLY hints.

They are NOT a verdict.

Verify them against the images.
"""

    return f"""
You are an expert construction drawing auditor.

Your task is to determine whether the ORIGINAL BID ITEM is correctly
associated with THIS SPECIFIC DRAWING REGION.

The original association may be wrong.

Do NOT assume the original bid item is correct.

============================================================
MOST IMPORTANT RULE
============================================================

You are NOT being asked:

"Is this region generally related to the bid item?"

You ARE being asked:

"Does this specific highlighted region directly represent, identify,
locate, or explicitly reference the original bid item?"

The distinction is critical.

============================================================
EVIDENCE PRIORITY
============================================================

Use the supplied images as the primary evidence.

You will receive:

1. A tight crop around the extracted region.
2. A larger local-context crop around the region.
3. A whole-sheet locator with the region marked in red.

The red rectangle is NOT part of the original construction drawing.

It is only an indicator showing which region is being audited.

Do not interpret the red rectangle itself as evidence.

OCR is secondary evidence.

OCR may be incomplete, incorrectly reconstructed, or misleading.

============================================================
ORIGINAL BID ITEM
============================================================

Bid Item:
{safe_text(
    row.get(
        "bid_item",
        ""
    )
)}

Subheading:
{safe_text(
    row.get(
        "subheading",
        ""
    )
)}

============================================================
DRAWING IDENTIFICATION
============================================================

Sheet Number:
{safe_text(
    row.get(
        "sheet_number",
        ""
    )
)}

Sheet Name:
{safe_text(
    row.get(
        "sheet_name",
        ""
    )
)}

Discipline:
{safe_text(
    row.get(
        "discipline",
        ""
    )
)}

============================================================
REGION OCR
============================================================

{safe_text(
    row.get(
        "text",
        ""
    )
)}

{legend_block}

{ocr_hint_block}

============================================================
REQUIRED AUDIT PROCEDURE
============================================================

Perform the following conceptual steps before deciding.

STEP 1 — IDENTIFY THE REGION

Determine exactly what the highlighted region contains.

Classify it as one of:

- MATERIAL_TAG
- FINISH_TAG
- ROOM_LABEL
- GRID_REFERENCE
- DIMENSION
- SPECIFICATION_TEXT
- GENERAL_NOTE
- SCHEDULE
- DETAIL
- EQUIPMENT_TAG
- PLAN_GEOMETRY
- TITLE_BLOCK
- OTHER

Do not assume the region is a material simply because its OCR contains
a material-related word.

------------------------------------------------------------

STEP 2 — IDENTIFY WHAT IT REPRESENTS

Based only on the drawing evidence, determine what the region itself
represents.

For example:

If it contains:
"09 05 61 MATERIALS"

then it represents specification text.

If it contains:
"CARPET C-1"
with a visible finish association to a floor area,
then it may represent a carpet finish.

If it contains:
"ELECTRICAL 118"
then it represents a room label.

If it contains:
"A / 5"
inside a circular grid bubble,
then it represents a grid reference.

------------------------------------------------------------

STEP 3 — INDEPENDENTLY IDENTIFY THE REGION

Temporarily ignore the original bid item.

Ask yourself:

"What would I say this highlighted region represents if I did not know
the original bid item?"

This is the counterfactual test.

------------------------------------------------------------

STEP 4 — COMPARE WITH THE BID ITEM

Only after identifying the region independently should you compare it
with the original bid item.

------------------------------------------------------------

STEP 5 — TEST FOR INDIRECT REASONING

Ask:

"Am I calling this RIGHT because this region could support, relate to,
belong to, or be used by the original bid item?"

If YES, that is NOT sufficient.

The relationship must be DIRECT.

============================================================
DIRECT ASSOCIATION REQUIREMENT
============================================================

RIGHT requires DIRECT association.

DIRECT means the highlighted region itself visibly:

- identifies the material/work;
- identifies a finish assigned to an area;
- identifies an object belonging to the bid item;
- points to or labels the relevant work;
- represents a schedule/detail entry explicitly corresponding to the
  bid item;
- or otherwise provides direct drawing evidence for the bid item.

============================================================
INDIRECT ASSOCIATION
============================================================

The following are INDIRECT and therefore WRONG:

- same trade
- same discipline
- same scope
- same specification division
- same material category
- same sheet
- same room
- nearby related work
- ancillary work
- preparation work
- specification mentioning the material
- general note concerning the trade
- a construction activity that could support the bid item

If the association requires the model to reason:

"X is related to Y, therefore this must be Y"

then classify WRONG.

============================================================
NO ASSOCIATION
============================================================

If the region does not provide evidence for the bid item,
classify WRONG.

============================================================
SPECIAL RULE — SPECIFICATION TEXT
============================================================

Specification text is NOT the same thing as the physical work shown
on the construction drawing.

For example:

Original Bid Item:
Furnish & Install Carpet

Region:
"09 05 61 Common Work Results for Flooring Preparation"
"2.01 MATERIALS"
"Patching Compound"

This is flooring-related.

However, it does NOT directly establish that the highlighted region
represents carpet installation.

Therefore:

decision = WRONG
association_strength = NONE or INDIRECT

Do NOT use general construction knowledge to convert specification
language into evidence that the highlighted region is carpet.

============================================================
SPECIAL RULE — ROOM LABEL
============================================================

A room name alone does not establish the bid item.

Example:

"ELECTRICAL 118"

does not prove that carpet, flooring, painting, ceiling, etc. is
associated with that region.

A room label without a direct finish/work indication is WRONG.

============================================================
SPECIAL RULE — GRID REFERENCES
============================================================

A grid bubble or grid-line reference is NOT a bid item.

Repeated short tags should be inspected carefully.

Do not classify a short tag as a product merely because it has
2-3 characters.

============================================================
SPECIAL RULE — SAME TRADE
============================================================

Same trade does NOT mean same bid item.

For example:

Flooring preparation
Floor finish schedule
Carpet
Vinyl flooring
Tile
Resilient flooring

may all belong to a broad flooring category, but the existence of
flooring-related information does not automatically prove a specific
carpet bid item.

============================================================
SPECIAL RULE — OCR
============================================================

OCR is secondary.

If OCR says something related to the bid item but the image shows
something else, trust the image.

Never classify RIGHT solely because a keyword appears in OCR.

============================================================
DECISION RULE
============================================================

DIRECT association:
    RIGHT

INDIRECT association:
    WRONG

NO association:
    WRONG

If evidence is insufficient:
    WRONG

If the decision requires an assumption:
    WRONG

Do NOT use "probably related" as a reason for RIGHT.

============================================================
CONFIDENCE
============================================================

HIGH:
The direct association is visually clear and unambiguous.

MEDIUM:
There is direct evidence, but some visual ambiguity exists.

LOW:
The evidence is unclear, incomplete, or difficult to interpret.

Important:

LOW confidence does NOT mean "probably RIGHT."

If direct evidence cannot be established, decision must be WRONG.

============================================================
REASON REQUIREMENT
============================================================

Your reason must describe the actual evidence.

Do not write generic statements such as:

"This is related to flooring and therefore supports carpet."

Instead explain:

1. what the region visibly contains;
2. what that drawing element represents;
3. whether it directly identifies the bid item.

Do not invent labels, tags, schedules, or symbols that are not visible.

============================================================
FINAL SELF-CHECK
============================================================

Before returning the result, ask:

1. What exactly is in the highlighted region?
2. What does it represent?
3. Would I identify it the same way if I did not know the bid item?
4. Is there DIRECT evidence connecting it to the bid item?
5. Am I relying on same-trade/same-sheet/specification reasoning?
6. Am I relying on OCR instead of the image?
7. Does my decision require an assumption?

If the answer to question 4 is NO,
decision MUST be WRONG.

Return the structured result.
"""


# ============================================================
# AZURE CLIENT
# ============================================================

def create_azure_client():

    load_dotenv()

    endpoint = os.getenv(
        "AZURE_OPENAI_ENDPOINT"
    )

    api_key = os.getenv(
        "AZURE_OPENAI_API_KEY"
    )

    deployment = os.getenv(
        "AZURE_OPENAI_DEPLOYMENT"
    )

    if (
        not endpoint
        or not api_key
        or not deployment
    ):

        raise RuntimeError(
            "Missing "
            "AZURE_OPENAI_ENDPOINT / "
            "AZURE_OPENAI_API_KEY / "
            "AZURE_OPENAI_DEPLOYMENT "
            "in .env"
        )

    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,
        base_url=(
            endpoint.rstrip("/")
            + "/openai/v1/"
        )
    )

    print(
        "Azure deployment:",
        deployment
    )

    return (
        client,
        deployment
    )


# ============================================================
# IMAGE
# ============================================================

def image_to_data_url(
    image_path
):

    with open(
        image_path,
        "rb"
    ) as f:

        encoded = (
            base64.b64encode(
                f.read()
            )
            .decode("utf-8")
        )

    return (
        "data:image/png;base64,"
        + encoded
    )


# ============================================================
# LLM CALL
# ============================================================

def call_llm(
    client,
    deployment,
    prompt,
    medium_path,
    local_path,
    locator_path
):

    schema = {

        "type": "object",

        "properties": {

            "region_type": {

                "type": "string",

                "enum": [

                    "MATERIAL_TAG",
                    "FINISH_TAG",
                    "ROOM_LABEL",
                    "GRID_REFERENCE",
                    "DIMENSION",
                    "SPECIFICATION_TEXT",
                    "GENERAL_NOTE",
                    "SCHEDULE",
                    "DETAIL",
                    "EQUIPMENT_TAG",
                    "PLAN_GEOMETRY",
                    "TITLE_BLOCK",
                    "OTHER",
                ],
            },

            "identified_content": {

                "type": "string"
            },

            "direct_bid_item_evidence": {

                "type": "string"
            },

            "association_strength": {

                "type": "string",

                "enum": [
                    "DIRECT",
                    "INDIRECT",
                    "NONE"
                ],
            },

            "decision": {

                "type": "string",

                "enum": [
                    "RIGHT",
                    "WRONG"
                ],
            },

            "confidence": {

                "type": "string",

                "enum": [
                    "HIGH",
                    "MEDIUM",
                    "LOW"
                ],
            },

            "reason": {

                "type": "string"
            },
        },

        "required": [

            "region_type",
            "identified_content",
            "direct_bid_item_evidence",
            "association_strength",
            "decision",
            "confidence",
            "reason",
        ],

        "additionalProperties": False,
    }

    medium_url = (
        image_to_data_url(
            medium_path
        )
    )

    local_url = (
        image_to_data_url(
            local_path
        )
    )

    locator_url = (
        image_to_data_url(
            locator_path
        )
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = client.responses.create(

                model=deployment,

                input=[{

                    "role": "user",

                    "content": [

                        {
                            "type": "input_text",
                            "text": prompt
                        },

                        {
                            "type": "input_text",
                            "text":
                                "IMAGE 1 — "
                                "TIGHT REGION CROP"
                        },

                        {
                            "type": "input_image",
                            "image_url":
                                medium_url
                        },

                        {
                            "type": "input_text",
                            "text":
                                "IMAGE 2 — "
                                "LOCAL DRAWING CONTEXT"
                        },

                        {
                            "type": "input_image",
                            "image_url":
                                local_url
                        },

                        {
                            "type": "input_text",
                            "text":
                                "IMAGE 3 — "
                                "WHOLE-SHEET LOCATOR. "
                                "The red box only marks the "
                                "audit region and is not part "
                                "of the original drawing."
                        },

                        {
                            "type": "input_image",
                            "image_url":
                                locator_url
                        },
                    ],
                }],

                text={
                    "format": {
                        "type":
                            "json_schema",

                        "name":
                            "drawing_audit",

                        "strict":
                            True,

                        "schema":
                            schema,
                    }
                },

                max_output_tokens=
                    MAX_OUTPUT_TOKENS,
            )

            if getattr(
                response,
                "status",
                None
            ) != "completed":

                reason = "unknown"

                try:

                    reason = (
                        response
                        .incomplete_details
                        .reason
                    )

                except Exception:
                    pass

                raise ValueError(
                    "Azure response incomplete: "
                    f"{reason}"
                )

            raw = (
                response
                .output_text
                .strip()
            )

            if not raw:
                raise ValueError(
                    "Azure returned empty output_text"
                )

            result = json.loads(
                raw
            )

            # ------------------------------------------------
            # Validate values
            # ------------------------------------------------

            if result.get(
                "decision"
            ) not in {
                "RIGHT",
                "WRONG"
            }:

                raise ValueError(
                    "Invalid decision: "
                    f"{result.get('decision')}"
                )

            if result.get(
                "confidence"
            ) not in {
                "HIGH",
                "MEDIUM",
                "LOW"
            }:

                raise ValueError(
                    "Invalid confidence: "
                    f"{result.get('confidence')}"
                )

            if result.get(
                "association_strength"
            ) not in {
                "DIRECT",
                "INDIRECT",
                "NONE"
            }:

                raise ValueError(
                    "Invalid association_strength: "
                    f"{result.get('association_strength')}"
                )

            # ------------------------------------------------
            # HARD SAFETY RULE
            #
            # Even if the LLM returns:
            #
            # association_strength = INDIRECT
            # decision = RIGHT
            #
            # we override it.
            # ------------------------------------------------

            if result[
                "association_strength"
            ] != "DIRECT":

                result["decision"] = (
                    "WRONG"
                )

            return {

                "llm_status":
                    "SUCCESS",

                "llm_region_type":
                    result[
                        "region_type"
                    ],

                "llm_identified_content":
                    result[
                        "identified_content"
                    ],

                "llm_direct_bid_item_evidence":
                    result[
                        "direct_bid_item_evidence"
                    ],

                "llm_association_strength":
                    result[
                        "association_strength"
                    ],

                "llm_decision":
                    result[
                        "decision"
                    ],

                "llm_confidence":
                    result[
                        "confidence"
                    ],

                "llm_reason":
                    result[
                        "reason"
                    ],

                "llm_error":
                    "",

                "llm_attempts":
                    attempt,
            }

        except Exception as e:

            print(
                f"  Attempt {attempt} failed: "
                f"{type(e).__name__}: {e}"
            )

            if attempt < MAX_RETRIES:

                time.sleep(
                    RETRY_DELAY
                )

    return {

        "llm_status":
            "ERROR",

        "llm_region_type":
            "",

        "llm_identified_content":
            "",

        "llm_direct_bid_item_evidence":
            "",

        "llm_association_strength":
            "NONE",

        "llm_decision":
            "LLM_ERROR",

        "llm_confidence":
            "LOW",

        "llm_reason":
            "",

        "llm_error":
            f"Failed after "
            f"{MAX_RETRIES} attempts",

        "llm_attempts":
            MAX_RETRIES,
    }


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(
    checkpoint_path,
    reset
):

    if (
        reset
        and checkpoint_path.exists()
    ):

        checkpoint_path.unlink()

        print(
            "Existing checkpoint deleted."
        )

    if not checkpoint_path.exists():

        return pd.DataFrame()

    return pd.read_csv(
        checkpoint_path
    )


def save_checkpoint(
    checkpoint_df,
    checkpoint_path
):

    checkpoint_df = (
        checkpoint_df
        .drop_duplicates(
            subset=[
                "row_index"
            ],
            keep="last"
        )
    )

    checkpoint_df.to_csv(
        checkpoint_path,
        index=False
    )


# ============================================================
# PHASE 3 — VALIDATION
# ============================================================

def phase3_validate(
    regions_df,
    legend,
    output_dir,
    max_regions,
    reset_checkpoint
):

    print("\n" + "=" * 70)
    print("PHASE 3 — AZURE LLM VALIDATION")
    print("=" * 70)

    client, deployment = (
        create_azure_client()
    )

    checkpoint_path = (
        output_dir /
        "phase3_checkpoint.csv"
    )

    checkpoint_df = (
        load_checkpoint(
            checkpoint_path,
            reset_checkpoint
        )
    )

    if checkpoint_df.empty:

        completed = set()

    else:

        done = checkpoint_df[
            checkpoint_df[
                "llm_decision"
            ].isin(
                [
                    "RIGHT",
                    "WRONG"
                ]
            )
        ]

        completed = set(
            done[
                "row_index"
            ]
            .astype(int)
        )

    print(
        f"Already validated: "
        f"{len(completed)}"
    )

    to_process = (
        regions_df[
            ~regions_df[
                "row_index"
            ].isin(completed)
        ]
        .copy()
    )

    if max_regions is not None:

        to_process = (
            to_process
            .head(max_regions)
            .copy()
        )

    print(
        "Remaining to process: "
        f"{len(to_process)}"
    )

    new_results = []

    for counter, (_, row) in enumerate(
        to_process.iterrows(),
        start=1
    ):

        print(
            f"\n[{counter}/{len(to_process)}] "
            f"{row['region_id']} | "
            f"{row['sheet_number']} | "
            f"{str(row['bid_item'])[:60]}"
        )

        medium_path = (
            row.get(
                "medium_path",
                ""
            )
        )

        local_path = (
            row.get(
                "local_path",
                ""
            )
        )

        locator_path = (
            row.get(
                "locator_path",
                ""
            )
        )

        # ----------------------------------------------------
        # Validate crop availability
        # ----------------------------------------------------

        if (
            not medium_path
            or not Path(
                medium_path
            ).exists()
            or not local_path
            or not Path(
                local_path
            ).exists()
            or not locator_path
            or not Path(
                locator_path
            ).exists()
        ):

            result = {

                "llm_status":
                    "ERROR",

                "llm_region_type":
                    "",

                "llm_identified_content":
                    "",

                "llm_direct_bid_item_evidence":
                    "",

                "llm_association_strength":
                    "NONE",

                "llm_decision":
                    "LLM_ERROR",

                "llm_confidence":
                    "LOW",

                "llm_reason":
                    "",

                "llm_error":
                    "One or more crop images "
                    "missing on disk",

                "llm_attempts":
                    0,
            }

        else:

            legend_entry = (
                find_legend_match(
                    row.get(
                        "text",
                        ""
                    ),
                    legend
                )
            )

            prompt = (
                build_llm_prompt(
                    row,
                    legend_entry
                )
            )

            result = call_llm(

                client,

                deployment,

                prompt,

                medium_path,

                local_path,

                locator_path
            )

        result.update({

            "row_index":
                int(
                    row[
                        "row_index"
                    ]
                ),

            "region_id":
                row[
                    "region_id"
                ],

            "sheet_number":
                row[
                    "sheet_number"
                ],

            "bid_item":
                row[
                    "bid_item"
                ],
        })

        new_results.append(
            result
        )

        print(
            "  -> "
            f"{result['llm_status']} / "
            f"{result['llm_decision']} / "
            f"{result['llm_confidence']} / "
            f"{result.get('llm_association_strength', '')}"
        )

        if result[
            "llm_error"
        ]:

            print(
                "  error: "
                f"{result['llm_error']}"
            )

        current_df = pd.DataFrame(
            new_results
        )

        if checkpoint_df.empty:

            combined = (
                current_df
            )

        else:

            combined = pd.concat(
                [
                    checkpoint_df,
                    current_df
                ],
                ignore_index=True
            )

        save_checkpoint(
            combined,
            checkpoint_path
        )

        if (
            counter
            < len(to_process)
        ):

            time.sleep(
                REQUEST_DELAY
            )

    if checkpoint_path.exists():

        return pd.read_csv(
            checkpoint_path
        )

    return pd.DataFrame(
        new_results
    )


# ============================================================
# MERGE RESULTS
# ============================================================

def merge_results(
    regions_df,
    llm_results_df
):

    regions_df = (
        regions_df.copy()
    )

    defaults = {

        "llm_status":
            "NOT_PROCESSED",

        "llm_region_type":
            "",

        "llm_identified_content":
            "",

        "llm_direct_bid_item_evidence":
            "",

        "llm_association_strength":
            "",

        "llm_decision":
            "NOT_PROCESSED",

        "llm_confidence":
            "",

        "llm_reason":
            "",

        "llm_error":
            "",

        "llm_attempts":
            0,
    }

    if llm_results_df.empty:

        for col, default in defaults.items():

            regions_df[
                col
            ] = default

    else:

        results = (
            llm_results_df
            .drop_duplicates(
                subset=[
                    "row_index"
                ],
                keep="last"
            )
        )

        merge_cols = (
            ["row_index"]
            + [
                c
                for c in defaults
                if c in results.columns
            ]
        )

        regions_df = (
            regions_df.merge(
                results[
                    merge_cols
                ],
                on="row_index",
                how="left"
            )
        )

        for col, default in defaults.items():

            if col not in regions_df.columns:

                regions_df[
                    col
                ] = default

            elif default != 0:

                regions_df[
                    col
                ] = (
                    regions_df[
                        col
                    ]
                    .fillna(default)
                )

        regions_df[
            "llm_attempts"
        ] = (
            pd.to_numeric(
                regions_df[
                    "llm_attempts"
                ],
                errors="coerce"
            )
            .fillna(0)
            .astype(int)
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    def status(row):

        decision = clean_value(
            row[
                "llm_decision"
            ]
        )

        association = clean_value(
            row[
                "llm_association_strength"
            ]
        )

        # Hard rule:
        # Only DIRECT association can survive as RIGHT.
        if (
            decision == "RIGHT"
            and association == "DIRECT"
        ):
            return "RIGHT"

        if decision == "WRONG":
            return "WRONG"

        if (
            decision == "RIGHT"
            and association != "DIRECT"
        ):
            return "WRONG"

        if decision == "LLM_ERROR":
            return "LLM_ERROR"

        return "NOT_PROCESSED"

    regions_df[
        "final_validation"
    ] = regions_df.apply(
        status,
        axis=1
    )

    # --------------------------------------------------------
    # Manual review
    # --------------------------------------------------------

    regions_df[
        "needs_manual_review"
    ] = (

        regions_df[
            "llm_confidence"
        ].isin(
            REVIEW_QUEUE_CONFIDENCE
        )

        | (
            regions_df[
                "final_validation"
            ]
            == "LLM_ERROR"
        )

        | (
            regions_df[
                "final_validation"
            ]
            == "NOT_PROCESSED"
        )

        | regions_df[
            "frequency_flag"
        ]

        | (
            regions_df[
                "llm_association_strength"
            ]
            == "INDIRECT"
        )
    )

    return regions_df


# ============================================================
# PHASE 4 — CORRECTED JSON
# ============================================================

def build_corrected_json(
    original_data,
    judged_df,
    locators
):

    """
    An item survives if it has at least one surviving region.

    WRONG regions are removed.

    NOT_PROCESSED / LLM_ERROR regions are retained.

    Reference sheets remain untouched.
    """

    corrected = json.loads(
        json.dumps(
            original_data
        )
    )

    results_by_row = {

        int(r["row_index"]):
            r

        for _, r
        in judged_df.iterrows()
    }

    items_map = defaultdict(
        list
    )

    for loc in locators:

        key = (

            loc[
                "drawing_index"
            ],

            loc[
                "scope_key"
            ],

            loc[
                "item_index"
            ],
        )

        items_map[
            key
        ].append(
            loc
        )

    drawings = corrected[
        "drawings_processing_results"
    ]

    removed_items_log = []

    removed_regions_log = []

    for (
        drawing_index,
        scope_key,
        item_index
    ), region_locators in items_map.items():

        try:

            drawing = drawings[
                drawing_index
            ]

        except IndexError:

            continue

        if (
            drawing.get(
                "sheet_number"
            )
            in REFERENCE_SHEETS
        ):

            continue

        try:

            item = (
                drawing[
                    "bid_items"
                ][
                    scope_key
                ][
                    item_index
                ]
            )

        except (
            KeyError,
            IndexError,
            TypeError
        ):

            continue

        original_regions = item.get(
            "regions",
            []
        )

        kept_regions = []

        for loc in region_locators:

            region_index = loc[
                "region_index"
            ]

            row_index = loc[
                "row_index"
            ]

            try:

                region = (
                    original_regions[
                        region_index
                    ]
                )

            except IndexError:

                continue

            row = results_by_row.get(
                row_index
            )

            if row is None:

                region[
                    "validation"
                ] = {

                    "decision":
                        "NOT_PROCESSED"
                }

                kept_regions.append(
                    region
                )

                continue

            decision = clean_value(
                row[
                    "final_validation"
                ]
            )

            # ------------------------------------------------
            # Remove WRONG
            # ------------------------------------------------

            if decision == "WRONG":

                removed_regions_log.append({

                    "region_id":
                        row.get(
                            "region_id",
                            ""
                        ),

                    "sheet_number":
                        drawing.get(
                            "sheet_number"
                        ),

                    "scope_key":
                        scope_key,

                    "bid_item":
                        item.get(
                            "bid_item",
                            ""
                        ),

                    "region_text":
                        row.get(
                            "text",
                            ""
                        ),

                    "region_type":
                        row.get(
                            "llm_region_type",
                            ""
                        ),

                    "association_strength":
                        row.get(
                            "llm_association_strength",
                            ""
                        ),

                    "identified_content":
                        row.get(
                            "llm_identified_content",
                            ""
                        ),

                    "direct_bid_item_evidence":
                        row.get(
                            "llm_direct_bid_item_evidence",
                            ""
                        ),

                    "reason":
                        clean_value(
                            row.get(
                                "llm_reason",
                                ""
                            )
                        ),
                })

                continue

            # ------------------------------------------------
            # Keep RIGHT / unprocessed
            # ------------------------------------------------

            region[
                "validation"
            ] = {

                "decision":
                    decision,

                "confidence":
                    clean_value(
                        row.get(
                            "llm_confidence",
                            ""
                        )
                    ),

                "region_type":
                    clean_value(
                        row.get(
                            "llm_region_type",
                            ""
                        )
                    ),

                "identified_content":
                    clean_value(
                        row.get(
                            "llm_identified_content",
                            ""
                        )
                    ),

                "association_strength":
                    clean_value(
                        row.get(
                            "llm_association_strength",
                            ""
                        )
                    ),

                "direct_bid_item_evidence":
                    clean_value(
                        row.get(
                            "llm_direct_bid_item_evidence",
                            ""
                        )
                    ),

                "reason":
                    clean_value(
                        row.get(
                            "llm_reason",
                            ""
                        )
                    ),

                "llm_status":
                    clean_value(
                        row.get(
                            "llm_status",
                            ""
                        )
                    ),

                "region_id":
                    clean_value(
                        row.get(
                            "region_id",
                            ""
                        )
                    ),
            }

            kept_regions.append(
                region
            )

        item[
            "regions"
        ] = kept_regions

    # --------------------------------------------------------
    # Remove items with zero surviving regions
    # --------------------------------------------------------

    for drawing in drawings:

        if (
            drawing.get(
                "sheet_number"
            )
            in REFERENCE_SHEETS
        ):

            continue

        bid_items = drawing.get(
            "bid_items",
            {}
        )

        if not isinstance(
            bid_items,
            dict
        ):

            continue

        for scope_key, items in list(
            bid_items.items()
        ):

            if not isinstance(
                items,
                list
            ):

                continue

            surviving = []

            for item in items:

                if len(
                    item.get(
                        "regions",
                        []
                    )
                ) == 0:

                    removed_items_log.append({

                        "sheet_number":
                            drawing.get(
                                "sheet_number"
                            ),

                        "scope_key":
                            scope_key,

                        "bid_item":
                            item.get(
                                "bid_item",
                                ""
                            ),
                    })

                    continue

                surviving.append(
                    item
                )

            bid_items[
                scope_key
            ] = surviving

    print(
        "\nRegions removed (WRONG): "
        f"{len(removed_regions_log)}"
    )

    print(
        "Items dropped entirely "
        "(0 regions left): "
        f"{len(removed_items_log)}"
    )

    return (
        corrected,
        removed_items_log,
        removed_regions_log
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    judged_df,
    output_dir
):

    audit_columns = [

        "region_id",

        "sheet_number",

        "sheet_name",

        "scope",

        "bid_item",

        "subheading",

        "text",

        "text_frequency",

        "frequency_flag",

        "ocr_hints",

        "llm_region_type",

        "llm_identified_content",

        "llm_association_strength",

        "llm_direct_bid_item_evidence",

        "final_validation",

        "llm_confidence",

        "llm_reason",

        "needs_manual_review",

        "medium_path",

        "local_path",

        "locator_path",
    ]

    available = [
        c
        for c in audit_columns
        if c in judged_df.columns
    ]

    audit_csv = (
        output_dir /
        "audit_findings.csv"
    )

    judged_df[
        available
    ].to_csv(
        audit_csv,
        index=False
    )

    review_csv = (
        output_dir /
        "manual_review_queue.csv"
    )

    judged_df[
        judged_df[
            "needs_manual_review"
        ]
    ][
        available
    ].to_csv(
        review_csv,
        index=False
    )

    print(
        f"Audit CSV: {audit_csv}"
    )

    print(
        "Manual review queue "
        f"({judged_df['needs_manual_review'].sum()} rows): "
        f"{review_csv}"
    )

    # --------------------------------------------------------
    # Additional diagnostic CSV
    # --------------------------------------------------------

    diagnostics = (
        judged_df[
            [
                "region_id",
                "sheet_number",
                "bid_item",
                "text",
                "llm_region_type",
                "llm_association_strength",
                "final_validation",
                "llm_confidence",
                "llm_reason",
            ]
        ]
    )

    diagnostic_csv = (
        output_dir /
        "audit_diagnostics.csv"
    )

    diagnostics.to_csv(
        diagnostic_csv,
        index=False
    )

    print(
        f"Diagnostic CSV: {diagnostic_csv}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "total_regions":
            int(len(judged_df)),

        "right":
            int(
                (
                    judged_df[
                        "final_validation"
                    ]
                    == "RIGHT"
                ).sum()
            ),

        "wrong":
            int(
                (
                    judged_df[
                        "final_validation"
                    ]
                    == "WRONG"
                ).sum()
            ),

        "llm_error":
            int(
                (
                    judged_df[
                        "final_validation"
                    ]
                    == "LLM_ERROR"
                ).sum()
            ),

        "not_processed":
            int(
                (
                    judged_df[
                        "final_validation"
                    ]
                    == "NOT_PROCESSED"
                ).sum()
            ),

        "manual_review":
            int(
                judged_df[
                    "needs_manual_review"
                ].sum()
            ),

        "direct_association":
            int(
                (
                    judged_df[
                        "llm_association_strength"
                    ]
                    == "DIRECT"
                ).sum()
            ),

        "indirect_association":
            int(
                (
                    judged_df[
                        "llm_association_strength"
                    ]
                    == "INDIRECT"
                ).sum()
            ),

        "no_association":
            int(
                (
                    judged_df[
                        "llm_association_strength"
                    ]
                    == "NONE"
                ).sum()
            ),
    }

    summary_path = (
        output_dir /
        "audit_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    print(
        f"Summary: {summary_path}"
    )

    return (
        audit_csv,
        review_csv
    )


# ============================================================
# MAIN RUN COMMAND
# ============================================================

def run_command(args):

    results_path = (
        Path(
            args.results
        )
        .resolve()
    )

    pdf_path = (
        Path(
            args.pdf
        )
        .resolve()
    )

    output_dir = (
        Path(
            args.output
        )
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print(
        "CONSTRUCTION DRAWING AUDIT PIPELINE v3"
    )
    print("=" * 70)

    print(
        "Results:",
        results_path
    )

    print(
        "PDF:",
        pdf_path
    )

    print(
        "Output dir:",
        output_dir
    )

    # --------------------------------------------------------
    # Phase 1
    # --------------------------------------------------------

    (
        original_data,
        regions_df,
        locators
    ) = phase1_extract(
        results_path,
        output_dir
    )

    # --------------------------------------------------------
    # Phase 2
    # --------------------------------------------------------

    phase2_df = (
        phase2_create_crops(
            regions_df,
            pdf_path,
            output_dir
        )
    )

    # --------------------------------------------------------
    # Phase 2.6
    # --------------------------------------------------------

    phase2_df = (
        phase2_6_frequency_scan(
            phase2_df,
            output_dir
        )
    )

    # --------------------------------------------------------
    # Phase 2.7
    # --------------------------------------------------------

    phase2_df = (
        phase2_7_add_ocr_hints(
            phase2_df,
            output_dir
        )
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend = load_legend(
        args.legend
    )

    # --------------------------------------------------------
    # Phase 3
    # --------------------------------------------------------

    llm_results_df = (
        phase3_validate(
            phase2_df,
            legend,
            output_dir,
            args.max_regions,
            args.reset_checkpoint
        )
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    judged_df = (
        merge_results(
            phase2_df,
            llm_results_df
        )
    )

    # --------------------------------------------------------
    # Save audit
    # --------------------------------------------------------

    save_outputs(
        judged_df,
        output_dir
    )

    # --------------------------------------------------------
    # Correct JSON
    # --------------------------------------------------------

    (
        corrected_data,
        removed_items,
        removed_regions
    ) = build_corrected_json(

        original_data,

        judged_df,

        locators
    )

    corrected_path = (
        output_dir /
        "corrected_results.json"
    )

    with open(
        corrected_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            json_safe(
                corrected_data
            ),
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Removed item log
    # --------------------------------------------------------

    removed_items_path = (
        output_dir /
        "removed_items_log.csv"
    )

    pd.DataFrame(
        removed_items
    ).to_csv(
        removed_items_path,
        index=False
    )

    # --------------------------------------------------------
    # Removed region log
    # --------------------------------------------------------

    removed_regions_path = (
        output_dir /
        "removed_regions_log.csv"
    )

    pd.DataFrame(
        removed_regions
    ).to_csv(
        removed_regions_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        "Corrected JSON:",
        corrected_path
    )

    print(
        "\nFinal validation counts:"
    )

    print(
        judged_df[
            "final_validation"
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\nAssociation counts:"
    )

    print(
        judged_df[
            "llm_association_strength"
        ]
        .value_counts(
            dropna=False
        )
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Construction drawing "
            "bid-item audit pipeline"
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True
    )

    # --------------------------------------------------------
    # Legend extraction
    # --------------------------------------------------------

    p_legend = sub.add_parser(
        "legend-extract",
        help=(
            "Dump reference-sheet text "
            "for manual legend building"
        )
    )

    p_legend.add_argument(
        "--pdf",
        required=True
    )

    p_legend.add_argument(
        "--results",
        required=True
    )

    p_legend.add_argument(
        "--output",
        default="legend_dump.json"
    )

    # --------------------------------------------------------
    # Full run
    # --------------------------------------------------------

    p_run = sub.add_parser(
        "run",
        help=(
            "Run the full audit pipeline"
        )
    )

    p_run.add_argument(
        "--results",
        required=True
    )

    p_run.add_argument(
        "--pdf",
        required=True
    )

    p_run.add_argument(
        "--legend",
        default=None,
        help="Path to hand-built legend.json"
    )

    p_run.add_argument(
        "--output",
        default="audit_output"
    )

    p_run.add_argument(
        "--max-regions",
        type=int,
        default=None
    )

    p_run.add_argument(
        "--reset-checkpoint",
        action="store_true"
    )

    args = parser.parse_args()

    if args.command == "legend-extract":

        legend_extract_command(
            Path(args.pdf),
            Path(args.results),
            Path(args.output)
        )

    elif args.command == "run":

        run_command(
            args
        )


if __name__ == "__main__":
    main()