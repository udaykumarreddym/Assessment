# Construction Drawing Bid-Item Audit Pipeline

## Overview

This project audits the association between **construction drawing regions** and their corresponding **bid items**.

The input `results.json` contains regions that have already been associated with bid items. The purpose of this pipeline is to independently verify whether those associations are correct by locating each region in the original `drawings.pdf` and using visual context, project references, OCR information, and a multimodal LLM.

The pipeline is designed to identify cases where a region has been incorrectly associated with a bid item and produce a cleaned `corrected_results.json`.

The main principle is:

> **The actual drawing region and its visual context are the primary evidence for determining whether a bid-item association is correct.**

Supporting information such as legends, OCR, specifications, and repeated text is used to improve interpretation but is not treated as sufficient evidence by itself.

---

## Input

The pipeline requires:

- `drawings.pdf` — construction drawing document.
- `results.json` — original region and bid-item associations.
- `legend.json` — project-specific legend information extracted from reference sheets.
- Azure OpenAI credentials for multimodal validation.

The original `results.json` contains information such as:

- Drawing/sheet information
- Bid items
- Regions
- Region coordinates
- Extracted text
- Specifications
- References
- Notes

---

## Pipeline

```text
results.json
      │
      ▼
1. Extract & Flatten Regions
      │
      ▼
2. Map Regions to PDF
      │
      ▼
3. Generate Drawing Crops
   ├── Medium region crop
   └── Whole-sheet locator
      │
      ▼
4. Extract / Use Reference Information
      │
      ▼
5. OCR Frequency Pre-pass
      │
      ▼
6. Azure OpenAI Multimodal Audit
      │
      ├── Identify the region
      ├── Determine association
      ├── Assign confidence
      └── Provide reasoning
      │
      ▼
7. Apply Audit Decision
      │
      ├── DIRECT → RIGHT
      ├── INDIRECT → WRONG
      └── NONE → WRONG
      │
      ▼
8. Generate Outputs
```

---

## 1. Region Extraction

The first phase loads and flattens `results.json` into a region-level dataset.

For each region, the pipeline retains information such as:

- Region ID
- Sheet number
- Sheet name
- Discipline
- Scope
- Bid item
- Region text
- Region coordinates
- Specifications
- References
- Drawing reference details
- Notes

Reference sheets are identified separately and excluded from normal bid-item judging.

For the provided project:

- **3,342 total regions**
- **323 bid items**
- **204 regions on the 3 reference sheets**
- **3,138 regions available for judging**

---

## 2. Mapping Regions to the PDF

Each region contains coordinate information (`quad_px`).

The pipeline maps these coordinates to the corresponding page in `drawings.pdf`.

The PDF rendering uses `pypdfium2` and handles rotated pages. The rendered page is normalized to the expected drawing dimensions before the region coordinates are used.

This allows the pipeline to locate the exact region in the original drawing instead of relying only on extracted text.

---

## 3. Creating Visual Context

For every judgeable region, the pipeline creates two visual representations.

### Medium Crop

A crop around the region provides detailed information about:

- Text
- Symbols
- Tags
- Notes
- Callouts
- Drawing elements

### Whole-Sheet Locator

A whole-sheet image is created with the target region highlighted.

This provides positional context, allowing the LLM to determine whether the region is, for example:

- Inside a room
- Along a grid line
- Part of a schedule
- Near a detail
- On the drawing perimeter
- Part of a title block

Both images are provided to the multimodal model.

---

## 4. Reference and Legend Information

The project contains reference sheets such as:

- `G-002.3`
- `G-501`
- `G-511`

These sheets contain project-specific information such as wall assembly codes and other drawing conventions.

The pipeline provides a `legend-extract` command to extract the available information from these sheets.

Because the reference sheets contain visual diagrams rather than simple text tables, the extracted text is treated as a starting point. The final `legend.json` is verified using the extracted information and rendered reference-sheet images.

Example:

```json
{
  "BR1": {
    "meaning": "Brick veneer wall assembly",
    "source_sheet": "G-501",
    "package": "Masonry"
  }
}
```

Legend information helps the LLM understand project-specific codes that may otherwise be ambiguous.

---

## 5. OCR Frequency Pre-pass

The pipeline analyzes the frequency of extracted region text.

Repeated short strings can sometimes indicate:

- Grid references
- Repeated drawing tags
- Plant/tree codes
- Schedule references
- Material or assembly codes

A frequency threshold is used to flag repeated text.

However, frequency is **not used as an automatic rejection rule**.

For example, `BR1` and `FB1` appear repeatedly but are legitimate legend-defined wall assembly codes.

Therefore:

> **Frequency is a review hint, not a verdict.**

---

## 6. Azure OpenAI Multimodal Validation

The main audit is performed using Azure OpenAI.

The model receives:

1. The region's drawing crop.
2. The whole-sheet locator.
3. Relevant legend information.
4. OCR and supporting extracted information.
5. The original bid-item association.

The original association may be incorrect, so the model is explicitly instructed **not to assume that the original bid item is correct**.

### Region-First Reasoning

The model is first required to determine what the highlighted region represents.

Examples include:

- Material tag
- Finish tag
- Room label
- Grid reference
- Dimension
- Specification text
- General note
- Schedule
- Detail
- Equipment tag
- Plan geometry
- Title block

Only after identifying the region does the model evaluate the bid-item association.

This reduces the risk of the original bid item influencing the interpretation of the drawing.

---

## 7. Association Classification

The model evaluates the relationship between the identified drawing region and the original bid item.

The association is interpreted as:

### DIRECT

The highlighted drawing region itself provides clear evidence supporting the bid item.

### INDIRECT

The region may be related to the bid item, but accepting the relationship requires additional assumptions or interpretation.

### NONE

The region does not provide meaningful evidence supporting the bid item.

The current automated decision rule is conservative:

```text
DIRECT   → RIGHT
INDIRECT → WRONG
NONE     → WRONG
```

A matching keyword, material name, discipline, scope, sheet, or specification division alone is not sufficient to classify a region as `RIGHT`.

---

## 8. Confidence

The LLM also provides a confidence level:

- `HIGH`
- `MEDIUM`
- `LOW`

Confidence represents how certain the model is about its assessment.

Low- and medium-confidence results are added to the manual review queue.

Confidence is treated as a **review signal**, not as proof that a decision is correct.

---

## 9. Corrected Results

After validation, the pipeline generates `corrected_results.json`.

Regions classified as `WRONG` are removed from their corresponding bid items.

If a bid item has no regions remaining, that bid item is removed.

Regions that are `NOT_PROCESSED` or have an LLM error are not silently deleted and remain available for follow-up.

The original region information is preserved for surviving regions, with a `validation` block added containing the audit information.

---

## 10. Manual Review

The pipeline creates `manual_review_queue.csv` for human inspection.

Regions are flagged for manual review when they have:

- `LOW` confidence
- `MEDIUM` confidence
- LLM errors
- Regions that were not processed
- Frequency flags

Manual review acts as a **human-in-the-loop quality-control step**.

The current implementation does not automatically modify the LLM decision based on manual review. The manual review is used to assess uncertain results and identify cases that may require further action.

---

## Outputs

The pipeline produces several outputs:

| Output | Purpose |
|---|---|
| `phase1_regions.csv` | Flattened region dataset |
| `phase2_regions.csv` | Region/PDF mapping and crop information |
| `phase2_6_frequency_flags.csv` | Repeated-text review hints |
| `phase3_checkpoint.csv` | LLM processing checkpoint |
| `audit_findings.csv` | Complete audit results |
| `manual_review_queue.csv` | Results requiring human inspection |
| `corrected_results.json` | Cleaned bid-item/region associations |
| `removed_regions_log.csv` | Regions removed as WRONG |
| `removed_items_log.csv` | Bid items removed because no regions remained |
| `legend_dump.json` | Extracted reference-sheet information |

---

## What Was Automated

The pipeline automates:

1. JSON flattening and region extraction.
2. Reference-sheet exclusion.
3. PDF coordinate mapping.
4. PDF rendering.
5. Region and context image generation.
6. Page caching.
7. OCR analysis.
8. Frequency analysis.
9. Legend matching.
10. Azure OpenAI multimodal validation.
11. LLM retries and checkpointing.
12. Structured response validation.
13. Confidence-based review flagging.
14. Bid-item/region merging.
15. Removal of incorrect regions.
16. Generation of corrected JSON and audit logs.

---

## What Was Checked Manually

I manually checked:

1. The structure of `results.json` and the drawing before relying on coordinate mapping.
2. PDF rendering and region alignment on rotated sheets.
3. Reference and legend sheets.
4. Legend information extracted from the reference sheets.
5. LLM results through the manual-review queue.

---

## Limitations

### Indirect Relationships

Some legitimate construction relationships may not be directly represented in the highlighted region.

Because the current method requires direct evidence for an automatic `RIGHT` decision, some valid indirect associations may be classified as `WRONG`.

### Visual Understanding

Small text, dense drawings, overlapping annotations, and complex linework may affect the LLM's ability to interpret a region correctly.

### OCR

OCR can be affected by:

- Small text
- Rotated text
- Low contrast
- Complex drawings
- Symbols
- Colored text

Therefore, OCR is supporting information rather than the primary source of truth.

### Project-Specific Information

Different projects may use different:

- Coordinate systems
- Sheet structures
- Legends
- Abbreviations
- Material codes
- Drawing conventions

The pipeline may therefore require project-specific configuration when applied to another dataset.

---

## Verified Testing

The non-LLM portions of the pipeline were tested against the actual project files.

The following were verified:

- **3,342 regions / 323 bid items**
- **3 reference sheets**
- PDF coordinate mapping
- Rotated-page handling
- Region cropping
- Whole-sheet locator generation
- Frequency analysis
- Legend extraction
- JSON filtering and corrected-output generation

The actual Azure OpenAI call and full-scale LLM processing require execution with valid Azure credentials.

---

## Running the Pipeline

### 1. Extract Legend Information

```bash
python drawing_audit.py legend-extract \
    --pdf drawings.pdf \
    --results results.json
```

Review the generated legend information and create `legend.json`.

### 2. Test the LLM on a Small Sample

```bash
python drawing_audit.py run \
    --results results.json \
    --pdf drawings.pdf \
    --legend legend.json \
    --max-regions 20 \
    --reset-checkpoint
```

Review:

- Decision
- Association strength
- Confidence
- Reasoning
- Visual evidence

before processing the complete dataset.

### 3. Run the Full Audit

```bash
python drawing_audit.py run \
    --results results.json \
    --pdf drawings.pdf \
    --legend legend.json
```

The checkpoint allows the pipeline to resume without reprocessing completed regions.

---

## Azure Configuration

Create a `.env` file containing:

```env
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=...
```

---

## Key Design Principle

The pipeline is designed around a simple rule:

> **Do not assume that an extracted region belongs to its original bid item. Verify the association from the actual drawing evidence.**

The combination of **visual context + project references + OCR + multimodal LLM reasoning + confidence-based human review** provides a structured approach for auditing the original region-to-bid-item associations.
