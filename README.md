# Drawing Audit Pipeline — Verified Run Notes

This code was tested against your actual `drawings.pdf` / `results.json`

in a sandboxed environment before delivery. What was and wasn't

verified, honestly:

## Verified end-to-end against your real files

- Phase 1 extraction: confirmed 3,342 total regions / 323 items /

  204 regions on the 3 reference sheets, matching the numbers you'd

  get by inspecting the JSON directly.

- Phase 2 rendering + cropping: ran on 1,920 of the 3,138 judgeable

  regions (ran out of time in the sandbox for the rest — see below),

  zero errors, including sheets 12-15 which are `/Rotate 270` with a

  portrait MediaBox. Manually confirmed by eye that a crop pulled from

  the rotated CU300 sheet contains exactly the text recorded for that

  region, and that a locator thumbnail correctly boxes the same region

  on the full sheet.

- Frequency pre-pass: ran on real data. Found that "BR1" and "FB1"

  (real legend-defined wall assembly codes) appear 17x/18x — right in

  the same range as clearly non-work tags. This is the concrete case

  for why the frequency flag is a **review hint**, not an auto-delete

  rule — and is worth mentioning on the call as a specific example you

  found in the data.

- Legend extraction: ran against the actual G-002.3/G-501/G-511 pages.

  Confirmed these are visual wall-assembly diagrams (not a clean text

  table) — plain text extraction gets you raw words near the diagrams,

  not a verified mapping. You need to build `legend.json` by hand (or

  with LLM help you verify by eye) using `legend_dump.json` and the

  three rendered legend page images as your source.

- JSON filtering (Phase 4): tested end-to-end with realistic mocked

  verdicts (not real LLM output) on the 1,920-region subset. Confirmed:

  top-level structure and all three reference sheets are byte-identical

  to the original; surviving items/regions keep every original field

  name (`quad_px`, `text`, etc.) with only a `validation` block added;

  WRONG regions are stripped; items left with zero regions are dropped

  entirely.

## NOT verified — you need to do this yourself before the call

- The actual Azure OpenAI call (`call_llm` / Phase 3). No API key was

  available in this sandbox. The request structure follows the

  pattern from your original working code, but you have not seen it

  run and neither have I. Test it on `--max-regions 20` first.

- The remaining ~1,218 of 3,138 regions never got their crops

  generated in this sandbox (hit a compute time limit, not a bug —

  the code was mid-run and caches to disk as it goes, so if you point

  it at the same output directory it resumes automatically, or just

  run it fresh on your own machine).

- Running the full 3,138-region batch through the LLM — cost and time

  at scale are untested. Budget for this.

- The "new sheet set" the interviewer will run it against on the call

  — the `find_pdf_page_index` fallback (search page text for the sheet

  number) is there for exactly this case but has not been exercised

  against different files.

## What to do first

1. `python drawing_audit.py legend-extract --pdf drawings.pdf --results results.json`

   then build `legend.json` by hand using the dump + rendered images.

2. `python drawing_audit.py run --results results.json --pdf drawings.pdf --legend legend.json --max-regions 20 --reset-checkpoint`

   and actually read the output — confirm the verdicts, confidence

   levels, association decisions, and reasoning make sense before

   trusting it on the full set.

3. Run the full set. Expect it to take a while and cost real API

   calls — 3,138 regions x 2 images each.

4. Review `manual_review_queue.csv` (LOW/MEDIUM confidence, errors,

   frequency-flagged items) yourself — this is the human-in-the-loop

   quality-control step. \*\*Manual review currently does not modify the

   LLM decision or corrected JSON automatically; it is used to assess

   and identify results that may require further action.\*\*
