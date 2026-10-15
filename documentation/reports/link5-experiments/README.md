# Link5 experiment review

Open `index.html`, or serve the repository root and visit:
`http://localhost:8765/documentation/reports/link5-experiments/`.

The report follows the light styling of the published PDI review and is staged locally, not published. Existing experiment evidence is referenced through repository-relative URLs; keep this report inside the repository. It does not copy or modify scientific artifacts.

The report contains the requested historical four-way Link5 AUROC (0.669580), a recomputation on the new experiment's 32 binary-labeled case identities (0.643750), balanced_v0 (0.708333), and refine_v1 (0.779167). All use the same v2 workbook AB labels; binary AUROC excludes 0.5. Cross-run comparisons also change masks, geometry, and support. Only balanced_v0 versus refine_v1 isolates graph selection in this retained experiment.

Method pictures are screenshots of `method-replay.html`, with `kind=pair|depth|erosion`, `version=old|new`, and `capture=1`. The depth cloud comparison uses identical retained tracked XYZ samples, not unavailable raw dense geometry. Source overlays show full saved pixel rejection maps. The pair pictures use actual historical naive and current refined graphs. The original 82 case replays and six seeded random Simple3D examples remain interactive.

Rebuild data and flowchart SVGs with the NumPy/OpenCV environment:

```sh
/tmp/link5-report-browser-env/bin/python documentation/publication/builders/build_link5_experiment_report.py
```

Render method pictures and verify the report with:

```sh
/tmp/link5-report-browser-env/bin/python documentation/publication/builders/verify_link5_experiment_report.py
```

Browser verification requires Playwright and an installed Chromium. The verification script starts a temporary local server when needed and writes screenshots and checks to `assets/qa/`. Exact input hashes, selection policy, comparison scope, and AUROCs are in `assets/provenance.json`.

The pair-selection rationale includes actual Link5 crops from COSMOS2.5_0054, frames 0, 23, 46, and 69. Each uses the same 547 × 340 source-pixel rectangle centered on the retained Link5 mask, with no resizing or rotation. `assets/pair-rationale.json` records crop coordinates, frame times, and video/mask/image hashes. The data builder regenerates these crops.

The scoring-pair section uses absolute counts from retained `pair_frames`: each used pair at each fresh-scored frame counts once. Reference and carry frames are excluded. Link2 uses 44 scored videos from `robot/results/selected-45-v1`; Link5 refine_v1 uses all 41 retained cases. Full-cohort totals are shown, with 39 matched video identities and per-video counts available in expandable details. Rebuild with `documentation/publication/builders/build_link5_pair_availability.py` (also called by the main builder). The architecture SVGs mark revised masking, Link5 depth support, and refined pair selection in purple with explicit REFINED labels.
