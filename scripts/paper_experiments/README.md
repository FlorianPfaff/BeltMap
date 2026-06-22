# Paper experiment scripts

Put dataset-specific, manuscript-specific, or evidence-freezing scripts here.

Scripts in this directory may intentionally reference concrete datasets, frame ranges, output directories, paper evidence filenames, or figure names. They are reproducibility helpers for a specific analysis, not stable package APIs.

A script belongs here if it:

- encodes a dataset such as Brick 20 g/s or Brick/Sand-lime 50:50 10 g/s;
- writes into `data/paper_evidence/`, `paper_notes/`, or a manuscript figure directory;
- freezes a paper table, contact sheet, ablation, or artifact path;
- contains one-off thresholds or frame IDs chosen for a paper experiment.

Keep the script header explicit about:

- required inputs,
- expected outputs,
- whether outputs are citable, exploratory, blocked, or superseded,
- the label status required before running metrics.

If a helper becomes generally useful, promote the reusable part into `beltmap/` and expose it through `beltmap/cli/` with tests and documentation.
