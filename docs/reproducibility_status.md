# Reproducibility Status

This document describes the local review release accompanying "Collaborative Carrier-Worker Scheduling in Marsupial Robotic Systems via Deep Reinforcement Learning". It records the assets that are present and the provenance questions that remain open. Status was checked on 2026-09-14; this release has not been synchronized to an anonymous remote.

## Training Configuration Provenance

Checkpoint `args.json` files are original records and have not been rewritten to match the manuscript.

| Method | Training scale | Recorded batch size | Recorded seed | Epochs |
| --- | --- | --- | --- | --- |
| MaRS-Net | 10, 20, 40, 60 | 2048 | 1234 | 100 |
| MaRS-Net | 30 | 4000 | 1234 | 100 |
| MaRS-Net | 100 | 1024 | 1234 | 100 |
| HDRL | 10, 20, 30, 40, 60, 100 | 1024 | 1234 | 100 |
| TDRL | 10, 20, 30, 40, 60, 100 | 1024 | 1234 | 100 |

The manuscript states a common batch size of 1024 and scale-specific seeds `1234+n`. Those settings differ from the released metadata. All 18 released weight files and their configuration files are byte-identical to the corresponding files in the locally retained baseline experiment collection. This establishes copy provenance, but does not independently establish which run produced every manuscript table entry. Available training log excerpts also contain different launch configurations, so run identity and completed-run records must be matched before drawing conclusions from an excerpt.

A locally retained synthetic summary table contains the manuscript's MaRS-Net objective values (for example, Sample-1280 means 1254.31 at n=20 and 2457.36 at n=40). It is not packaged here and does not resolve the training-setting discrepancy. That table also labels metaheuristic results as best-of-five repetitions, with the selected run's time, while the manuscript describes the iteration/time stopping rule without this aggregation detail. The experiment protocol needs reconciliation before exact table reproduction is claimed.

Do not change the recorded seeds or batch sizes to remove these discrepancies. The unresolved step is to match original runs, checkpoints, and evaluation records, then either supply the intended weights or correct the description of the experiment that actually ran.

## Instance Generation and Evaluation Details

The released on-the-fly training generators sample processing durations uniformly from the discrete set `{60,65,...,100}`, whereas the manuscript writes a continuous uniform distribution on `[60,100]`. Deadline generation uses `300 + 40*j + Uniform(-40,40)` for `j=0,...,n-1`, followed by a random permutation. The manuscript uses task indices starting at 1 and does not describe this permutation. The generation protocol and the dataset provenance therefore need clarification; no generator or dataset has been changed during this preparation.

There are 100 Excel instances per released synthetic scale and 10 per industrial scale. For directory input, the loader processes all Excel files regardless of `--val_size` or `--offset`. Use an explicit Excel file for a single-instance test. See [reproduce.md](reproduce.md) for statistical and timing interpretation.

## Experiment Coverage

| Manuscript experiment | Released entrypoint and assets | Coverage / outstanding work |
| --- | --- | --- |
| Synthetic main comparison | `scripts/eval_drl.py`, `scripts/benchmark_all.py`; three DRL methods; checkpoints at 10/20/40/60/100; 100 instances per synthetic scale | Evaluation is available; training provenance, original per-instance results, and full-method RPD aggregation need reconciliation. |
| Conventional comparison | `scripts/run_conventional.py`; Gurobi, OR-Tools, ALNS, IGA, DABC, DIWO | Single-instance runs are available. Match repeat selection, budgets, and result aggregation to the reported table. |
| Industrial evaluation | Explicit industrial directories or `Industrial_Dataset`; checkpoints and data at 10/20/30/40 | DRL evaluation is available. Gurobi/ALNS need separate runs and aggregation. |
| Cross-scale generalization | Explicit test directory with a different checkpoint scale | Evaluation is available; the selected-transfer aggregation and bootstrap plot pipeline are not included. |
| Pair-first / Single-stream / w/o R-stream / Flat Triplet | Main model only | Variant implementations, variant checkpoints, and the training/inference memory measurement protocol are not included. |
| Five training seeds | One checkpoint per method and scale | Five-run weights and the variability aggregation script are not included. |
| Sampling-budget study | `--decode_strategy sample --width K` | Individual budgets can be evaluated; matched-instance timing, repeated sampling seeds, and bootstrap analysis are not included. |
| Weight sensitivity | Released model uses weight 0.4 | The separately trained weight-specific models and study script are not included. Reweighting one schedule is not the retraining experiment. |
| Execution disturbances | Nominal deterministic scheduling environment | Disturbed replay, 20 realizations per instance, and the study result records are not included. |
| Webots workflow | `media/Webots_carrier_worker_cooperation.mp4` | Demonstration recording only; no world files or controller source is included. |
| Gantt example | Two chart images | Additional single-instance illustration; source instance and schedule traces are not packaged. |

See [reproduce.md](reproduce.md) for commands and the distinction between standard deviation, uncertainty of the mean, and RPD. Neither these commands nor a successful smoke test establishes numerical reproduction of all manuscript tables.

## Terminology and Assets

Public descriptions use Collaborative Carrier-Worker Scheduling Problem (CCWSP). Internal `ahasp`, `cur_time`, and `visited_` identifiers remain for compatibility. Robot states describe availability after scheduled operations; the task mask records scheduling decisions, not physical completion at a global clock time.

The architecture image is copied from the current manuscript. Its embedded "Completed task mask" label still needs correction in the figure source to "Scheduled task mask"; the README clarifies its meaning. The old supplementary PDF is excluded from this review release. Third-party attribution and licensing are retained.

## Anonymous Access and Video

The existing anonymous mirror is not ready for reviewer access. On 2026-09-14, its public options endpoint returned a redirect to the non-anonymous source repository. Its README and checkpoint file endpoints returned HTML from the public repository, and its archive endpoint returned `text/html` rather than a ZIP. An HTTP 200 response therefore did not demonstrate anonymous code or binary checkpoint availability.

Before sharing the review release, configure the anonymous mirror to serve the review branch without redirecting to the source repository. Then verify README rendering, archive extraction, and all 18 weight downloads, including their binary content and hashes. Do not instruct reviewers to recover missing LFS objects through the public source repository. No remote settings were changed during this local preparation.

The current Webots MP4 is 25,520,939 bytes, 28.14 seconds, 1920x1080, approximately 46.875 fps, and progressive H.264. It exceeds the 20 MB submission limit and needs a separate compressed submission copy. The repository recording is retained unchanged. ICRA 2027 specifies a maximum of 180 seconds, minimum height 480, minimum 20 fps, and progressive scanning for submitted videos; repository playback does not replace submission of an accompanying video. See the [official submission instructions](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/).

ICRA 2027 uses double-anonymous review. External URLs are permitted, but reviewers need not consult them; the complete paper, including any supplementary text, must fit within eight pages. These repository documents are optional implementation references, not additional mandatory review pages. See the [official submission instructions](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/).

## Local Verification

- All 94 tracked Python sources passed syntax parsing. The seven changed Python files differ only in the problem acronym inside command-line descriptions or error messages; model computation is unchanged.
- SHA-256 comparisons confirmed that all 576 tracked checkpoint, configuration, and instance files were unchanged. The architecture figure matches the current manuscript image.
- The four unified command-line entrypoints displayed their help successfully. All checks ran in the existing Python 3.10 Conda environment.
- CPU evaluation passed for MaRS-Net, HDRL, and TDRL with greedy and Sample-1280 decoding. These checks used a single 20-task instance, except the MaRS-Net greedy run, which evaluated all 100 instances and returned mean objective 1377.9319, matching the manuscript's rounded 1377.93.
- A 40-task MaRS-Net checkpoint successfully evaluated a 60-task instance with Sample-1280 using an explicit test-file path. This verifies the interface, not the complete generalization experiment.
- Local Markdown links and image/video paths resolved. Tracked text and inspected image, spreadsheet, and video metadata showed no direct author-account or local-path identifiers matching the audit search terms. Sampled video frames showed the simulation scene without identifying overlays; this was not an exhaustive frame-by-frame anonymity audit.
- Anonymous remote download verification failed as described above. Full experiment reproduction, standardized timing/memory measurements, and training-run provenance remain unresolved.
