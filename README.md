# Alshicrypt multimodal — October 2026

Samuel Cavazos, Chief Data Scientist at [Alshival.Ai](https://alshival.ai).

Reproducible research on learning stochastic image transformations with Pokémon
and synthetic static images. This is an experimental learning benchmark, not a
secure encryption implementation. The original project remains in `deprecated/`.

Paper and reproducibility repository:
[Alshival-Ai/alshicrypt-multimodal-oct-2026](https://github.com/Alshival-Ai/alshicrypt-multimodal-oct-2026).

The 809 Pokémon sprites come from
[Pokemon sprite images](https://www.kaggle.com/datasets/yehongjiang/pokemon-sprites-images),
shared by Kaggle user **yehongjiang**. The pinned original-project submodule
contains the exact RGBA images used in this study. The split manifest records
their file hashes; retain these files for replication of the reported results.

Clone the repository with its original-project submodule:

```bash
git clone --recurse-submodules https://github.com/Alshival-Ai/alshicrypt-multimodal-oct-2026.git
cd alshicrypt-multimodal-oct-2026
```

For an existing clone, run `git submodule update --init --recursive`.

## Final 32-step paper

The final manuscript is [paper/main.pdf](paper/main.pdf), with source in
[paper/main.tex](paper/main.tex). It develops the tensor formulation, an explicitly
unproven conjecture for a future keyed post-quantum system, and the measured
Pokémon experiment. Multimodal applicability concerns the formulation; only RGBA
images are evaluated. This prototype does not establish cryptographic security.

The new study keeps β ≈ 0.082996 and uses the endpoint of **32 stochastic steps**:
signal coefficient 0.25, noise variance 0.9375. Three fresh 44,296-parameter
coupling models train for **2,000 optimizer updates each**, with seeds 17, 29,
and 43. Alpha is transformed, and payloads retain float32 precision. The split is
inherited from the exploratory study; its test set has already been examined.

All three 32-step models recovered every tested RGBA image exactly after byte
rounding, including the 81 Pokémon, 32 static images, and nine structured probes.
Mean Pokémon float round-trip MAE is **1.46 × 10⁻⁷**. Mean RGB preview correlation
fell from **0.1195** at 16 steps to **0.0515** at 32 steps. Forward-target error
decreased while reverse-target error increased slightly; full results retain both.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m paper_study all
```

Separate stages are `train`, `evaluate`, `render`, and `build`. Use
`--output runs/paper-32step-replication` for a fresh replication. The default run
directory is `runs/paper-32step-v1/`. Checkpoints retain schedule metadata and are
frozen before test evaluation. Interrupted training resumes from validation
checkpoints. Do not edit `paper_study/core.py` or `alshicrypt/` during a campaign.
Reporting and manuscript edits do not change the experiment hash.

Results are in [paper/generated/summary.json](paper/generated/summary.json).
The [25-example overview](paper/examples/overview.png), five larger sheets, and
per-image float payloads, noise fields, and metrics are in `paper/examples/`.
The published snapshot includes the frozen `runs/paper-32step-v1/` campaign and
the `runs/campaign-v1/` checkpoints used for the 16-step comparison. These allow
inspection of saved results and regeneration without retraining. PDF compilation
uses Tectonic; install its executable at `.tools/tectonic`. The compiled paper
is also included in the repository.

## Completed exploratory study (16-step Gaussian endpoint)

The RTX 5070 campaign completed 35 configurations at 1,000–2,000 updates each:
15 screening runs, 18 repeated runs, and two fixed-noise diagnostics. The split is
647 training / 81 validation / 81 test sprites. Selection was frozen before test
evaluation; the original automated suite contained 17 checks.

Validation selected the **44,296-parameter Gaussian coupling model with
Pokémon-only training** as the exploratory study's main approach. Its mean held-out forward
plus reverse target MAE is **0.04924**, compared with **0.05155** for equal mixed
training and **0.12187** for static-only training. Uniform static did not help
under this fixed update budget; this does not test adding extra synthetic updates
while preserving the full Pokémon exposure.

Every retained byte model recovered its own serialized test outputs exactly,
but Pokémon-trained byte models matched only **20.88% ± 0.70%** of prescribed
forward target bytes across seeds. Architectural invertibility and target learning
are separate results. The legacy diagnostic also found identical RGB fields
across all 81 validation images, with their original alpha masks unchanged.

Read [the research report](research/main.pdf), [aggregate results](research/generated/summary.json),
and [all per-seed metrics](research/generated/results.csv). Local checkpoints remain
under `runs/campaign-v1/`; for example, the selected architecture's seed-17
checkpoint is `extended/gaussian_coupling_pokemon_fresh_s17/best.pt` within that
directory. The following commands reproduce the complete study.

## Setup

The local reference system is an RTX 5070 with 12 GB VRAM. Use Python 3.12 and a
Blackwell-compatible PyTorch build:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
```

`requirements-lock.txt` records the exact environment used for the local study.
No pretrained weights or external AI API are required.

To reproduce every installed dependency exactly, use
`uv pip install --python .venv/bin/python -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match`,
then install this project with `uv pip install --python .venv/bin/python --no-deps -e .`.

## Reproduce the exploratory study

The local study uses `./scripts/reproduce.sh`: 1,000-update screening and
2,000-update repeated trials (10 and 20 defined epochs). This also runs checks,
the sealed test evaluation, diagnostics, report generation, and PDF compilation
when a TeX engine is available. Pass `--output runs/another-campaign` for a separate
replication. To inspect or run stages individually:

```bash
.venv/bin/python -m alshicrypt.cli prepare
.venv/bin/python -m alshicrypt.cli screen
.venv/bin/python -m alshicrypt.cli extend --extended-steps 2000
.venv/bin/python -m alshicrypt.cli test
.venv/bin/python -m alshicrypt.cli diagnose
.venv/bin/python scripts/research_report.py
```

`scripts/reproduce.sh` runs these stages in order and routes reports to `research/`.
Use that script instead of the frozen legacy CLI's `all` or `report` commands,
which still contain the historical `paper/` destination. `--output` selects an independent campaign.
`--screen-steps` (maximum 1,000) and `--extended-steps` (maximum 10,000) set update
caps. One epoch means 100 updates, including for synthetic data. Defaults profile
100 updates per architecture and cap training at 30 minutes per screening run and
two hours per extended run, reserving 20% for overhead. Evaluation and checkpoint
overhead are additional; the recorded training and total times distinguish them.
Every candidate within a phase receives the same update budget. A run trains both
directions, so paired models use more parameters than tied invertible models;
the report records this difference rather than claiming parameter matching.

Completed runs are reused only for identical configurations. Interrupted runs
resume from the last validation checkpoint with optimizer and random states.
Do not edit model/training source during an active run. The two published
campaign directories are versioned for reproducibility; new campaign directories
and runtime logs under `runs/` are ignored. The wrapper writes portable metric
summaries to `research/generated/`.

## Experimental controls

- Versioned 80/10/10 split: duplicates, named variants and documented evolution
  links remain together. The CSV does not supply a complete family taxonomy.
- RGB **and alpha** are transformed. Inputs stay at native 120 × 120 resolution.
  Exact recovery compares decoded RGBA pixel bytes; PNG container bytes and
  embedded metadata are outside this study.
- Pokémon, static and mixed training use equal update/sample budgets.
- Static pixels are independent uniform integers, generated on demand. Validation
  and test have separate fixed streams; probes contain colors, ramps and impulses.
- Both models receive the cumulative noise field. A recorded seed can reproduce
  that field with the specified generator, shape, dtype and stream position. The
  reported eight-byte seed alternative counts only the seed, not the additional
  stream-position/protocol metadata; the main conditioning-size metric counts
  the entire noise field. Conditioning is
  explicitly counted; it is not a secret key or free hidden communication.
- Byte targets are addition modulo 256 with exact PNG transport. Gaussian targets
  are the endpoint law of 16 diffusion steps with signal factor 0.5; transport uses
  float32 NumPy arrays without clipping. Gaussian PNG clipping is a diagnostic.
- Independent CNNs use categorical byte outputs or float regression. Coupling
  networks have a constructed inverse; they still must learn the target map.
- A byte follow-up (`guided`) retains the coupling architecture and adds supervised
  intermediate shifts. It exposes extra teacher information and is a training
  intervention, not an architecture-only comparison. The original 12-condition
  screen is retained alongside these three additional runs.
- The Gaussian coupling's fixed attenuation is a declared architecture advantage:
  this benchmark compares inductive biases, not equally uninformed learners.
- Validation selects checkpoints and architectures. `selection.json` freezes all
  retained candidates before `test` opens the holdout. Editing the frozen selection
  or its checkpoint files blocks further test evaluation for that campaign.
- Exact invertible round trips alone are not evidence of learning. Forward target
  agreement, reverse recovery from teacher outputs, untrained controls and static
  transfer are reported separately.
- The main paper track is selected by validation target-learning improvement over
  its untrained control, after architecture promotion. Absolute accuracy and the
  byte/float transport differences remain explicit in the paper.

The exploratory manuscript is `research/main.tex`. Generate its tables and figures
with `scripts/research_report.py`, then compile `research/main.tex` with Tectonic.
The new final paper uses the separate `paper_study` runner described above.
