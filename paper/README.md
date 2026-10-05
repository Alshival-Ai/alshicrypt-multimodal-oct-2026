# Final manuscript

Reproducibility repository:
https://github.com/Alshival-Ai/alshicrypt-multimodal-oct-2026.
The Pokémon images are from yehongjiang's
[Pokemon sprite images](https://www.kaggle.com/datasets/yehongjiang/pokemon-sprites-images)
dataset on Kaggle. After cloning, run
`git submodule update --init deprecated/alshicrypt-multimodal` to obtain the pinned
original project and exact dataset. Omit `--recursive`: the original project's
unused historical wiki entry has no submodule URL. The repository includes both frozen
campaigns under `runs/`, with their checkpoints, results, and selection records.

`main.tex` has three main sections: tensor theory, the future-system post-quantum
conjecture, and architecture/training/results. The appendix contains 25
predetermined test examples. The conjecture is unproven and concerns a future
keyed design; the current model has an explicit inverse and no security claim.
Section 2.2 includes a conditional proof sketch for a one-time-pad-protected
message carried through a public, exactly reversible learned representation.
The sketch covers passive classical or quantum observation under fresh secret
pads and the stated metadata assumptions. It is separate from the Gaussian
experiment and assumes an exact inverse after carrier serialization.

From the repository root:

```bash
.venv/bin/python -m paper_study all
```

The runner trains three fresh models for 2,000 updates, freezes checkpoints,
evaluates, generates assets, and compiles with `.tools/tectonic`. Completed runs
are reused only with matching protocol and source. To regenerate figures and the
PDF without training, run `paper_study render` and `paper_study build` with
`.venv/bin/python -m`.

Section 1.1 diagrams the teacher and learned paths, their shared noise input,
and the three error measurements. Section 1.2 illustrates a single Mew trajectory
at steps 0, 1, 2, 5, 16, and 32.
Section 1.3 uses Mew's channel planes to explain the two coupling groups, a
block's forward and inverse operations, and the four-block ordering.
Section 1.4 develops the differential-geometric and algebraic interpretation,
with a two-coordinate polynomial coupling illustration, its Jacobian, and the
inverse composition. It distinguishes the implemented smooth SiLU conditioners
from a possible polynomial variant.
Regenerate the four generated conceptual figures with `.venv/bin/python -m paper_study illustrate`,
then rebuild the PDF. The trajectory's seed, source hash, coefficients, and unclipped RGBA
snapshots are saved in `generated/mew-stochastic-*`.
The coupling diagram and its source/display metadata are in `generated/mew-coupling-diagram.*`.

`paper_study explain` regenerates the process overview, the geometry figure,
the before/after training
validation table (means across all three seeds), the test table, and the compact
main-text example from saved results and images. It verifies frozen selection,
performs no inference, and does not train or reevaluate models. Use
`.venv/bin/python -m paper_study explain`, then `paper_study build`, for these
editorial assets. All 25 test examples remain in the appendix.

`generated/` contains all three seeds' test metrics, learning histories, protocol,
selection hashes, the inherited split, and a descriptive 16-step comparison.
`examples/` contains 25 triplets, five sheets, the overview, exact float payloads
and conditioning fields, and per-image metrics. PNG ciphertext images are
display-only previews; decoding uses the `.npy` files. The test set was already
examined in the earlier research, which remains in `research/`.
