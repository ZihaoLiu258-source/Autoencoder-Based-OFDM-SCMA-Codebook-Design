# Contributing

Contributions that improve correctness, reproducibility, or documentation are welcome after the repository is released by the authors.

## Development workflow

1. Create a focused branch from `main`.
2. Keep the main fixed-point training path unchanged unless the scientific method itself is being revised.
3. Write ablations to a dedicated directory under `artifacts/phase1/`; never overwrite the released checkpoint under `artifacts/main/checkpoints/`.
4. Record seeds, raw error counts, total bits, stopping reasons, checkpoint hashes, and complete logs for result-changing experiments.
5. Run `python -m tools.check_repository` and, when dependencies are installed, `python -m tools.validate_phase1_artifacts` before opening a pull request.

## Code and artifact standards

- Prefer small, documented scripts with explicit constants and deterministic seeds.
- Use repository-relative paths in logs and metadata.
- Do not commit caches, temporary plots, local environment files, credentials, or files larger than 50 MiB.
- Distinguish empirical observations from general claims in documentation.
- Describe any change to the channel, detector, normalization, stopping rule, or random-number coupling in the pull request.

For large methodological changes, open an issue first so the experiment design and computational cost can be reviewed before a full training run.
