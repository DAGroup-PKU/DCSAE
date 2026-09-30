# Validation

Validated locally with Python 3.12 on CPU; direct dependency versions are recorded
in `requirements-tested.txt`.

- All Python sources parse; all included YAML and JSON configurations parse.
- SAE and DiT training `--help` entrypoints succeed when invoked directly.
- Seven offline tests pass: imports, training entrypoints, strict checkpoint
  loading, per-image PSNR averaging, latent normalization round-trip, a small
  DiT CFG sampling run, and end-to-end SAE PSNR evaluation using a locally created
  tiny DINOv2 encoder and three synthetic images (including a partial last batch).
- Output reuse is rejected to avoid mixing evaluation samples.
- Original machine paths, SSH endpoints, experiment artifacts and authentication
  files are excluded. No original Git metadata is copied.

These checks do not constitute full CUDA/DDP/FSDP training validation. No real
SAE/DiT benchmark checkpoint or ImageNet evaluation was run, and no measured
PSNR/rFID/gFID benchmark result is claimed. Other encoder families are retained
but were not exercised with their pretrained weights in these offline tests.

A local macOS `torchrun --standalone` help check timed out during rendezvous;
CPU checks validate the training CLI directly instead. Full torchrun training
still requires verification on the target Linux/CUDA system.
