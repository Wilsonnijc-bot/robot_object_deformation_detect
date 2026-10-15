# Reg2Inv scoring code used by the frame-0 experiment

Copied on 2026-10-15 from the deleted CAD-reference experiment (`reg2inv_foundationpose`),
byte for byte. `paired_scoring.py`, `reg2invbaselinecode/reg2inv.py`, `reg2invbaselinecode/geometry.py`,
`gpu_ransac.py` and `execution_cache.py` match the `core_sources` hashes frozen in
`../results/20261014-45/inputs/manifest.json`. `orchestrate.py` supplies `Monitor` to `../run.py`.

- `third_party/Reg2Inv` and `third_party/KNN_CUDA`: upstream source at the commits in `source_provenance.json`.
- Checkpoint `reg2inv-real3dad.pth.tar` (SHA-256 `2edcce83…60c3`): ERIS `/scratch/z/zy992/models/reg2inv/`.
- The conda environment and native libraries were deleted with the CAD workspace. `setup_eris_original.sh`
  is the script that built them there (it also installed FoundationPose pieces the frame-0 run does not use).
  To rerun, rebuild them and pass `REG2INV_ENV` and `REG2INV_NATIVE` to `../gpu_job.sh`.
