#!/usr/bin/env bash
set -Eeuo pipefail
root=${1:?experiment root}; dependency=${2:?dependency root}
test -n "${SLURM_JOB_ID:-}" || exit 2
cd "$root"
mkdir -p results/20261012/metadata models
log="results/20261012/metadata/setup-${SLURM_JOB_ID}.log"
exec > "$log" 2>&1
trap 'printf "%s\n" "$?" > "results/20261012/metadata/setup-${SLURM_JOB_ID}.exit"' EXIT
module load CUDA/12.9.0
export CUDA_HOME=/apps/software/CUDA/12.9.0
export PATH="$root/env/bin:$dependency/env/compiler/bin:$CUDA_HOME/bin:$PATH"
export CC="$dependency/env/compiler/bin/x86_64-conda-linux-gnu-cc"
export CXX="$dependency/env/compiler/bin/x86_64-conda-linux-gnu-c++"
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=9.0
export LD_LIBRARY_PATH="$dependency/env/compiler/lib:$root/native/lib:${LD_LIBRARY_PATH:-}"
if test ! -f "$root/native/lib/libGL.so.1"; then
    mkdir -p tools
    if test ! -x tools/bin/micromamba; then
        curl --fail --location --retry 3 https://micro.mamba.pm/api/micromamba/linux-64/latest -o tools/micromamba.tar.bz2
        tar -xjf tools/micromamba.tar.bz2 -C tools bin/micromamba
    fi
    MAMBA_ROOT_PREFIX="$root/tools/mamba-root" tools/bin/micromamba --no-rc create --yes --prefix "$root/native" --override-channels --channel conda-forge libgl libegl
fi
base="$dependency/env/foundationpose/bin/python"
test -x env/bin/python || "$base" -m venv --system-site-packages env
python="$root/env/bin/python"
"$python" -m pip install --no-deps numpy==1.26.4 scipy==1.14.1 scikit-learn==1.5.2 faiss-cpu==1.9.0.post1 easydict==1.13 einops==0.8.1 ipdb ninja libarchive-c
"$python" -m pip install ipdb==0.13.13 decorator==5.3.1
"$python" - <<'PY'
import hashlib, pathlib, gdown, libarchive
root=pathlib.Path.cwd(); archive=root/'models/real3dad.rar'; target=root/'models/reg2inv-real3dad.pth.tar'
expected='2edcce83f8fe6ab2573831ef61bcd663b2e2c3f2dac4fded3365022ec1ac60c3'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
if not target.exists():
 if not archive.exists():
  result=gdown.download(id='13tXihXZJoF53n_ZQtNVvGlqOr7woNfq0',output=str(archive),quiet=False)
  if result is None:raise RuntimeError('Official Reg2Inv checkpoint download failed')
 with libarchive.file_reader(str(archive)) as reader:
  found=False
  for entry in reader:
   if str(entry.pathname).endswith('snapshots/snapshot.pth.tar'):
    with target.open('wb') as f:
     for block in entry.get_blocks():f.write(block)
    found=True;break
  if not found:raise RuntimeError('Checkpoint not found in official archive')
assert sha(target)==expected, 'Checkpoint hash differs from historical official weights'
print('CHECKPOINT_VERIFIED',expected)
PY
if ! compgen -G 'third_party/Reg2Inv/geotransformer/ext*.so' > /dev/null; then
    (cd third_party/Reg2Inv; "$python" setup.py build_ext --inplace)
fi
"$python" - <<'PY'
import sys,json,importlib.metadata as m
sys.path.insert(0,'third_party/Reg2Inv')
import torch
import geotransformer.ext,numpy,faiss,open3d,collada
from pathlib import Path
Path('results/20261012/metadata/environment.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,numpy=numpy.__version__,packages={k:m.version(k) for k in ['faiss-cpu','open3d','pycollada','scipy','scikit-learn']}),indent=2)+'\n')
print('SETUP_COMPLETE')
PY
