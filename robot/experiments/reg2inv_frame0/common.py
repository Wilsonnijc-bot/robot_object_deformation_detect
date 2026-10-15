"""Owned frame-0 ablation paths and immutable provenance."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results/20261014-45'
OBSERVATIONS = ROOT / 'observations'
# Reg2Inv scoring code, byte-identical to the hashes frozen in results/20261014-45/inputs/manifest.json.
SOURCE_ROOT = ROOT / 'reg2inv_core'
CHECKPOINT = Path('/scratch/z/zy992/models/reg2inv/reg2inv-real3dad.pth.tar')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def manifest():
    return json.loads((RESULTS / 'inputs/manifest.json').read_text())['entries']


def code_sha():
    return hashlib.sha256(json.dumps({name: sha(ROOT / name) for name in
        ['common.py', 'worker.py', 'run.py', 'preflight.py', 'summarize.py', 'link2_islands.py']},
        sort_keys=True).encode()).hexdigest()
