"""Download only the pinned model files and verify their recorded hashes."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'sources/model_download_manifest.json').read_text())
snapshot_download('Qwen/Qwen3-VL-2B-Instruct',
                  revision='89644892e4d85e24eaac8bacfd4f463576704203',
                  allow_patterns=[row['path'] for row in manifest['files']],
                  local_dir=args.model_dir)
for row in manifest['files']:
    with (args.model_dir/row['path']).open('rb') as source:
        actual = hashlib.file_digest(source, 'sha256').hexdigest()
    if actual != row['sha256']:
        raise RuntimeError('Model hash mismatch: '+row['path'])
print('Pinned model files verified.')
