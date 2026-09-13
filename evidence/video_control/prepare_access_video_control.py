"""Pin and fetch the official V-JEPA 2 encoder for an additional control."""
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

R=Path(__file__).resolve().parents[1]
O=R/'results/access_video_control_v1'; O.mkdir(exist_ok=True)
repo='facebook/vjepa2-vitl-fpc64-256'
plan=O/'PLAN.json'
if plan.exists(): info=json.loads(plan.read_text())
else:
    revision=HfApi().model_info(repo).sha
    info={'repo':repo,'revision':revision,'scope':'post hoc independent frozen-video representation control',
          'input':'exact original eight RGB observation frames, original order; no extra or future frames',
          'encoder':'frozen official ViT-L; BF16, native 256-pixel spatial preprocessing; no predictor',
          'features':'mean of final encoder tokens, PCA fit on training folds only to 48 dimensions; no labels for projection',
          'regression':'same original numerical head, residual target, penalties 1/10, clipping 2/1, visual scale .25, masks, fallback, covariance .1',
          'folds':'original six development family folds; no final-model reselection',
          'arms':['original_numeric','original_full','vjepa2_48'],
          'reporting':'all arms, both development and previously evaluated cohort; paired post hoc unadjusted intervals',
          'limitations':'equal residual dimension and observations do not equalize encoder pretraining, decoder supervision or spatial processing',
          'independent_confirmation':False}
    plan.write_text(json.dumps(info,indent=2),encoding='utf-8')
print(json.dumps(info),flush=True)
dest=snapshot_download(repo_id=repo,revision=info['revision'],local_dir=R/'models/vjepa2-vitl-fpc64-256',
                       allow_patterns=['config.json','video_preprocessor_config.json','model.safetensors','README.md'],max_workers=2)
print('MODEL_READY '+dest,flush=True)
