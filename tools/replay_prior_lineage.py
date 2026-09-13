"""Recalculate the supplied prior-lineage ID/hash overlaps and receipt minimum.

Raw-image hashes were verified at staging; this portable calculation checks
the distributed hash tables, not the undistributed complete raw images.
"""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'evidence/prior_lineage'
def read(p):return json.loads((root/p).read_text())
train=read('TRAIN_SELECTION_IMAGE_IDS.json')
evaluation=read('CONFIRMATION_IMAGE_IDS.json')
history=read('results/projection_prior/seed17/history.json')
receipt=read('results/projection_prior/seed17/summary.json')
expected=read('AUDIT.json')
hashes={x['image_sha256'] for x in train}
other={h for x in evaluation for h in x['frame_sha256']}
actual={'fitting_and_selection_rows':len(train),'fitting_and_selection_unique_images':len(hashes),
        'confirmation_episodes':len(evaluation),'confirmation_unique_frame_hashes':len(other),
        'episode_id_overlap':len({x['id'] for x in train}&{x['id'] for x in evaluation}),
        'family_identifier_overlap':len({x['family'] for x in train}&{x['family'] for x in evaluation}),
        'exact_image_hash_overlap':len(hashes&other)}
assert all(expected[k]==v for k,v in actual.items())
assert len(history)==100
best=min(history,key=lambda x:x['validation_nll_per_coordinate'])
assert best['epoch']==receipt['best_epoch']
assert best['validation_nll_per_coordinate']==receipt['best_validation_nll_per_coordinate']
print(json.dumps({'status':'passed',**actual,'receipt_minimum_matches':True},indent=2))
