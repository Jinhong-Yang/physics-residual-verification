"""Descriptive, post hoc influence checks on the saved evaluation units.

No fitting or threshold selection. Bootstrap draws are those in the archives.
Episode results do not establish independence of shared physical groups.
"""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    data = ROOT / 'evidence/results'
    main_values = np.load(data / 'com_observation_v23/confirmation_v1/FAMILY_ACTION_ERRORS.npz', allow_pickle=False)
    public = np.load(data / 'com_observation_v39/physprobe_independent_confirmation_v1/CONFIRMATION_ERRORS.npz', allow_pickle=False)
    rows = []
    comparisons = [
        ('main', 'full_vs_huber', main_values['v22_vlm__family_action_error'], main_values['huber_1.345_rgb__family_action_error'], main_values['bootstrap_draw']),
        ('main', 'full_vs_numeric', main_values['v22_vlm__family_action_error'], main_values['v22_numeric__family_action_error'], main_values['bootstrap_draw']),
        ('public', 'gated_vs_numeric', public['error_gated_vlm'], public['error_numeric'], public['bootstrap_draw']),
        ('public', 'gated_vs_raw', public['error_gated_vlm'], public['error_raw_vlm'], public['bootstrap_draw']),
    ]
    for task, contrast, candidate, reference, draws in comparisons:
        delta = candidate - reference
        assert len(delta) > 1 and np.isfinite(delta).all()
        boot_reference = reference[draws].mean(1)
        boot_candidate = candidate[draws].mean(1)
        relative = 100 * (1 - boot_candidate / boot_reference)
        lo, hi = np.quantile(relative, [.025, .975])
        loo_delta = (delta.sum() - delta) / (len(delta) - 1)
        rows.append(dict(task=task, comparison=contrast, units=len(delta),
                         improved_units=int((delta < 0).sum()),
                         tied_units=int((delta == 0).sum()),
                         worsened_units=int((delta > 0).sum()),
                         mean_delta_mm=float(delta.mean()*1000),
                         median_delta_mm=float(np.median(delta)*1000),
                         relative_improvement_percent=float(100*(1-candidate.mean()/reference.mean())),
                         relative_low95_percent=float(lo), relative_high95_percent=float(hi),
                         leave_one_out_min_mean_delta_mm=float(loo_delta.min()*1000),
                         leave_one_out_max_mean_delta_mm=float(loo_delta.max()*1000),
                         analysis='post hoc descriptive sensitivity; fixed comparator; no multiplicity adjustment'))
    out = ROOT / 'replayed'; out.mkdir(exist_ok=True)
    with (out/'effect_sensitivity.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    (out/'EFFECT_SENSITIVITY.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))

if __name__ == '__main__':
    main()
