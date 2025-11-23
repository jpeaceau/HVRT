# Archive Index

This directory contains experimental validation data for H-VRT performance claims in README.md.

## Contents

### Experimental Data

**`experimental/results/svm_pilot/`**
- `pilot_results_with_snr.json` - Primary validation study (15 trials with SNR measurements)
- `pilot_results.json` - Original pilot results (pre-SNR)

### Experiment Scripts

**`experimental/experiments/`**
- `exp_svm_pilot_with_snr.py` - Reproduces SVM pilot with SNR (~30 seconds)
- `exp_svm_feasibility_study.py` - Full-scale 50k+ design (not yet executed)

### Analysis Documents

**`docs/`**
- `SVM_PILOT_SNR_ANALYSIS.md` - SNR analysis explaining >100% accuracy
- `SVM_PERFORMANCE_ANALYSIS.md` - SVM feasibility and speedup analysis

## Reproduce Results

```bash
cd experimental/experiments
python exp_svm_pilot_with_snr.py  # ~30 seconds
```

Results will be written to `experimental/results/svm_pilot/pilot_results_with_snr.json`

## Key Findings

- **Well-behaved data:** H-VRT 93.9% R², 126.2% SNR | Random 95.3% R², 97.9% SNR
- **Heavy-tailed data:** H-VRT 106.6% R², 130.1% SNR | Random 85.3% R², 101.3% SNR
- **SVM speedup:** 24-38x training time reduction

## Citation

```bibtex
@misc{hvrt_experiments2025,
  author = {Peace, Jake},
  title = {H-VRT Experimental Validation: SVM Feasibility Study},
  year = {2025},
  url = {https://github.com/hotprotato/hvrt/tree/main/archive}
}
```
