# SVM Pilot Study: Signal-to-Noise Ratio Analysis

**Experiment:** SVM Feasibility Pilot with SNR Measurements
**Data:** `../experimental/results/svm_pilot/pilot_results_with_snr.json`

## Summary

H-VRT improves data quality by increasing Signal-to-Noise Ratio (SNR) by 26-30% through intelligent sample selection.

## SNR Definition

```
SNR = Var(signal) / Var(noise)
```

Where signal = X @ weights (true pattern), noise = y - signal (residual).

## Results

### Well-Behaved Data (Normal Distribution)

| Method | R² Retention | SNR Retention | Speedup |
|--------|-------------|---------------|---------|
| H-VRT | 93.9% | **126.2%** | 23.5x |
| Random | 95.3% | 97.9% | 25.9x |

### Heavy-Tailed Data (Cauchy + Rare Events)

| Method | R² Retention | SNR Retention | Speedup |
|--------|-------------|---------------|---------|
| H-VRT (hybrid) | **106.6%** | **130.1%** | 24.0x |
| Random | 85.3% | 101.3% | 24.0x |

## Key Findings

**Well-behaved data:**
- Random achieves slightly better R² (95.3% vs 93.9%) due to CLT
- H-VRT achieves 26% SNR improvement by selecting high-signal regions
- Both methods enable ~24x SVM training speedup

**Heavy-tailed data:**
- H-VRT achieves +21pp accuracy advantage (106.6% vs 85.3%)
- H-VRT improves SNR by 30% (cleaner training data)
- Random sampling fails when CLT assumptions break down

## Why >100% Accuracy?

H-VRT achieves 106.6% accuracy retention (better than full dataset) by:
1. Removing low-signal, high-noise samples (noise filtering)
2. Preserving rare extreme events (1.03x capture ratio)
3. Improving SNR by 30% → better SVM generalization

## Experimental Setup

- Scale: 10,000 samples, 20 features
- Reduction: 20% retention (5x compression)
- Replications: 3 per condition
- Model: SVM (RBF kernel, C=1.0)
- Data types:
  - Well-behaved: Linear signal + Normal noise
  - Heavy-tailed: Linear signal + Cauchy noise + 5% rare events

## Conclusion

SNR measurements explain H-VRT's superiority on heavy-tailed data:
- Acts as intelligent denoiser (30% SNR improvement)
- Filters low-signal noise while preserving high-signal extremes
- Essential for non-well-behaved data where CLT fails
