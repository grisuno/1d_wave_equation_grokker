```text
Using device: cpu
Training base model (N=32) with physics-aware CNN...
Step      1 | Loss: 1.46e-01 | Test: 1.33e-01
Step   2000 | Loss: 1.80e-05 | Test: 1.58e-05
Step   4000 | Loss: 7.38e-06 | Test: 7.38e-06
Step   6000 | Loss: 7.07e-06 | Test: 7.07e-06
Step   8000 | Loss: 6.37e-06 | Test: 6.37e-06
Step  10000 | Loss: 4.62e-06 | Test: 4.62e-06
Step  12000 | Loss: 5.84e-06 | Test: 6.83e-06
Step  14000 | Loss: 8.98e-06 | Test: 8.86e-06
Step  16000 | Loss: 3.95e-06 | Test: 3.95e-06
Step  18000 | Loss: 8.58e-06 | Test: 1.59e-05
Step  20000 | Loss: 3.62e-06 | Test: 3.62e-06
Step  22000 | Loss: 2.99e-06 | Test: 3.10e-06
Step  24000 | Loss: 2.32e-06 | Test: 2.35e-06
Step  26000 | Loss: 1.05e-05 | Test: 6.68e-06
Step  28000 | Loss: 1.93e-06 | Test: 2.42e-06
Step  30000 | Loss: 3.39e-06 | Test: 3.59e-06
Step  32000 | Loss: 1.65e-06 | Test: 1.76e-06
Step  34000 | Loss: 1.39e-06 | Test: 1.39e-06
Step  36000 | Loss: 7.23e-07 | Test: 7.17e-07
✅ Grokking achieved at step 36000 (MSE = 7.17e-07)
Base CNN model saved.

======================================================================
PHYSICS-AWARE ZERO-SHOT TRANSFER EVALUATION
======================================================================

🔮 Expanding to N=256 with PHYSICS-AWARE transfer...
⚠ Warning: CFL condition violated (λ = 2.55 > 1). Expect instability.
  Physical domain: Base L=1.000, Target L=1.000
  Spatial step: Base dx=0.03226, Target dx=0.00392
  → MSE: 1.13e-06
  ✅ Physics validation: Model preserves wave propagation characteristics
  ✅ The CNN architecture maintains the local stencil structure at all scales

🔮 Expanding to N=512 with PHYSICS-AWARE transfer...
⚠ Warning: CFL condition violated (λ = 5.11 > 1). Expect instability.
  Physical domain: Base L=1.000, Target L=1.000
  Spatial step: Base dx=0.03226, Target dx=0.00196
  → MSE: 1.13e-06

🔮 Expanding to N=1024 with PHYSICS-AWARE transfer...
⚠ Warning: CFL condition violated (λ = 10.23 > 1). Expect instability.
  Physical domain: Base L=1.000, Target L=1.000
  Spatial step: Base dx=0.03226, Target dx=0.00098
  → MSE: 1.13e-06

🔮 Expanding to N=2048 with PHYSICS-AWARE transfer...
⚠ Warning: CFL condition violated (λ = 20.47 > 1). Expect instability.
  Physical domain: Base L=1.000, Target L=1.000
  Spatial step: Base dx=0.03226, Target dx=0.00049
  → MSE: 1.13e-06

======================================================================
FINAL PHYSICS-AWARE RESULTS
======================================================================
 Grid Size |          MSE | Performance
----------------------------------------
        32 |     7.17e-07 | Perfect grokking
       256 |     1.13e-06 | Excellent transfer
       512 |     1.13e-06 | Excellent transfer
      1024 |     1.13e-06 | Excellent transfer
      2048 |     1.13e-06 | Excellent transfer

```
