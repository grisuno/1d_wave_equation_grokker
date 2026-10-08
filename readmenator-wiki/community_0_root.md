# root

*Community 0 | 2 files | cohesion 1.00*

## Definition

This community groups 2 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `WaveGrokCNN`, `__init__`, `evaluate_expanded_physics`, `expand_model_weights_physics`, `forward`, `generate_wave_data`, `main`, `step`. Core file: `app.py` (9 symbols). Documented purpose: Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: xx/xx/xxxx Licencia: GPL v3  Descripción:.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `app.py` | py | utility | 9 | yes |
| `install.sh` | sh | utility | 0 | no |

## Key Symbols

- `generate_wave_data` (function, `app.py:43`) `def generate_wave_data(N, T, c, dt, L, seed)` - Generate synthetic dataset for 1D wave equation using exact numerical scheme.
- `step` (method, `app.py:67`) `def step(u_t, u_tm1)` - One-step wave propagation using finite difference.
- `WaveGrokCNN` (class, `app.py:112`) `class WaveGrokCNN(Module)` - Physics-aware CNN architecture that respects the local stencil structure
- `__init__` (method, `app.py:118`) `def __init__(self, hidden_dim)`
- `forward` (method, `app.py:150`) `def forward(self, x)`
- `expand_model_weights_physics` (method, `app.py:161`) `def expand_model_weights_physics(model, target_N, base_N)` - Physics-aware weight expansion that preserves the discrete Laplacian structure.
- `evaluate_expanded_physics` (method, `app.py:224`) `def evaluate_expanded_physics(model, base_N, target_N, device, seed)` - Evaluate expanded model with proper physics scaling.
- `train_until_grokking` (method, `app.py:258`) `def train_until_grokking(model, X, Y, device, grok_threshold, max_steps)`
- `main` (method, `app.py:307`) `def main()`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- No cross-community bridges recorded. This community is self-contained.

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- Why do 1 file(s) lack file-level docs (e.g. `install.sh`)? What purpose do they serve?
- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `app.py`
- `install.sh`
