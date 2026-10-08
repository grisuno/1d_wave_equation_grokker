# API

## app.py

### generate_wave_data (function) `def generate_wave_data(N, T, c, dt, L, seed)`
- Defined: `app.py:43`
- Doc: Generate synthetic dataset for 1D wave equation using exact numerical scheme.

### expand_model_weights_physics (method) `def expand_model_weights_physics(model, target_N, base_N)`
- Defined: `app.py:161`
- Doc: Physics-aware weight expansion that preserves the discrete Laplacian structure.

### evaluate_expanded_physics (method) `def evaluate_expanded_physics(model, base_N, target_N, device, seed)`
- Defined: `app.py:224`
- Doc: Evaluate expanded model with proper physics scaling.

### train_until_grokking (method) `def train_until_grokking(model, X, Y, device, grok_threshold, max_steps)`
- Defined: `app.py:258`

### main (method) `def main()`
- Defined: `app.py:307`

### step (method) `def step(u_t, u_tm1)`
- Defined: `app.py:67`
- Doc: One-step wave propagation using finite difference.

### __init__ (method) `def __init__(self, hidden_dim)`
- Defined: `app.py:118`

### forward (method) `def forward(self, x)`
- Defined: `app.py:150`
