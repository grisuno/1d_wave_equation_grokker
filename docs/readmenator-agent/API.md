# API

## app.py
- `generate_wave_data` (function) `app.py:43` `def generate_wave_data(N, T, c, dt, L, seed)` -- Generate synthetic dataset for 1D wave equation using exact numerical scheme.
- `step` (method) `app.py:67` `def step(u_t, u_tm1)` -- One-step wave propagation using finite difference.
- `WaveGrokCNN.__init__` (method) `app.py:118` `def __init__(self, hidden_dim)`
- `WaveGrokCNN.forward` (method) `app.py:150` `def forward(self, x)`
- `WaveGrokCNN.expand_model_weights_physics` (method) `app.py:161` `def expand_model_weights_physics(model, target_N, base_N)` -- Physics-aware weight expansion that preserves the discrete Laplacian structure.
- `WaveGrokCNN.evaluate_expanded_physics` (method) `app.py:224` `def evaluate_expanded_physics(model, base_N, target_N, device, seed)` -- Evaluate expanded model with proper physics scaling.
- `WaveGrokCNN.train_until_grokking` (method) `app.py:258` `def train_until_grokking(model, X, Y, device, grok_threshold, max_steps)`
- `WaveGrokCNN.main` (method) `app.py:307` `def main()`
