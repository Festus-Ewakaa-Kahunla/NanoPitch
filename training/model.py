"""
NanoPitch Model — a lightweight neural network for real-time pitch tracking.

=== What This Model Does ===

Given a short chunk of audio (represented as a mel spectrogram), the model
predicts two things at every 10ms time step:

  1. VAD (Voice Activity Detection): Is someone singing/speaking right now?
     Output: a probability between 0 and 1.

  2. Pitch Posteriorgram: What pitch is being sung?
     Output: 360 probabilities, one for each possible pitch bin.
     The bins cover 6 octaves (B0 through ~B6) at 20-cent resolution.
     A "cent" is 1/100 of a semitone — so 20 cents ≈ 1/5 of a semitone.

=== Architecture (adapted from RNNoise) ===

The model uses GRUs (Gated Recurrent Units), which are a type of recurrent
neural network well-suited for sequential data like audio. GRUs can "remember"
what they heard in previous frames, which helps track pitch continuously.

Signal flow:
    40 mel bands (input)
      │
      ▼
    Conv1d(40 → 64, kernel=3) + tanh    ← extract local patterns
    Conv1d(64 → 96, kernel=3) + tanh    ← combine into features
      │
      ▼
    GRU layer 1 (96 units)              ← track patterns over time
    GRU layer 2 (96 units)              ← deeper temporal modeling
    GRU layer 3 (96 units)              ← even deeper
      │
      ▼
    Concatenate [conv_out, gru1, gru2, gru3] = 384 features
      │
      ├──→ Dense(384 → 1)   + sigmoid  → VAD probability
      └──→ Dense(384 → 360) + sigmoid  → pitch posteriorgram

Total: ~333K parameters — small enough to run on a laptop CPU or in a browser.

=== Why This Design? ===

- Conv layers act as a learned feature extractor (like a smarter mel filterbank)
- GRU layers capture temporal context (pitch is continuous, not independent per frame)
- Multiple GRU layers at different depths capture different time scales
- Concatenating all layers gives the output heads access to both low-level
  and high-level features (a "skip connection" pattern from RNNoise)
- Sigmoid outputs ensure values are in [0, 1], interpretable as probabilities
"""

import torch
from torch import nn


# ═══════════════════════════════════════════════════════════════════════
# Pitch Posteriorgram Constants
#
# We represent pitch as a probability distribution over 360 bins.
# Each bin is 20 cents wide. There are 1200 cents in an octave
# (12 semitones × 100 cents), so 360 bins = 6 octaves.
#
# Bin 0 ≈ B0 (31.7 Hz), just below C1 on a standard piano
# Bin 359 ≈ B6 (~2006 Hz), above typical soprano range
# ═══════════════════════════════════════════════════════════════════════

PITCH_BINS = 360
PITCH_FMIN = 31.7          # Hz — ~B0 (see bin_to_f0 / nanopitch.c)
PITCH_CENTS_PER_BIN = 20   # resolution in cents

N_MELS = 40  # number of mel spectrogram bands

# Maximum layer size supported by the C/WASM inference engine.
# cond_size and gru_size must not exceed this, or the exported model
# will crash in the browser. Matches NC_MAX_LAYER_SIZE in nanopitch.h.
MAX_LAYER_SIZE = 512
DEFAULT_VOICING_THRESHOLD = 0.3
DEFAULT_ONSET_PENALTY = 0.75


# ═══════════════════════════════════════════════════════════════════════
# Pitch Conversion Utilities
# ═══════════════════════════════════════════════════════════════════════

def f0_to_bin(f0_hz):
    """Convert fundamental frequency (Hz) to pitch bin index.

    The formula uses the logarithmic relationship between frequency and
    musical pitch: going up one octave doubles the frequency, and there
    are 1200 cents per octave.

        bin = 1200 * log2(f0 / f_min) / cents_per_bin

    Returns -1 for unvoiced frames (f0 <= 0).
    """
    import numpy as np
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    result = np.full_like(f0_hz, -1.0)
    voiced = f0_hz > 0
    result[voiced] = 1200.0 * np.log2(f0_hz[voiced] / PITCH_FMIN) / PITCH_CENTS_PER_BIN
    return result


def bin_to_f0(bins):
    """Convert pitch bin index back to Hz. Inverse of f0_to_bin."""
    import numpy as np
    bins = np.asarray(bins, dtype=np.float64)
    return PITCH_FMIN * 2.0 ** (bins * PITCH_CENTS_PER_BIN / 1200.0)


def f0_to_posteriorgram(f0_hz, n_frames=None, sigma_bins=1.2):
    """Create a Gaussian-blurred pitch posteriorgram from f0 values.

    For each voiced frame, we place a Gaussian bump centered at the true
    pitch bin. This is a "soft" label — instead of a hard one-hot vector,
    nearby bins also get some probability mass. This helps the model learn
    because pitch is continuous, not discrete.

    Args:
        f0_hz: (T,) array of f0 in Hz (0 = unvoiced)
        sigma_bins: width of the Gaussian in bins (1.2 ≈ 24 cents)

    Returns:
        (T, 360) float32 array — one probability distribution per frame
    """
    import numpy as np
    if n_frames is None:
        n_frames = len(f0_hz)

    f0_hz = np.asarray(f0_hz[:n_frames], dtype=np.float64)
    bins = f0_to_bin(f0_hz)

    posteriorgram = np.zeros((n_frames, PITCH_BINS), dtype=np.float32)
    bin_indices = np.arange(PITCH_BINS, dtype=np.float64)

    for t in range(n_frames):
        if bins[t] < 0:
            continue  # unvoiced frame — all zeros (no pitch)
        # Gaussian centered at the true pitch bin
        dist = bin_indices - bins[t]
        posteriorgram[t] = np.exp(-0.5 * (dist / sigma_bins) ** 2)

    return posteriorgram


def viterbi_decode(posteriorgram, transition_width=12,
                   voicing_threshold=DEFAULT_VOICING_THRESHOLD,
                   onset_penalty=DEFAULT_ONSET_PENALTY):
    """Decode a pitch posteriorgram into a smooth f0 track using Viterbi.

    The Viterbi algorithm finds the most likely sequence of pitch states
    over time, given:
      - Observation probabilities: the model's pitch posteriorgram
      - Transition constraints: pitch can't jump more than ±12 bins per frame
      - Voicing model: an unvoiced state with onset/offset penalties

    This is a proper dynamic programming algorithm (not just argmax), so
    it produces smoother pitch tracks and handles brief dropouts better.

    State space: 360 voiced bins + 1 unvoiced state = 361 total.
    Transition: voiced states can reach neighbors within ±transition_width.
    Observation: log(posteriorgram) for voiced, log(1 - max_post) for unvoiced.

    Vectorized implementation using numpy strided windows for speed.

    Args:
        posteriorgram: (T, 360) pitch probabilities from the model
        transition_width: max pitch change per frame in bins (12 = 240 cents)
        voicing_threshold: min confidence to initialize as voiced
        onset_penalty: log-domain cost for voiced↔unvoiced transitions

    Returns:
        f0_hz: (T,) float32 array of decoded f0 in Hz (0 = unvoiced)
    """
    import numpy as np

    T, N = posteriorgram.shape
    if T == 0:
        return np.zeros(0, dtype=np.float32)

    tw = int(transition_width)
    W = 2 * tw + 1  # window size for transition neighborhood
    log_obs = np.log(posteriorgram + 1e-10)

    # Viterbi tables: V[t] = best log-probability ending in each state
    # States 0..N-1 = pitched, state N = unvoiced
    V = np.full((T, N + 1), -np.inf, dtype=np.float64)
    bp = np.zeros((T, N + 1), dtype=np.int32)  # backpointers

    # ── Initialize frame 0 ──
    max_post = posteriorgram[0].max()
    if max_post > voicing_threshold:
        V[0, :N] = log_obs[0]
    V[0, N] = np.log(1.0 - max_post + 1e-10)

    # ── Forward pass (vectorized per frame) ──
    for t in range(1, T):
        max_post_t = posteriorgram[t].max()
        prev = V[t - 1, :N]  # (N,) previous voiced scores

        # Find best predecessor within ±tw bins for each state.
        # Pad prev with -inf, then use as_strided to get all windows at once.
        padded = np.pad(prev, (tw, tw), constant_values=-np.inf)
        # windows[s, k] = padded[s + k] = prev[s + k - tw] for k in 0..W-1
        windows = np.lib.stride_tricks.as_strided(
            padded, shape=(N, W),
            strides=(padded.strides[0], padded.strides[0]))
        # Best within each window
        best_k = np.argmax(windows, axis=1)          # (N,) offset within window
        best_val = windows[np.arange(N), best_k]     # (N,) best score
        best_from_voiced = np.clip(np.arange(N) - tw + best_k, 0, N - 1)

        # Option: come from unvoiced state (onset penalty)
        from_unvoiced = V[t - 1, N] - onset_penalty

        # Choose best predecessor for each voiced state
        use_voiced = best_val >= from_unvoiced
        V[t, :N] = np.where(use_voiced, best_val, from_unvoiced) + log_obs[t]
        bp[t, :N] = np.where(use_voiced, best_from_voiced, N)

        # Unvoiced state: from best voiced (offset penalty) or stay unvoiced
        best_voiced_score = prev.max()
        best_voiced_idx = prev.argmax()
        from_voiced = best_voiced_score - onset_penalty
        stay_uv = V[t - 1, N]
        uv_obs = np.log(1.0 - max_post_t + 1e-10)

        if stay_uv >= from_voiced:
            V[t, N] = stay_uv + uv_obs
            bp[t, N] = N
        else:
            V[t, N] = from_voiced + uv_obs
            bp[t, N] = best_voiced_idx

    # ── Backtrace ──
    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = np.argmax(V[T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]

    # ── Convert to f0 ──
    f0_hz = np.zeros(T, dtype=np.float32)
    voiced_mask = path < N
    if voiced_mask.any():
        f0_hz[voiced_mask] = bin_to_f0(path[voiced_mask].astype(np.float64))

    return f0_hz


def viterbi_decode_realtime(posteriorgram, transition_width=12,
                          voicing_threshold=DEFAULT_VOICING_THRESHOLD,
                          onset_penalty=DEFAULT_ONSET_PENALTY):
    """Realtime (greedy) Viterbi — matches the C/WASM deployment exactly.

    Unlike the offline version, this processes frames left-to-right and
    emits the best state immediately at each frame, without backtracing.
    This is what runs in real-time in the browser.

    The tradeoff: realtime Viterbi can't "change its mind" about earlier
    frames when it sees future evidence. In practice, the difference is
    small for well-trained models, but it can occasionally miss brief
    voiced segments or produce slightly less smooth pitch tracks.

    Same transition model as offline:
      - 360 voiced bins + 1 unvoiced state
      - ±transition_width bin transitions allowed per frame
      - onset/offset penalty for voiced↔unvoiced switches

    Args / Returns: same as viterbi_decode.
    """
    import numpy as np

    T, N = posteriorgram.shape
    if T == 0:
        return np.zeros(0, dtype=np.float32)

    tw = int(transition_width)
    W = 2 * tw + 1

    # Only keep the current column of scores (no backpointer storage)
    prev = np.full(N + 1, -np.inf, dtype=np.float64)
    f0_hz = np.zeros(T, dtype=np.float32)

    for t in range(T):
        max_post_t = posteriorgram[t].max()
        log_obs_t = np.log(posteriorgram[t] + 1e-10)
        uv_obs = np.log(1.0 - max_post_t + 1e-10)

        curr = np.full(N + 1, -np.inf, dtype=np.float64)

        if t == 0:
            # Initialize
            if max_post_t > voicing_threshold:
                curr[:N] = log_obs_t
            curr[N] = uv_obs
        else:
            prev_voiced = prev[:N]

            # Best predecessor within ±tw for each voiced state (vectorized)
            padded = np.pad(prev_voiced, (tw, tw), constant_values=-np.inf)
            windows = np.lib.stride_tricks.as_strided(
                padded, shape=(N, W),
                strides=(padded.strides[0], padded.strides[0]))
            best_val = np.max(windows, axis=1)

            # From unvoiced (onset penalty)
            from_uv = prev[N] - onset_penalty

            # Voiced states: best of (from voiced neighbor, from unvoiced)
            curr[:N] = np.maximum(best_val, from_uv) + log_obs_t

            # Unvoiced state: best of (stay unvoiced, from any voiced)
            from_voiced = prev_voiced.max() - onset_penalty
            curr[N] = max(prev[N], from_voiced) + uv_obs

        # Emit: pick best current state (greedy, no backtrace)
        best_state = np.argmax(curr)
        if best_state < N:
            f0_hz[t] = bin_to_f0(float(best_state))

        prev = curr

    return f0_hz


# ═══════════════════════════════════════════════════════════════════════
# The Neural Network
# ═══════════════════════════════════════════════════════════════════════

class NanoPitch(nn.Module):
    """Lightweight GRU network for real-time pitch tracking and VAD.

    Args:
        n_mels: number of mel spectrogram input bands (40)
        cond_size: width of the first conv layer (64)
        gru_size: number of units in each GRU layer (96)
    """

    def __init__(self, n_mels=N_MELS, cond_size=64, gru_size=96):
        super().__init__()
        if cond_size > MAX_LAYER_SIZE or gru_size > MAX_LAYER_SIZE:
            raise ValueError(
                f"cond_size={cond_size} or gru_size={gru_size} exceeds "
                f"MAX_LAYER_SIZE={MAX_LAYER_SIZE}. The C/WASM engine "
                f"cannot run models larger than this. Increase "
                f"NC_MAX_LAYER_SIZE in nanopitch.h if you really need it.")
        self.n_mels = n_mels
        self.cond_size = cond_size
        self.gru_size = gru_size

        # ── Causal convolutional feature extractor ──
        # Conv1d slides a small kernel (size 3) across the time axis.
        # "Causal" means we only look at the current and past frames, never
        # the future — essential for real-time streaming. We achieve this by
        # padding 2 frames on the LEFT only, so:
        #   output[t] = f(input[t-2], input[t-1], input[t])
        # Two stacked causal conv layers look at 5 frames of past context
        # total (50ms at our 10ms hop rate), with zero latency.
        self.conv1 = nn.Conv1d(n_mels, cond_size, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(cond_size, gru_size, kernel_size=3, padding=0)
        # Note: we apply left-padding manually in forward() rather than using
        # PyTorch's built-in padding, to keep the same Conv1d for C export.

        # ── Recurrent layers ──
        # GRU = Gated Recurrent Unit. Unlike a plain RNN, GRUs have "gates"
        # that control how much old information to keep vs. replace.
        # This prevents the vanishing gradient problem and allows the network
        # to learn long-range dependencies (e.g., tracking a sustained note).
        # batch_first=True means input shape is (batch, time, features).
        self.gru1 = nn.GRU(gru_size, gru_size, batch_first=True)
        self.gru2 = nn.GRU(gru_size, gru_size, batch_first=True)
        self.gru3 = nn.GRU(gru_size, gru_size, batch_first=True)

        # ── Output heads ──
        # We concatenate the outputs of all layers [conv, gru1, gru2, gru3]
        # to give the output heads access to features at every depth level.
        cat_size = gru_size * 4
        self.dense_vad = nn.Linear(cat_size, 1)           # voice activity
        self.dense_pitch = nn.Linear(cat_size, PITCH_BINS) # pitch bins

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        print(f"NanoPitch: {n_params:,} parameters "
              f"(cond={cond_size}, gru={gru_size})")

    def _init_weights(self):
        """Initialize GRU recurrent weights with orthogonal matrices.

        This is a common trick that helps GRUs learn more stably at the
        start of training, by ensuring the hidden-to-hidden weight matrices
        preserve gradient magnitudes.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.GRU):
                for pname, p in module.named_parameters():
                    if 'weight_hh' in pname:
                        nn.init.orthogonal_(p)

    def forward(self, mel, states=None):
        """Run the model on a batch of mel spectrograms.

        Args:
            mel: (batch, time, 40) — log-mel spectrogram input
            states: optional GRU hidden states (for continuing a stream)

        Returns:
            vad:    (batch, time, 1)   — voice activity probability
            pitch:  (batch, time, 360) — pitch posteriorgram
            states: list of 3 GRU hidden states

        Output has the SAME length as input thanks to causal padding.
        Each output frame only depends on current and past input frames.
        """
        B = mel.size(0)
        device = mel.device

        if states is None:
            h1 = torch.zeros(1, B, self.gru_size, device=device)
            h2 = torch.zeros(1, B, self.gru_size, device=device)
            h3 = torch.zeros(1, B, self.gru_size, device=device)
        else:
            h1, h2, h3 = states

        # Conv1d expects (batch, channels, time).
        x = mel.permute(0, 2, 1)                         # (B, 40, T)
        # Causal padding: pad 2 zeros on the LEFT for each conv (kernel=3).
        # This ensures output[t] only depends on input[t-2], input[t-1], input[t].
        x = torch.nn.functional.pad(x, (2, 0))           # (B, 40, T+2)
        x = torch.tanh(self.conv1(x))                     # (B, 64, T)
        x = torch.nn.functional.pad(x, (2, 0))           # (B, 64, T+2)
        x = torch.tanh(self.conv2(x))                     # (B, 96, T)
        x = x.permute(0, 2, 1)                            # (B, T, 96)

        # Each GRU processes the sequence and returns:
        #   output: (B, T, hidden) — one hidden state per time step
        #   h_new:  (1, B, hidden) — the final hidden state
        g1, h1 = self.gru1(x, h1)
        g2, h2 = self.gru2(g1, h2)
        g3, h3 = self.gru3(g2, h3)

        # Concatenate features from all depths (skip connections)
        cat = torch.cat([x, g1, g2, g3], dim=-1)  # (B, T, 384)

        # Output heads: sigmoid squashes to [0, 1]
        vad = torch.sigmoid(self.dense_vad(cat))      # (B, T, 1)
        pitch = torch.sigmoid(self.dense_pitch(cat))   # (B, T, 360)

        return vad, pitch, [h1, h2, h3]

    def forward_single_frame(self, mel_frame, states):
        """Process one frame at a time (for real-time/streaming use).

        With causal convolutions, we maintain a small history buffer for
        each conv layer (2 past frames each). Audio arrives every 10ms,
        and we produce one output immediately — zero added latency.
        """
        conv1_buf = states['conv1_buf']  # (1, 2, n_mels) — 2 past frames
        conv2_buf = states['conv2_buf']  # (1, 2, cond_size)
        gru_states = states['gru_states']

        # Conv1: append new frame to history, run on [past2, past1, current]
        conv1_in = torch.cat([conv1_buf, mel_frame], dim=1)  # (1, 3, n_mels)
        x = torch.tanh(self.conv1(conv1_in.permute(0, 2, 1)))  # (1, cond, 1)
        conv1_buf = conv1_in[:, 1:, :]  # keep last 2 for next call

        # Conv2: same pattern
        x_t = x.permute(0, 2, 1)  # (1, 1, cond_size)
        conv2_in = torch.cat([conv2_buf, x_t], dim=1)  # (1, 3, cond_size)
        x = torch.tanh(self.conv2(conv2_in.permute(0, 2, 1)))  # (1, gru, 1)
        conv2_buf = conv2_in[:, 1:, :]
        x = x.permute(0, 2, 1)  # (1, 1, gru_size)

        h1, h2, h3 = gru_states
        g1, h1 = self.gru1(x, h1)
        g2, h2 = self.gru2(g1, h2)
        g3, h3 = self.gru3(g2, h3)

        cat = torch.cat([x, g1, g2, g3], dim=-1)
        vad = torch.sigmoid(self.dense_vad(cat))
        pitch = torch.sigmoid(self.dense_pitch(cat))

        new_states = {
            'conv1_buf': conv1_buf,
            'conv2_buf': conv2_buf,
            'gru_states': [h1, h2, h3],
        }
        return vad, pitch, new_states

    def init_streaming_state(self, device='cpu'):
        """Create initial state for streaming inference (all zeros)."""
        return {
            'conv1_buf': torch.zeros(1, 2, self.n_mels, device=device),
            'conv2_buf': torch.zeros(1, 2, self.cond_size, device=device),
            'gru_states': [
                torch.zeros(1, 1, self.gru_size, device=device),
                torch.zeros(1, 1, self.gru_size, device=device),
                torch.zeros(1, 1, self.gru_size, device=device),
            ],
        }


# ═══════════════════════════════════════════════════════════════════════
# Quick test — run this file directly to verify the model works
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model = NanoPitch()

    # Test batch mode (used during training)
    x = torch.randn(2, 100, N_MELS)  # 2 clips, 100 frames each
    vad, pitch, states = model(x)
    print(f"Input:  {x.shape}")
    print(f"VAD:    {vad.shape}   range [{vad.min():.3f}, {vad.max():.3f}]")
    print(f"Pitch:  {pitch.shape} range [{pitch.min():.3f}, {pitch.max():.3f}]")

    # Test streaming mode (used during deployment)
    state = model.init_streaming_state()
    for t in range(10):
        frame = torch.randn(1, 1, N_MELS)
        v, p, state = model.forward_single_frame(frame, state)
    print(f"Streaming OK: vad={v.shape}, pitch={p.shape}")
