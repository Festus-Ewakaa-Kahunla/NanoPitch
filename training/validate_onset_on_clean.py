#!/usr/bin/env python3
"""
Validate onset_penalty choice on a held-out slice of clean.npz.

Methodology:
    Our test-set sweep picked onset_penalty=0.35 as optimal. To verify
    this isn't overfitting to the specific 600 clips in test.npz, we run
    the same sweep on UNSEEN clean singing clips (last 10% of clean.npz
    segments, deterministic seed).

    If the val-set optimum matches the test-set optimum (0.35), that's
    evidence the tuning generalizes.

Note: clean.npz contains studio singing only (no noise). This validation
checks whether onset_penalty choice transfers to different clean data,
not how it performs at various SNR levels.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from model import NanoPitch, viterbi_decode_realtime


ONSET_VALUES = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0]
CLIP_FRAMES = 500       # 5 seconds at 10ms hop
VAL_FRACTION = 0.10     # last 10% of segments held out
N_VAL_CLIPS = 100       # number of clips to evaluate
RANDOM_SEED = 42        # deterministic clip selection


def build_segments(lengths, min_len):
    """Build (start, end) segment indices from the per-file lengths array."""
    segments = []
    offset = 0
    for length in lengths:
        if length >= min_len:
            segments.append((offset, offset + length))
        offset += length
    return segments


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    model_kwargs = ckpt.get('model_kwargs', {}) if isinstance(ckpt, dict) else {}
    cond_size = model_kwargs.get('cond_size', 64)
    gru_size = model_kwargs.get('gru_size', 96)
    model = NanoPitch(cond_size=cond_size, gru_size=gru_size).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="../data")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)

    # Load clean data
    clean_path = Path(args.data_dir) / "clean.npz"
    print(f"Loading {clean_path}")
    clean = np.load(str(clean_path))
    clean_mel = clean['mel']
    clean_f0 = clean['f0']
    clean_lengths = clean['lengths']

    # Deterministic train/val split: last 10% of segments reserved for validation.
    all_segments = build_segments(clean_lengths, CLIP_FRAMES)
    n_total = len(all_segments)
    val_start = int(n_total * (1 - VAL_FRACTION))
    val_segments = all_segments[val_start:]
    print(f"Total segments: {n_total}, held-out val segments: {len(val_segments)}")

    # Sample N_VAL_CLIPS clips with a deterministic seed.
    rng = np.random.default_rng(seed=RANDOM_SEED)
    n_clips = min(N_VAL_CLIPS, len(val_segments))
    seg_indices = rng.choice(len(val_segments), size=n_clips, replace=False)

    # Forward pass once per clip (slow part, independent of onset_penalty).
    cached = []
    for i in tqdm(seg_indices, desc="Forward pass", unit="clip"):
        start, end = val_segments[i]
        offset = rng.integers(0, end - start - CLIP_FRAMES + 1)
        s = start + offset
        mel = clean_mel[s:s + CLIP_FRAMES].astype(np.float32)
        f0 = clean_f0[s:s + CLIP_FRAMES].astype(np.float32)

        mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)
        with torch.no_grad():
            v, p, _ = model(mel_t)
        cached.append({
            'pred_pitch': p.squeeze(0).cpu().numpy(),
            'pred_vad': v.squeeze(0).cpu().numpy().squeeze(-1),
            'f0_ref': f0,
        })

    # Sweep onset_penalty (Viterbi only, fast).
    print("\n--- Validation onset sweep (held-out clean clips) ---")
    results = {}
    for op in ONSET_VALUES:
        rpa_list, vdr_list, gross_list = [], [], []
        for row in cached:
            f0_dec = viterbi_decode_realtime(row['pred_pitch'], onset_penalty=op)
            f0_ref = row['f0_ref']
            vg = f0_ref > 0
            vp = f0_dec > 0
            both = vg & vp

            if vg.sum() > 0:
                vdr = float(np.mean(vp[vg]))
            else:
                vdr = float('nan')

            if both.sum() > 0:
                cents = np.abs(1200 * np.log2(f0_dec[both] / (f0_ref[both] + 1e-10) + 1e-10))
                rpa = float(np.mean(cents < 50))
                gross = float(np.mean(cents >= 50))
            else:
                rpa = float('nan')
                gross = float('nan')

            rpa_list.append(rpa)
            vdr_list.append(vdr)
            gross_list.append(gross)

        results[op] = {
            'rpa': float(np.nanmean(rpa_list)),
            'vdr': float(np.nanmean(vdr_list)),
            'gross': float(np.nanmean(gross_list)),
        }

    # Ranked table
    print("\n" + "=" * 60)
    print("  HELD-OUT VAL-SET RANKING (by RPA)")
    print("=" * 60)
    print(f"  {'Onset':>6}  {'Val RPA':>9}  {'Val VDR':>9}  {'Val Gross':>10}")
    ranked = sorted(results.items(), key=lambda kv: -kv[1]['rpa'])
    for op, m in ranked:
        print(f"  {op:>6.2f}  {m['rpa']*100:>8.2f}%  "
              f"{m['vdr']*100:>8.2f}%  {m['gross']*100:>9.2f}%")

    winner = ranked[0][0]
    print(f"\n  Val-set optimal onset_penalty = {winner}")
    print(f"  (compare to test-set optimal: 0.35)")

    if abs(winner - 0.35) <= 0.1:
        print("  ✓ Val and test optima agree within 0.1 — strong evidence no test-set overfitting.")
    elif abs(winner - 0.35) <= 0.2:
        print("  ~ Val and test optima close but not identical — mild overfitting possible.")
    else:
        print("  ✗ Val and test optima differ noticeably — test-set overfitting likely.")

    # Save JSON
    out_path = Path(args.checkpoint).parent.parent / "val_onset_sweep.json"
    with open(out_path, 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
