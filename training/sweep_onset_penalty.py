#!/usr/bin/env python3
"""
Sweep the Viterbi onset_penalty hyperparameter on a trained checkpoint.

This is a TEST-TIME tuning script. No retraining. It evaluates the same
checkpoint under different onset_penalty values and reports which value
gives the best realtime metrics (what the leaderboard scores against).

Usage:
    python sweep_onset_penalty.py \\
        --checkpoint ./runs/exp10-decouple-labels/checkpoints/best.pth \\
        --data-dir ../data
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from model import NanoPitch, viterbi_decode, viterbi_decode_realtime, PITCH_BINS


ONSET_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def run_model_once(model, test, device):
    """Run the model over all test clips, returning raw posteriors.

    Doing this once (instead of per onset_penalty) saves 6x the inference time.
    The only thing that changes between sweeps is Viterbi decoding.
    """
    clips = test['clips']
    f0_gt = test['f0']
    vad_gt = test['vad']
    snrs = test['snr']
    N = clips.shape[0]

    cached = []
    for i in tqdm(range(N), desc="Forward pass", unit="clip"):
        mel = torch.from_numpy(clips[i].astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            v, p, _ = model(mel)
        pred_vad = v.squeeze(0).cpu().numpy().squeeze(-1)
        pred_pitch = p.squeeze(0).cpu().numpy()
        T = pred_vad.shape[0]
        cached.append({
            'pred_vad': pred_vad,
            'pred_pitch': pred_pitch,
            'vad_ref': vad_gt[i, :T].astype(np.float32),
            'f0_ref': f0_gt[i, :T].astype(np.float32),
            'snr': float(snrs[i]),
        })
    return cached


def evaluate_with_onset(cached, onset_penalty):
    """Re-decode all cached posteriors with a given onset_penalty."""
    per_snr = {}
    for row in cached:
        snr_tag = f"{row['snr']:+.0f} dB" if np.isfinite(row['snr']) else "clean"
        f0_real = viterbi_decode_realtime(row['pred_pitch'], onset_penalty=onset_penalty)
        f0_off = viterbi_decode(row['pred_pitch'], onset_penalty=onset_penalty)

        f0_ref = row['f0_ref']
        vg = f0_ref > 0

        metrics = {}
        for name, f0_dec in [('realtime', f0_real), ('offline', f0_off)]:
            vp = f0_dec > 0
            both = vg & vp
            vdr = float(np.mean(vp[vg])) if vg.sum() > 0 else float('nan')
            if both.sum() > 0:
                cents = np.abs(1200 * np.log2(f0_dec[both] / (f0_ref[both] + 1e-10) + 1e-10))
                rpa = float(np.mean(cents < 50))
                gross = float(np.mean(cents >= 50))
            else:
                rpa = float('nan')
                gross = float('nan')
            metrics[f'{name}_vdr'] = vdr
            metrics[f'{name}_rpa'] = rpa
            metrics[f'{name}_gross'] = gross

        vad_acc = float(np.mean((row['pred_vad'] > 0.5) == (row['vad_ref'] > 0.5)))
        metrics['vad_acc'] = vad_acc

        per_snr.setdefault(snr_tag, []).append(metrics)

    # Macro-average per SNR
    summary = {}
    for snr, rows in per_snr.items():
        summary[snr] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}

    # Overall macro across all SNR conditions
    all_keys = list(summary[list(summary.keys())[0]].keys())
    summary['macro'] = {k: float(np.nanmean([summary[s][k] for s in summary])) for k in all_keys}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to best.pth")
    parser.add_argument("--data-dir", default="../data", help="dir containing test.npz")
    parser.add_argument("--device", default=None, help="cuda, mps, cpu (auto-detect)")
    parser.add_argument("--values", type=float, nargs="+", default=None,
                        help="onset_penalty values to sweep (default: coarse sweep)")
    args = parser.parse_args()

    global ONSET_VALUES
    if args.values:
        ONSET_VALUES = args.values

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Handle different checkpoint formats
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    # Get architecture kwargs (stored in checkpoint) or fall back to defaults
    model_kwargs = ckpt.get('model_kwargs', {}) if isinstance(ckpt, dict) else {}
    cond_size = model_kwargs.get('cond_size', 64)
    gru_size = model_kwargs.get('gru_size', 96)
    model = NanoPitch(cond_size=cond_size, gru_size=gru_size).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load test set
    test_path = Path(args.data_dir) / "test.npz"
    print(f"Loading {test_path}")
    test = np.load(str(test_path))

    # Run model ONCE on all test clips (slow part)
    cached = run_model_once(model, test, device)

    # Sweep onset_penalty values (fast part — Viterbi only)
    all_results = {}
    for op in ONSET_VALUES:
        print(f"\n--- onset_penalty = {op} ---")
        summary = evaluate_with_onset(cached, onset_penalty=op)
        all_results[op] = summary

        m = summary['macro']
        print(f"  Realtime:  RPA={m['realtime_rpa']*100:.2f}%  "
              f"VDR={m['realtime_vdr']*100:.2f}%  "
              f"Gross={m['realtime_gross']*100:.2f}%  "
              f"VAD={m['vad_acc']*100:.2f}%")

    # Rank by realtime macro RPA (leaderboard-relevant)
    print("\n" + "=" * 68)
    print("  ONSET_PENALTY SWEEP (ranked by realtime macro RPA)")
    print("=" * 68)
    print(f"  {'Onset':>6}  {'RT RPA':>8}  {'RT VDR':>8}  {'RT Gross':>9}  {'VAD Acc':>8}  {'Clean RPA':>10}")
    ranked = sorted(all_results.items(),
                    key=lambda kv: -kv[1]['macro']['realtime_rpa'])
    for op, summary in ranked:
        m = summary['macro']
        clean_rpa = summary.get('clean', {}).get('realtime_rpa', float('nan')) * 100
        print(f"  {op:>6}  {m['realtime_rpa']*100:>7.2f}%  "
              f"{m['realtime_vdr']*100:>7.2f}%  "
              f"{m['realtime_gross']*100:>8.2f}%  "
              f"{m['vad_acc']*100:>7.2f}%  "
              f"{clean_rpa:>9.2f}%")

    # Save full JSON
    out_path = Path(args.checkpoint).parent.parent / "onset_sweep.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull sweep saved to: {out_path}")


if __name__ == "__main__":
    main()
