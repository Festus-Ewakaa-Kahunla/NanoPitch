#!/usr/bin/env python3
"""
NanoPitch Evaluation Script
===========================

Evaluates a trained NanoPitch model on the test set and produces a report
showing performance at each noise level. This helps you understand:

  - How well your model tracks pitch in clean conditions vs. noisy ones
  - Whether VAD (voice detection) degrades gracefully with more noise
  - Where the model's pitch errors tend to occur

The test set contains clips pre-mixed at specific SNR levels:
  -5 dB (very noisy), 0 dB, +5 dB, +10 dB, +20 dB (mostly clean), and clean

Metrics explained:
  VAD Acc  — What fraction of frames have correct voice/silence labels?
  VDR      — Voicing Detection Rate: of truly voiced frames, how many detected?
  RPA      — Raw Pitch Accuracy: of voiced frames, how many within 50 cents?
  RCA      — Raw Chroma Accuracy: like RPA but octave-invariant, using the
             circular distance on the 1200-cent octave (wraps both ways).
  Gross    — Gross error rate: frames with pitch error > 50 cents
  Med.c    — Median pitch error in cents (lower is better; 100 cents = 1 semitone)

Usage:
    python evaluate.py --checkpoint runs/exp1/checkpoints/best.pth --data-dir ../data
    python evaluate.py --checkpoint runs/exp1/checkpoints/best.pth --data-dir ../data --csv results.csv
    python evaluate.py --checkpoint runs/exp1/checkpoints/best.pth --data-dir ../data --json results.json
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from model import (NanoPitch, viterbi_decode, viterbi_decode_realtime,
                   DEFAULT_ONSET_PENALTY, DEFAULT_VOICING_THRESHOLD,
                   PITCH_BINS)


def _pitch_metrics(f0_dec, f0_ref):
    """Compute pitch metrics between decoded and reference f0 tracks."""
    voiced_gt = f0_ref > 0
    voiced_pred = f0_dec > 0
    vdr = float(np.mean(voiced_pred[voiced_gt])) if voiced_gt.sum() > 0 else np.nan

    both = voiced_gt & voiced_pred
    if both.sum() > 0:
        cents_err = np.abs(1200.0 * np.log2(
            f0_dec[both] / (f0_ref[both] + 1e-10) + 1e-10))
        rpa = float(np.mean(cents_err < 50))
        chroma_err = np.mod(cents_err, 1200.0)
        chroma_circular = np.minimum(chroma_err, 1200.0 - chroma_err)
        rca = float(np.mean(chroma_circular < 50))
        gross_err = float(np.mean(cents_err > 50))
        median_cents = float(np.median(cents_err))
    else:
        rpa = rca = gross_err = median_cents = np.nan

    return {'vdr': vdr, 'rpa': rpa, 'rca': rca,
            'gross_err': gross_err, 'median_cents': median_cents}


def evaluate_model(model, test_path, device='cpu', transition_width=12,
                   voicing_threshold=DEFAULT_VOICING_THRESHOLD,
                   onset_penalty=DEFAULT_ONSET_PENALTY,
                   vad_threshold=0.5, vad_reference="vad"):
    """Run model on test.npz and return per-clip results.

    For each clip, reports metrics using BOTH decoders:
      - Viterbi (Offline): full backtrace, globally optimal — upper bound
      - Viterbi (Realtime): greedy, matches C/WASM deployment — what users see

    Returns:
        list of dicts with keys: snr, vad_acc, and for each decoder:
        {vdr, rpa, rca, gross_err, median_cents} prefixed with 'offline_' or 'realtime_'
    """
    model.eval()
    test = np.load(test_path)

    clips = test['clips']       # (N, T, 40) fp16
    f0_gt = test['f0']          # (N, T) fp16 — ground-truth f0 in Hz
    vad_gt = test['vad']        # (N, T) fp16
    snrs = test['snr']          # (N,) fp32
    N = clips.shape[0]

    results = []
    for i in tqdm(range(N), desc="Evaluating", unit="clip"):
        mel = torch.from_numpy(clips[i].astype(np.float32)).unsqueeze(0)
        mel = mel.to(device)

        with torch.no_grad():
            v, p, _ = model(mel)

        pred_vad = v.squeeze(0).cpu().numpy().squeeze(-1)
        pred_pitch = p.squeeze(0).cpu().numpy()
        T = pred_vad.shape[0]

        vad_ref = vad_gt[i, :T].astype(np.float32)
        f0_ref = f0_gt[i, :T].astype(np.float32)  # ground-truth f0 in Hz

        # Decode predictions with BOTH decoders
        f0_offline = viterbi_decode(
            pred_pitch,
            transition_width=transition_width,
            voicing_threshold=voicing_threshold,
            onset_penalty=onset_penalty,
        )
        f0_realtime = viterbi_decode_realtime(
            pred_pitch,
            transition_width=transition_width,
            voicing_threshold=voicing_threshold,
            onset_penalty=onset_penalty,
        )

        vad_label = vad_ref > 0.5
        f0_label = f0_ref > 0
        if vad_reference == "vad":
            vad_ref_label = vad_label
        elif vad_reference == "f0":
            vad_ref_label = f0_label
        elif vad_reference == "union":
            vad_ref_label = vad_label | f0_label
        elif vad_reference == "intersection":
            vad_ref_label = vad_label & f0_label
        else:
            raise ValueError(f"unknown VAD reference: {vad_reference}")

        vad_acc = float(np.mean((pred_vad > vad_threshold) == vad_ref_label))

        row = {'clip': i, 'snr': float(snrs[i]), 'vad_acc': vad_acc}
        for prefix, f0_dec in [('offline', f0_offline), ('realtime', f0_realtime)]:
            m = _pitch_metrics(f0_dec, f0_ref)
            for k, val in m.items():
                row[f'{prefix}_{k}'] = val
        results.append(row)

    return results


def format_snr(snr):
    """Format SNR for display."""
    if not np.isfinite(snr):
        return "clean"
    return f"{snr:+.0f} dB"


def print_report(results):
    """Print per-level summary showing both offline and realtime Viterbi."""
    by_snr = {}
    for r in results:
        by_snr.setdefault(r['snr'], []).append(r)

    def sm(vals):
        vals = [v for v in vals if not np.isnan(v)]
        return np.mean(vals) if vals else float('nan')

    sorted_snrs = sorted(by_snr.keys(), key=lambda x: x if np.isfinite(x) else 1e6)

    for decoder, prefix in [("Viterbi (Offline)", "offline"),
                            ("Viterbi (Realtime — matches browser)", "realtime")]:
        print()
        print("=" * 72)
        print(f"  NanoPitch — {decoder}")
        print("=" * 72)
        print()
        print(f"  {'Condition':<10s}  {'VAD Acc':>8s}  {'VDR':>8s}  "
              f"{'RPA':>8s}  {'RCA':>8s}  {'Gross':>8s}  {'Med.¢':>8s}")
        print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

        for snr in sorted_snrs:
            clips = by_snr[snr]
            tag = format_snr(snr)
            vad = sm([c['vad_acc'] for c in clips])
            vdr = sm([c[f'{prefix}_vdr'] for c in clips])
            rpa = sm([c[f'{prefix}_rpa'] for c in clips])
            rca = sm([c[f'{prefix}_rca'] for c in clips])
            gross = sm([c[f'{prefix}_gross_err'] for c in clips])
            med = sm([c[f'{prefix}_median_cents'] for c in clips])
            print(f"  {tag:<10s}  {vad:8.1%}  {vdr:8.1%}  "
                  f"{rpa:8.1%}  {rca:8.1%}  {gross:8.1%}  {med:8.1f}")

        # Overall
        print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
        vad = sm([c['vad_acc'] for c in results])
        vdr = sm([c[f'{prefix}_vdr'] for c in results])
        rpa = sm([c[f'{prefix}_rpa'] for c in results])
        rca = sm([c[f'{prefix}_rca'] for c in results])
        gross = sm([c[f'{prefix}_gross_err'] for c in results])
        med = sm([c[f'{prefix}_median_cents'] for c in results])
        print(f"  {'overall':<10s}  {vad:8.1%}  {vdr:8.1%}  "
              f"{rpa:8.1%}  {rca:8.1%}  {gross:8.1%}  {med:8.1f}")

    print()
    print(f"  {len(results)} clips evaluated")
    print(f"  Metrics: VAD Acc = voice activity accuracy")
    print(f"           VDR = voicing detection rate (recall)")
    print(f"           RPA = raw pitch accuracy (within 50 cents)")
    print(f"           RCA = raw chroma accuracy (octave-invariant, circular 1200¢ distance)")
    print(f"           Gross = gross pitch error rate (>50 cents)")
    print(f"           Med.¢ = median pitch error in cents")
    print(f"  Offline Viterbi = global optimum (backtrace)")
    print(f"  Realtime Viterbi  = greedy (matches browser deployment)")
    print()


def save_csv(results, path):
    """Save per-clip results to CSV."""
    fields = ['clip', 'snr', 'vad_acc',
              'offline_vdr', 'offline_rpa', 'offline_rca',
              'offline_gross_err', 'offline_median_cents',
              'realtime_vdr', 'realtime_rpa', 'realtime_rca',
              'realtime_gross_err', 'realtime_median_cents']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: r[k] for k in fields}
            row['snr'] = format_snr(r['snr'])
            w.writerow(row)
    print(f"Per-clip results saved to {path}")


def save_json(results, path):
    """Save per-level summary to JSON with both decoder results."""
    by_snr = {}
    for r in results:
        key = format_snr(r['snr'])
        by_snr.setdefault(key, []).append(r)

    def sm(vals):
        vals = [v for v in vals if not np.isnan(v)]
        return round(np.mean(vals), 4) if vals else None

    summary = {}
    for tag, clips in by_snr.items():
        entry = {'n_clips': len(clips),
                 'vad_acc': sm([c['vad_acc'] for c in clips])}
        for prefix in ['offline', 'realtime']:
            entry[f'{prefix}_vdr'] = sm([c[f'{prefix}_vdr'] for c in clips])
            entry[f'{prefix}_rpa'] = sm([c[f'{prefix}_rpa'] for c in clips])
            entry[f'{prefix}_rca'] = sm([c[f'{prefix}_rca'] for c in clips])
            entry[f'{prefix}_gross_err'] = sm([c[f'{prefix}_gross_err'] for c in clips])
            entry[f'{prefix}_median_cents'] = sm([c[f'{prefix}_median_cents'] for c in clips])
        summary[tag] = entry

    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="NanoPitch evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth")
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--transition-width", type=int, default=12,
                        help="max Viterbi pitch movement per frame, in pitch bins")
    parser.add_argument("--voicing-threshold", type=float,
                        default=DEFAULT_VOICING_THRESHOLD,
                        help="initial voiced-state threshold for Viterbi")
    parser.add_argument("--onset-penalty", type=float, default=DEFAULT_ONSET_PENALTY,
                        help="Viterbi voiced/unvoiced transition penalty")
    parser.add_argument("--vad-threshold", type=float, default=0.5,
                        help="threshold used when scoring the VAD output")
    parser.add_argument("--vad-reference", choices=["vad", "f0", "union", "intersection"],
                        default="vad",
                        help="reference labels used when scoring VAD accuracy")
    parser.add_argument("--csv", default=None, help="Save per-clip CSV")
    parser.add_argument("--json", default=None, help="Save summary JSON")
    args = parser.parse_args()

    # Load model
    warnings.warn(
        "Loading checkpoint via torch.load() executes Python deserialization. "
        "Only evaluate checkpoints from trusted sources.",
        RuntimeWarning,
    )
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    kwargs = ckpt.get('model_kwargs', {'cond_size': 64, 'gru_size': 96})
    model = NanoPitch(**kwargs)
    model.load_state_dict(ckpt['state_dict'])
    model.to(args.device)
    model.eval()

    # Run evaluation
    test_path = os.path.join(args.data_dir, 'test.npz')
    t0 = time.time()
    results = evaluate_model(
        model,
        test_path,
        device=args.device,
        transition_width=args.transition_width,
        voicing_threshold=args.voicing_threshold,
        onset_penalty=args.onset_penalty,
        vad_threshold=args.vad_threshold,
        vad_reference=args.vad_reference,
    )
    dt = time.time() - t0

    # Report
    print_report(results)
    print(f"  Evaluated in {dt:.1f}s")

    if args.csv:
        save_csv(results, args.csv)
    if args.json:
        save_json(results, args.json)


if __name__ == "__main__":
    main()
