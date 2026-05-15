"""
Duration analysis for VPC datasets.
Reads utt2dur if available, otherwise computes from wav.scp directly.

Usage:
    python run_duration_analysis.py
"""

import subprocess
import io
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from utils import setup_logger

logger = setup_logger(__name__)

DATASETS = [
    "IEMOCAP_dev",
    "IEMOCAP_test",
    "libri_dev_enrolls",
    "libri_dev_trials_mixed",
    "libri_test_enrolls",
    "libri_test_trials_mixed",
]

DATA_DIR = Path("data")
OUTPUT_DIR = Path("exp/duration_analysis")


def get_durations_from_utt2dur(utt2dur_path: Path):
    durations = {}
    with open(utt2dur_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                durations[parts[0]] = float(parts[1])
    return durations


def get_duration_from_wav_scp(wav_value: str) -> float:
    wav_value = wav_value.strip()
    if wav_value.endswith("|"):
        cmd = wav_value[:-1].strip()
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        waveform, sr = torchaudio.load(io.BytesIO(result.stdout))
    else:
        waveform, sr = torchaudio.load(wav_value)
    return waveform.shape[1] / sr


def get_durations_from_wav_scp(wav_scp_path: Path):
    durations = {}
    with open(wav_scp_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        utt_id, *rest = line.split(" ")
        wav_value = " ".join(rest)
        try:
            durations[utt_id] = get_duration_from_wav_scp(wav_value)
        except Exception as e:
            logger.warning(f"Skipping {utt_id}: {e}")
        if (i + 1) % 200 == 0:
            logger.info(f"  {i+1}/{len(lines)} done")
    return durations


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_durations = {}

    for dataset in DATASETS:
        data_path = DATA_DIR / dataset
        utt2dur = data_path / "utt2dur"

        if utt2dur.exists():
            logger.info(f"[{dataset}] Reading from utt2dur...")
            durations = get_durations_from_utt2dur(utt2dur)
        else:
            logger.info(f"[{dataset}] No utt2dur found, computing from wav.scp...")
            durations = get_durations_from_wav_scp(data_path / "wav.scp")

        all_durations[dataset] = list(durations.values())

    # Print summary
    print("\n---- Duration Summary ----")
    print(f"{'dataset':<35} {'mean (s)':>10} {'min (s)':>8} {'max (s)':>8} {'total (h)':>10} {'n':>6}")
    print("-" * 80)
    for dataset, durs in all_durations.items():
        print(f"{dataset:<35} {sum(durs)/len(durs):>10.2f} {min(durs):>8.2f} {max(durs):>8.2f} {sum(durs)/3600:>10.2f} {len(durs):>6}")

    # Plot distribution
    n_datasets = len(DATASETS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, dataset in enumerate(DATASETS):
        durs = all_durations[dataset]
        mean_dur = sum(durs) / len(durs)
        axes[i].hist(durs, bins=50, color="steelblue", edgecolor="white", linewidth=0.5)
        axes[i].axvline(mean_dur, color="red", linestyle="--", linewidth=1.5, label=f"mean={mean_dur:.2f}s")
        axes[i].set_title(dataset, fontsize=10)
        axes[i].set_xlabel("Duration (s)")
        axes[i].set_ylabel("Count")
        axes[i].legend(fontsize=9)

    plt.suptitle("Duration Distribution by Dataset", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "duration_distribution.png"
    plt.savefig(out_path, dpi=150)
    logger.info(f"Saved plot to {out_path}")
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
