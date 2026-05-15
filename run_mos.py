"""
MOS evaluation using UTMOS (https://github.com/tarepan/SpeechMOS)
Reads Kaldi-style wav.scp files and outputs MOS scores per utterance and dataset.

Usage:
    python run_mos.py --datasets libri_dev libri_dev_mcadams IEMOCAP_dev IEMOCAP_dev_mcadams
    python run_mos.py --datasets libri_dev --data_dir data --output_csv exp/mos_results.csv
"""

import argparse
import logging
import subprocess
import io
import csv
from pathlib import Path

import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_wav_from_scp_entry(value: str):
    """Load wav from a wav.scp value (file path or pipe command like 'flac -c -d -s xxx.flac |')."""
    value = value.strip()
    if value.endswith("|"):
        cmd = value[:-1].strip()
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise IOError(f"Command failed: {cmd}\n{result.stderr.decode()}")
        waveform, sr = torchaudio.load(io.BytesIO(result.stdout))
    else:
        waveform, sr = torchaudio.load(value)
    return waveform, sr


def read_wav_scp(wav_scp_path: Path):
    """Parse wav.scp and return {utt_id: wav_value} dict."""
    entries = {}
    with open(wav_scp_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, *rest = line.split(" ")
            entries[utt_id] = " ".join(rest)
    return entries


def compute_mos(model, waveform: torch.Tensor, sr: int, device: str) -> float:
    """Resample to 16kHz if needed and run UTMOS."""
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.mean(dim=0, keepdim=True)  # mono
    waveform = waveform.to(device)
    with torch.no_grad():
        score = model(waveform, sr=16000)
    return score.item()


def main():
    parser = argparse.ArgumentParser()
    default_datasets = [
        "libri_dev", "libri_test",
        "IEMOCAP_dev", "IEMOCAP_test",
    ]
    parser.add_argument("--datasets", nargs="+", default=default_datasets,
                        help="Dataset folder names under data_dir (default: original datasets only)")
    parser.add_argument("--data_dir", default="data", help="Root data directory (default: data)")
    parser.add_argument("--output_csv", default="exp/mos_results.csv", help="Output CSV path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_utts", type=int, default=None,
                        help="Limit utterances per dataset (for quick testing)")
    args = parser.parse_args()

    logger.info(f"Loading UTMOS model on {args.device}...")
    model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    model = model.to(args.device)
    model.eval()

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for dataset in args.datasets:
        wav_scp = Path(args.data_dir) / dataset / "wav.scp"
        if not wav_scp.exists():
            logger.warning(f"wav.scp not found: {wav_scp}, skipping.")
            continue

        entries = read_wav_scp(wav_scp)
        if args.max_utts:
            entries = dict(list(entries.items())[:args.max_utts])

        logger.info(f"[{dataset}] Processing {len(entries)} utterances...")
        scores = []
        for i, (utt_id, wav_value) in enumerate(entries.items()):
            try:
                waveform, sr = load_wav_from_scp_entry(wav_value)
                score = compute_mos(model, waveform, sr, args.device)
                scores.append(score)
                all_rows.append({"dataset": dataset, "utt_id": utt_id, "MOS": round(score, 4)})
                if (i + 1) % 100 == 0:
                    logger.info(f"  {i+1}/{len(entries)} done, avg so far: {sum(scores)/len(scores):.3f}")
            except Exception as e:
                logger.warning(f"  Skipping {utt_id}: {e}")

        if scores:
            avg = sum(scores) / len(scores)
            logger.info(f"[{dataset}] Mean MOS: {avg:.4f} (n={len(scores)})")

    # Save per-utterance CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "utt_id", "MOS"])
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info(f"Saved per-utterance results to {output_path}")

    # Print summary table
    print("\n---- MOS Summary ----")
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[row["dataset"]].append(row["MOS"])
    print(f"{'dataset':<45} {'mean MOS':>10} {'n':>6}")
    print("-" * 65)
    for dataset, scores in grouped.items():
        print(f"{dataset:<45} {sum(scores)/len(scores):>10.4f} {len(scores):>6}")


if __name__ == "__main__":
    main()
