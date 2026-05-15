"""
WER + MOS analysis for train-clean-360.

Usage:
    python run_train360_analysis.py
    python run_train360_analysis.py --skip_wer   # MOS only
    python run_train360_analysis.py --skip_mos   # WER only
    python run_train360_analysis.py --max_utts 1000  # quick test
"""

import argparse
import csv
import io
import subprocess
import sys
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from evaluation.utility.asr.speechbrain_asr import InferenceSpeechBrainASR
from evaluation.utility.asr.speechbrain_asr.inference import ASRDataset
from utils import read_kaldi_format, setup_logger, scan_checkpoint

logger = setup_logger(__name__)

DATASET = "train-clean-360"
DATA_DIR = Path("data")
ASR_MODEL_DIR = Path("exp/asr")
OUTPUT_DIR = Path("exp/train360_analysis")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_wer(data_path, max_utts):
    out_dir = OUTPUT_DIR / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)
    wer_file = out_dir / "wer"
    text_file = out_dir / "text"

    model_path = scan_checkpoint(ASR_MODEL_DIR, 'CKPT') or ASR_MODEL_DIR
    logger.info(f"Loading ASR model from {model_path}")
    model = InferenceSpeechBrainASR(
        model_path=model_path,
        asr_hparams="hyperparams.yaml",
        model_type="EncoderASR",
        device=DEVICE,
    )

    references = read_kaldi_format(data_path / "text", values_as_string=True)
    if max_utts:
        references = dict(list(references.items())[:max_utts])

    if wer_file.exists() and text_file.exists() and not max_utts:
        logger.info("Using cached WER results.")
        hypotheses = read_kaldi_format(text_file, values_as_string=True)
        scores = model.compute_wer(ref_texts=references, hyp_texts=hypotheses, out_file=wer_file)
    else:
        wav_scp = data_path / "wav.scp"
        dataset_obj = ASRDataset(wav_scp_file=wav_scp, asr_model=model.asr_model)
        if max_utts:
            dataset_obj.data = dataset_obj.data[:max_utts]
        dataloader = DataLoader(dataset_obj, batch_size=8, shuffle=False,
                                num_workers=1, collate_fn=dataset_obj.collate_fn)
        hypotheses = model.transcribe_audios(data=dataloader, out_file=text_file)
        scores = model.compute_wer(ref_texts=references, hyp_texts=hypotheses, out_file=wer_file)

    wer = scores.summarize("error_rate")
    print(f"\n[WER] {DATASET}: {wer:.2f}%")
    return wer


def load_wav(wav_value):
    wav_value = wav_value.strip()
    if wav_value.endswith("|"):
        cmd = wav_value[:-1].strip()
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return torchaudio.load(io.BytesIO(result.stdout))
    return torchaudio.load(wav_value)


def run_mos(data_path, max_utts):
    logger.info("Loading UTMOS model...")
    model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    model = model.to(DEVICE).eval()

    entries = {}
    with open(data_path / "wav.scp") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, *rest = line.split(" ")
            entries[utt_id] = " ".join(rest)
    if max_utts:
        entries = dict(list(entries.items())[:max_utts])

    scores = []
    rows = []
    for i, (utt_id, wav_value) in enumerate(entries.items()):
        try:
            waveform, sr = load_wav(wav_value)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            waveform = waveform.mean(dim=0, keepdim=True).to(DEVICE)
            with torch.no_grad():
                score = model(waveform, sr=16000).item()
            scores.append(score)
            rows.append({"dataset": DATASET, "utt_id": utt_id, "MOS": round(score, 4)})
        except Exception as e:
            logger.warning(f"Skipping {utt_id}: {e}")
        if (i + 1) % 1000 == 0:
            logger.info(f"  MOS {i+1}/{len(entries)}, avg={sum(scores)/len(scores):.3f}")

    out_csv = OUTPUT_DIR / "mos_train360.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "utt_id", "MOS"])
        writer.writeheader()
        writer.writerows(rows)

    mean_mos = sum(scores) / len(scores)
    print(f"\n[MOS] {DATASET}: {mean_mos:.4f} (n={len(scores)})")
    return scores


def plot_distributions(durs, mos_scores):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(durs, bins=50, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[0].axvline(sum(durs)/len(durs), color="red", linestyle="--",
                    label=f"mean={sum(durs)/len(durs):.2f}s")
    axes[0].set_title(f"{DATASET} - Duration Distribution")
    axes[0].set_xlabel("Duration (s)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    if mos_scores:
        axes[1].hist(mos_scores, bins=50, color="darkorange", edgecolor="white", linewidth=0.5)
        axes[1].axvline(sum(mos_scores)/len(mos_scores), color="red", linestyle="--",
                        label=f"mean={sum(mos_scores)/len(mos_scores):.3f}")
        axes[1].set_title(f"{DATASET} - MOS Distribution")
        axes[1].set_xlabel("MOS")
        axes[1].set_ylabel("Count")
        axes[1].legend()

    plt.tight_layout()
    out_path = OUTPUT_DIR / "train360_distribution.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_wer", action="store_true")
    parser.add_argument("--skip_mos", action="store_true")
    parser.add_argument("--max_utts", type=int, default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATA_DIR / DATASET

    # Duration (from utt2dur, instant)
    durs = []
    with open(data_path / "utt2dur") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                durs.append(float(parts[1]))
    if args.max_utts:
        durs = durs[:args.max_utts]
    print(f"\n---- {DATASET} Summary ----")
    print(f"n:        {len(durs)}")
    print(f"mean dur: {sum(durs)/len(durs):.2f}s")
    print(f"min dur:  {min(durs):.2f}s")
    print(f"max dur:  {max(durs):.2f}s")
    print(f"total:    {sum(durs)/3600:.2f}h")

    wer = run_wer(data_path, args.max_utts) if not args.skip_wer else None
    mos_scores = run_mos(data_path, args.max_utts) if not args.skip_mos else []

    plot_distributions(durs, mos_scores)


if __name__ == "__main__":
    main()
