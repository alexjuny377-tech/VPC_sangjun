"""
WER evaluation for libri enroll/trial datasets using the existing ASR model.

Usage:
    python run_wer_enroll_trial.py
"""

import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

sys.path.insert(0, str(Path(__file__).parent))
from evaluation.utility.asr.speechbrain_asr import InferenceSpeechBrainASR
from evaluation.utility.asr.speechbrain_asr.inference import ASRDataset
from utils import read_kaldi_format, setup_logger, scan_checkpoint

logger = setup_logger(__name__)

DATASETS = [
    "libri_dev_enrolls",
    "libri_dev_trials_mixed",
    "libri_test_enrolls",
    "libri_test_trials_mixed",
]

DATA_DIR = Path("data")
MODEL_DIR = Path("exp/asr")
RESULTS_DIR = Path("exp/asr")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8


def main():
    model_path = scan_checkpoint(MODEL_DIR, 'CKPT') or MODEL_DIR
    logger.info(f"Loading ASR model from {model_path} on {DEVICE}")
    model = InferenceSpeechBrainASR(
        model_path=model_path,
        asr_hparams="hyperparams.yaml",
        model_type="EncoderASR",
        device=DEVICE,
    )

    print("\n---- WER Results (enroll / trial) ----")
    print(f"{'dataset':<35} {'WER':>8}")
    print("-" * 45)

    for dataset in DATASETS:
        data_path = DATA_DIR / dataset
        out_dir = RESULTS_DIR / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        wer_file = out_dir / "wer"
        text_file = out_dir / "text"

        references = read_kaldi_format(data_path / "text", values_as_string=True)

        if wer_file.exists() and text_file.exists():
            logger.info(f"[{dataset}] Using cached results.")
            hypotheses = read_kaldi_format(text_file, values_as_string=True)
            scores = model.compute_wer(ref_texts=references, hyp_texts=hypotheses, out_file=wer_file)
        else:
            logger.info(f"[{dataset}] Running ASR inference...")
            dataset_obj = ASRDataset(wav_scp_file=data_path / "wav.scp", asr_model=model.asr_model)
            dataloader = DataLoader(dataset_obj, batch_size=BATCH_SIZE, shuffle=False,
                                    num_workers=1, collate_fn=dataset_obj.collate_fn)
            hypotheses = model.transcribe_audios(data=dataloader, out_file=text_file)
            scores = model.compute_wer(ref_texts=references, hyp_texts=hypotheses, out_file=wer_file)

        wer = scores.summarize("error_rate")
        print(f"{dataset:<35} {wer:>8.2f}%")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
