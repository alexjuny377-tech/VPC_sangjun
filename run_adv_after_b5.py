"""
B5 익명화 결과에 adversarial perturbation을 적용하는 스크립트.

단일 GPU:
    python run_adv_after_b5.py

2장 병렬 (터미널 2개):
    CUDA_VISIBLE_DEVICES=0 python run_adv_after_b5.py --shard 0 --num_shards 2
    CUDA_VISIBLE_DEVICES=1 python run_adv_after_b5.py --shard 1 --num_shards 2

흐름:
    data/*_asrbn_hifigan_bn_tdnnf_wav2vec2_vq_48_v1/wav/*.wav
    → adversarial perturbation (발화마다 랜덤 target)
    → data/*_asrbn_hifigan_bn_tdnnf_wav2vec2_vq_48_v1_adv/wav/*.wav

평가:
    python run_evaluation.py --config configs/track1/eval_pre.yaml \\
        --overwrite '{"anon_data_suffix": "_asrbn_hifigan_bn_tdnnf_wav2vec2_vq_48_v1_adv"}'
"""

import argparse
import glob
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

from adversarial_perturbation import (
    AdversarialPerturbationOptimizer,
    build_model,
)

B5_SUFFIX = "_asrbn_hifigan_bn_tdnnf_wav2vec2_vq_48_v1"
ADV_SUFFIX = B5_SUFFIX + "_adv"

DEFAULT_CONFIG = {
    'eps':       0.005,
    'alpha':     1.0,
    'beta':      0.1,
    'gamma':     0.05,
    'n_iters':   20,
    'lr':        5e-3,   # 300iter 1e-3 대비 수렴 속도 보완
    'proj_type': 'linf',
    'verbose':   True,
    'log_every': 5,
}


def copy_metadata(src_dir: Path, dst_dir: Path):
    """wav.scp 경로 수정 + 나머지 메타데이터 파일 복사."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["utt2spk", "spk2utt", "spk2gender", "text", "utt2dur", "enrolls"]:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)

    wav_scp = src_dir / "wav.scp"
    if wav_scp.exists():
        content = wav_scp.read_text()
        updated = content.replace(B5_SUFFIX, ADV_SUFFIX)
        (dst_dir / "wav.scp").write_text(updated)


def apply_perturbation_to_dataset(
    src_dir: Path,
    dst_dir: Path,
    attacker: AdversarialPerturbationOptimizer,
    device: torch.device,
    shard: int = 0,
    num_shards: int = 1,
):
    wav_dir = src_dir / "wav"
    out_wav_dir = dst_dir / "wav"
    out_wav_dir.mkdir(parents=True, exist_ok=True)

    all_wav_files = sorted(wav_dir.glob("*.wav"))
    wav_files = all_wav_files[shard::num_shards]   # 인터리브 방식으로 분할
    print(f"\n[{src_dir.name}] shard {shard}/{num_shards}  ({len(wav_files)}/{len(all_wav_files)} 발화)")

    for i, wav_path in enumerate(wav_files):
        out_path = out_wav_dir / wav_path.name
        if out_path.exists():                                  # 이미 처리된 파일 건너뜀
            continue

        x, sr = torchaudio.load(wav_path)
        x = x.to(device)                                     # (1, T)

        with torch.no_grad():
            target_emb = F.normalize(torch.randn(1, 192, device=device), dim=1)

        x_adv, _ = attacker.run(x, target_emb)
        torchaudio.save(str(out_path), x_adv.cpu(), sr)

        if (i + 1) % 10 == 0 or (i + 1) == len(wav_files):
            print(f"  {i+1}/{len(wav_files)} 완료")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="exp/asv_ssl",
                        help="VPC ASV 체크포인트 폴더")
    parser.add_argument("--data_dir", default="data",
                        help="VPC data 루트 디렉토리")
    parser.add_argument("--dummy", action="store_true",
                        help="실제 모델 없이 Dummy 모델로 테스트")
    parser.add_argument("--shard", type=int, default=0,
                        help="현재 프로세스 번호 (0-based)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="총 병렬 프로세스 수")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  shard {args.shard}/{args.num_shards}")

    ckpt_dir = None if args.dummy else args.checkpoint_dir
    model    = build_model(ckpt_dir).to(device)
    attacker = AdversarialPerturbationOptimizer(model, DEFAULT_CONFIG)

    data_root = Path(args.data_dir)
    b5_dirs   = sorted(data_root.glob(f"*{B5_SUFFIX}"))

    if not b5_dirs:
        print(f"B5 데이터 없음: {data_root}/*{B5_SUFFIX}")
        return

    print(f"처리할 데이터셋 {len(b5_dirs)}개:")
    for d in b5_dirs:
        print(f"  {d.name}")

    for src_dir in b5_dirs:
        dst_dir = data_root / src_dir.name.replace(B5_SUFFIX, ADV_SUFFIX)
        if args.shard == 0:          # 메타데이터는 shard 0만 복사
            copy_metadata(src_dir, dst_dir)
        else:
            (dst_dir / "wav").mkdir(parents=True, exist_ok=True)
        apply_perturbation_to_dataset(
            src_dir, dst_dir, attacker, device,
            shard=args.shard, num_shards=args.num_shards,
        )

    print("\n" + "="*60)
    print("완료! 평가 실행 명령:")
    print(
        f'python run_evaluation.py --config configs/track1/eval_pre.yaml \\\n'
        f'    --overwrite \'{{"anon_data_suffix": "{ADV_SUFFIX}"}}\' --force_compute True'
    )


if __name__ == "__main__":
    main()
