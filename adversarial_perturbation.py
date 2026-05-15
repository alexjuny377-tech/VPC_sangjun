"""
Adversarial Audio Perturbation Pipeline — VoicePrivacy Challenge 2026
======================================================================
Goal: x_adv = x + δ  →  ASV embeds x_adv near a fake target speaker

Architecture under attack (white-box):
  WavLM-Large (24 layers, 1024-dim)
  + Fbank (80-dim mel)
  → ECAPA_TDNN_test (dual-path, element-wise mult fusion)
  → 192-dim L2-normalized speaker embedding

Optimization:
  L_total = α·L_adv + β·L_STFT + γ·L_STOI
  Only δ is updated; entire ASV model is frozen (eval + requires_grad=False).

⚠️  Real VPC 통합 시 주의사항:
    실제 모델의 WavLMFeatureExtractor.extract_features()는 torch.no_grad()로 감싸져 있어
    δ로의 gradient 역전파가 차단됩니다. 공격 시에는 반드시 no_grad 없이 호출해야 합니다.
    → Section 4의 RealASVModelWrapper 참고.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.optim import Adam
from typing import Optional, Tuple


# ============================================================
# Section 1: Dummy Model Definitions
#   실제 VPC 체크포인트로 교체할 수 있도록 구조를 맞춤.
#   channels=[1024,1024,1024,1024,3072], lin_neurons=192 (실제 설정값)
# ============================================================

class DummyWavLM(nn.Module):
    """
    WavLM-Large 시뮬레이터. 입력 파형 → 24개 트랜스포머 레이어 표현 출력.

    실제 모델: evaluation/privacy/asv/WavLM.py
    체크포인트: exp/asv_ssl/WavLM-Large.pt

    핵심 구조:
      CNN feature extractor (~320× 다운샘플) → Transformer 24층
      출력: (B, 24, T', 1024)  (T' ≈ T // 320)

    ⚠️  gradient 흐름을 위해 no_grad() 없이 구현.
        실제 WavLMFeatureExtractor.extract_features()는 no_grad를 사용하므로
        공격용 래퍼(RealASVModelWrapper)에서 이를 우회해야 함.
    """

    def __init__(self, num_layers: int = 24, hidden_dim: int = 1024):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # WavLM CNN feature extractor 근사 (총 stride ≈ 320)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=10, stride=5, padding=5),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=8, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=2),
            nn.GELU(),
        )  # 5 × 4 × 2 × 2 × 2 = 320× 다운샘플

        # 24개 레이어 시뮬레이션 (각각 독립적인 1×1 conv)
        self.layer_heads = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            for _ in range(num_layers)
        ])

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T)  16kHz 원시 파형
        Returns:
            layer_reps: (B, num_layers, T', hidden_dim)
        """
        x = self.cnn(waveform.unsqueeze(1))              # (B, 1024, T')
        layers = torch.stack(
            [h(x).transpose(1, 2) for h in self.layer_heads], dim=1
        )                                                 # (B, 24, T', 1024)
        return layers


class DummyFbank(nn.Module):
    """
    SpeechBrain Fbank 시뮬레이터.

    실제 모델: speechbrain.lobes.features.Fbank(n_mels=80)
               + speechbrain.processing.features.InputNormalization
    """

    def __init__(self, n_mels: int = 80, sample_rate: int = 16000,
                 n_fft: int = 400, hop_length: int = 160):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20.0,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T)
        Returns:
            fbank: (B, T', 80)  — SpeechBrain convention (time-first)
        """
        mel = self.mel(waveform).clamp(min=1e-7).log()  # (B, 80, T')
        # sentence-level mean norm (SpeechBrain InputNormalization 근사)
        mel = mel - mel.mean(dim=-1, keepdim=True)
        return mel.transpose(1, 2)                        # (B, T', 80)


class WeightedLayerSum(nn.Module):
    """
    WavLM 24개 레이어를 학습 가능한 가중치로 합산.
    ecapa_model_wavlm_24layer.py의 WeightedLayerSum과 동일 구조.
    """

    def __init__(self, num_layers: int = 24):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, layer_reps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            layer_reps: (num_layers, B, T', D)  ← 실제 모델의 내부 포맷
        Returns:
            (B, T', D)
        """
        w = F.softmax(self.layer_weights, dim=0)         # (24,)
        w = w.view(-1, 1, 1, 1)                          # (24, 1, 1, 1)
        return (layer_reps * w).sum(dim=0)               # (B, T', D)


class DummyECAPATDNN(nn.Module):
    """
    Hybrid ECAPA-TDNN 시뮬레이터. 실제 ECAPA_TDNN_test와 동일한 fusion 구조.

    실제 모델: evaluation/privacy/asv/ecapa_model_wavlm_24layer.py
    체크포인트: exp/asv_ssl/embedding_model.ckpt

    Fusion 메커니즘:
        Fbank  → blocks  (TDNN + SERes2Net×3) → x_cat
        WavLM  → blocks_features (SERes2Net×4) → x_feat_cat
        resample x_feat_cat → Fbank time axis에 맞춤
        x_mul = x_cat ⊙ x_feat_cat          (element-wise multiplication)
        x_mul → MFA → ASP → BN → FC → 192-dim embedding
    """

    def __init__(
        self,
        fbank_dim:  int = 80,
        wavlm_dim:  int = 1024,
        channel:    int = 1024,   # 실제 설정: [1024, 1024, 1024, 1024, 3072]
        emb_dim:    int = 192,
    ):
        super().__init__()

        # --- Fbank 처리 branch (self.blocks) ---
        # TDNN(input→ch) + SERes2Net×3
        self.fbank_branch = nn.Sequential(
            nn.Conv1d(fbank_dim, channel, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channel, channel, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(channel, channel, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv1d(channel, channel, kernel_size=3, padding=6, dilation=6),  # extra SE
            nn.ReLU(),
        )

        # --- WavLM 처리 branch (self.blocks_features) ---
        # SERes2Net×4 (WavLM dim → channel 먼저 맞춤)
        self.wavlm_proj = nn.Conv1d(wavlm_dim, channel, kernel_size=1)
        self.wavlm_branch = nn.Sequential(
            nn.Conv1d(channel, channel, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(channel, channel, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv1d(channel, channel, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(),
            nn.Conv1d(channel, channel, kernel_size=3, padding=8, dilation=8),
            nn.ReLU(),
        )

        # --- Multi-layer feature aggregation (MFA) ---
        # 실제: channels[-1]=3072, 여기선 channel*3 근사
        mfa_in = channel * 3   # xl[1:]을 cat하므로 ×(n_blocks-1)
        mfa_out = channel * 3
        self.mfa = nn.Conv1d(mfa_in, mfa_out, kernel_size=1)

        # --- Attentive Statistics Pooling (ASP) ---
        self.attn_fc = nn.Sequential(
            nn.Conv1d(mfa_out, 128, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(128, mfa_out, kernel_size=1),
        )

        # --- BN + FC → 192-dim embedding ---
        self.asp_bn = nn.BatchNorm1d(mfa_out * 2)
        self.fc = nn.Conv1d(mfa_out * 2, emb_dim, kernel_size=1)

    def _asp(self, x: torch.Tensor) -> torch.Tensor:
        """Attentive Statistics Pooling: weighted mean + weighted std → (B, 2C)."""
        attn = F.softmax(self.attn_fc(x), dim=2)        # (B, C, T)
        mean = (x * attn).sum(dim=2)                     # (B, C)
        std  = ((x**2 * attn).sum(dim=2) - mean**2).clamp(min=1e-6).sqrt()
        return torch.cat([mean, std], dim=1)             # (B, 2C)

    def forward(
        self,
        fbank:         torch.Tensor,          # (B, T_fbank, 80)
        wavlm_layers:  torch.Tensor,          # (B, 24, T_wavlm, 1024)
        weighted_sum:  WeightedLayerSum,
    ) -> torch.Tensor:
        """Returns L2-normalized 192-dim speaker embedding."""

        # 1) WavLM weighted layer sum
        #    wavlm_layers: (B, 24, T', 1024) → WeightedLayerSum 입력 포맷: (24, B, T', 1024)
        #    실제 코드는 WavLMExtractor 출력이 (T', 24, B, 1024)이어서 permute(1,2,0,3)을 쓰지만
        #    여기서는 (B, 24, T', 1024) 포맷이므로 permute(1, 0, 2, 3)으로 맞춤
        wl = wavlm_layers.permute(1, 0, 2, 3)           # (24, B, T', 1024)
        wavlm = weighted_sum(wl)                          # (B, T', 1024)

        # 2) Conv1d 포맷으로 전치: (B, C, T)
        x_f = fbank.transpose(1, 2)                      # (B, 80, T_fbank)
        x_w = wavlm.transpose(1, 2)                      # (B, 1024, T_wavlm)

        # 3) 각 branch 처리
        #    실제: xl = [각 block 출력] 저장 후 xl[1:] cat
        #    여기서는 중간 feature 저장을 위해 직접 구현
        xl = []
        x_cur = x_f
        for layer in self.fbank_branch:
            x_cur = layer(x_cur)
            if isinstance(layer, nn.ReLU):
                xl.append(x_cur)
        # xl: list of (B, channel, T_fbank)

        xl_f = []
        x_wcur = F.relu(self.wavlm_proj(x_w))
        for layer in self.wavlm_branch:
            x_wcur = layer(x_wcur)
            if isinstance(layer, nn.ReLU):
                xl_f.append(x_wcur)
        # xl_f: list of (B, channel, T_wavlm)

        # 4) Multi-layer concat
        x_cat     = torch.cat(xl[1:], dim=1)             # (B, channel*(n-1), T_fbank)
        x_feat_cat = torch.cat(xl_f[1:], dim=1)          # (B, channel*(n-1), T_wavlm)

        # 5) Temporal alignment: WavLM → Fbank 시간축 맞춤 (resample_tensor)
        T_target = x_cat.shape[2]
        x_feat_resampled = F.interpolate(
            x_feat_cat, size=T_target, mode='linear', align_corners=False
        )                                                  # (B, channel*(n-1), T_fbank)

        # 6) Element-wise multiplication fusion ← 핵심 설계
        x_mul = x_cat * x_feat_resampled                 # (B, channel*(n-1), T_fbank)

        # 7) MFA → ASP → BN → FC → L2 norm
        x_mfa = F.relu(self.mfa(x_mul))                  # (B, mfa_out, T_fbank)
        x_asp = self._asp(x_mfa)                          # (B, 2*mfa_out)
        x_bn  = self.asp_bn(x_asp)
        x_fc  = self.fc(x_bn.unsqueeze(2)).squeeze(2)    # (B, 192)
        return F.normalize(x_fc, dim=1)


class HybridASVModel(nn.Module):
    """
    전체 Hybrid ASV 모델 래퍼 (Dummy 버전).

    실제 VPC 모델로 교체 방법:
      1. self.wavlm  → WavLM.py의 WavLM 인스턴스 + WavLM-Large.pt 로드
      2. self.fbank  → speechbrain.lobes.features.Fbank(n_mels=80)
      3. self.ecapa  → ECAPA_TDNN_test() + embedding_model.ckpt 로드
      4. self.wls    → 체크포인트의 weighted_sum 가중치 로드
    """

    def __init__(self):
        super().__init__()
        self.wavlm = DummyWavLM()
        self.fbank = DummyFbank()
        self.wls   = WeightedLayerSum()
        self.ecapa = DummyECAPATDNN()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T)  16kHz
        Returns:
            embedding: (B, 192)  L2 normalized
        """
        fbank_feats  = self.fbank(waveform)                     # (B, T', 80)
        wavlm_layers = self.wavlm.extract_features(waveform)    # (B, 24, T'', 1024)
        return self.ecapa(fbank_feats, wavlm_layers, self.wls)  # (B, 192)


# ============================================================
# Section 2: Loss Functions
# ============================================================

def loss_adv(emb_adv: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
    """
    Targeted attack loss.
    cos_sim(emb_adv, target_emb)를 최대화 = (1 - cos_sim)을 최소화.
    두 임베딩이 가까워질수록 0에 수렴.
    """
    cos_sim = F.cosine_similarity(emb_adv, target_emb, dim=1)
    return (1.0 - cos_sim).mean()


def loss_stft(
    x:          torch.Tensor,
    x_adv:      torch.Tensor,
    n_fft:      int = 512,
    hop_length: int = 128,
) -> torch.Tensor:
    """
    STFT magnitude L2 loss.
    x와 x_adv의 주파수 특성 차이를 최소화 → 지각적 유사성 보존.
    torch.stft는 differentiable이므로 δ gradient 흐름 보장.
    """
    window = torch.hann_window(n_fft, device=x.device)

    def stft_mag(sig: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            sig, n_fft=n_fft, hop_length=hop_length,
            window=window, return_complex=True
        ).abs()                                            # (B, F, T_stft)

    return F.mse_loss(stft_mag(x_adv), stft_mag(x))


def loss_stoi_approx(
    x:          torch.Tensor,
    x_adv:      torch.Tensor,
    n_fft:      int = 512,
    hop_length: int = 128,
    n_mels:     int = 64,
    sample_rate: int = 16000,
    frame_len:  int = 30,
) -> torch.Tensor:
    """
    Differentiable STOI 근사.

    실제 STOI (Taal et al. 2011):
      1/3-octave 대역 분해 → 30-frame 단위 단기 상관계수 평균
      → 클수록 명료도 높음

    근사 방법:
      mel-spectrogram으로 주파수 대역 분해 (1/3-octave 근사)
      → frame_len 길이 세그먼트 단위로 정규화 교차상관 계산
      → loss = 1 - 평균상관  (0에 가까울수록 원본 명료도 유지)
    """
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels,
    ).to(x.device)

    X   = mel_tf(x).clamp(min=1e-8)       # (B, n_mels, T_frame)
    X_p = mel_tf(x_adv).clamp(min=1e-8)

    # frame_len 단위로 자름 (나머지 버림)
    T = X.shape[-1]
    n_seg = T // frame_len
    if n_seg == 0:
        return torch.tensor(0.0, device=x.device)

    X   = X[..., :n_seg * frame_len].reshape(*X.shape[:-1],   n_seg, frame_len)
    X_p = X_p[..., :n_seg * frame_len].reshape(*X_p.shape[:-1], n_seg, frame_len)
    # shape: (B, n_mels, n_seg, frame_len)

    # 각 (band, segment)를 zero-mean + unit-norm으로 정규화
    X   = X   - X.mean(dim=-1, keepdim=True)
    X_p = X_p - X_p.mean(dim=-1, keepdim=True)
    X   = F.normalize(X,   dim=-1, eps=1e-8)
    X_p = F.normalize(X_p, dim=-1, eps=1e-8)

    # 정규화 교차상관: (B, n_mels, n_seg) → 평균
    corr = (X * X_p).sum(dim=-1)                          # (B, n_mels, n_seg)
    return (1.0 - corr.mean())


# ============================================================
# Section 3: Adversarial Perturbation Optimizer
# ============================================================

class AdversarialPerturbationOptimizer:
    """
    δ를 최적화하여 embed(x + δ) ≈ target_embedding이 되도록 함.

    설정값(config) 예시:
        eps       : L∞ 제약 (0.005 ≈ 16-bit 오디오의 ~164 수준)
        alpha     : L_adv 가중치
        beta      : L_STFT 가중치
        gamma     : L_STOI 가중치
        n_iters   : 최적화 반복 횟수
        lr        : Adam learning rate
        proj_type : 'linf' 또는 'l2'
    """

    def __init__(self, model: nn.Module, config: dict):
        self.model  = model
        self.config = config
        self._freeze_model()

    def _freeze_model(self):
        """모델 전체를 eval + requires_grad=False로 고정."""
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _proj_linf(delta: torch.Tensor, eps: float) -> torch.Tensor:
        return delta.clamp(-eps, eps)

    @staticmethod
    def _proj_l2(delta: torch.Tensor, eps: float) -> torch.Tensor:
        norm = delta.norm(p=2)
        if norm > eps:
            delta = delta * (eps / norm)
        return delta

    def run(
        self,
        x:                torch.Tensor,    # (1, T)  clean utterance
        target_embedding: torch.Tensor,    # (1, 192) frozen target embedding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_adv : (1, T)  adversarial waveform
            delta  : (1, T)  final perturbation
        """
        cfg = self.config
        eps       = cfg.get('eps',       0.005)
        alpha     = cfg.get('alpha',     1.0)
        beta      = cfg.get('beta',      0.1)
        gamma     = cfg.get('gamma',     0.05)
        n_iters   = cfg.get('n_iters',   300)
        lr        = cfg.get('lr',        1e-3)
        proj_type = cfg.get('proj_type', 'linf')
        verbose   = cfg.get('verbose',   True)
        log_every = cfg.get('log_every', 50)

        device = x.device
        x_orig  = x.detach()
        tgt_emb = target_embedding.detach()

        # δ: 유일한 학습 파라미터, 0으로 초기화
        delta = torch.zeros_like(x_orig, requires_grad=True, device=device)
        optimizer = Adam([delta], lr=lr)

        best_loss  = float('inf')
        best_delta = delta.detach().clone()

        for i in range(n_iters):
            optimizer.zero_grad()

            x_adv = (x_orig + delta).clamp(-1.0, 1.0)

            # --- Forward (모델 파라미터는 frozen, δ만 gradient 흐름) ---
            emb_adv = self.model(x_adv)               # (1, 192)

            # --- Hybrid Loss ---
            l_adv  = loss_adv(emb_adv, tgt_emb)
            l_stft = loss_stft(x_orig, x_adv)
            l_stoi = loss_stoi_approx(x_orig, x_adv)
            l_total = alpha * l_adv + beta * l_stft + gamma * l_stoi

            # --- Backward: δ에만 gradient 적용 ---
            l_total.backward()
            optimizer.step()

            # --- L∞ / L2 투영 (지각적 제약 강제) ---
            with torch.no_grad():
                if proj_type == 'linf':
                    delta.data = self._proj_linf(delta.data, eps)
                else:
                    delta.data = self._proj_l2(delta.data, eps)

            # 최적 δ 기록
            if l_total.item() < best_loss:
                best_loss  = l_total.item()
                best_delta = delta.detach().clone()

            if verbose and (i % log_every == 0 or i == n_iters - 1):
                cos = F.cosine_similarity(emb_adv.detach(), tgt_emb, dim=1).item()
                print(
                    f"[{i:>4d}/{n_iters}] "
                    f"total={l_total.item():.4f} | "
                    f"adv={l_adv.item():.4f} | "
                    f"stft={l_stft.item():.4f} | "
                    f"stoi={l_stoi.item():.4f} | "
                    f"cos_sim={cos:.4f}"
                )

        x_adv_final = (x_orig + best_delta).clamp(-1.0, 1.0)
        return x_adv_final.detach(), best_delta


# ============================================================
# Section 4: 실제 VPC 모델 통합 래퍼 (TODO)
# ============================================================

class RealASVModelWrapper(nn.Module):
    """
    실제 VPC ASV 체크포인트 통합용 래퍼.

    ⚠️  CRITICAL: gradient 흐름을 위해 WavLM forward에 no_grad() 금지.
        원본 WavLMFeatureExtractor.extract_features()는 내부에서 torch.no_grad()를
        사용하므로, 아래 forward()처럼 직접 wavlm.extract_features()를 호출해야 함.

    Note: ECAPA_TDNN_test 내부에 self.weighted_sum이 있으므로 wls를 별도로 전달하지 않음.
    """

    def __init__(self, wavlm_model, fbank, normalizer, ecapa):
        super().__init__()
        self.wavlm      = wavlm_model
        self.fbank      = fbank
        self.normalizer = normalizer
        self.ecapa      = ecapa

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T)  16kHz, [-1, 1] range
        Returns:
            embedding: (B, 192)  L2 normalized
        """
        # Fbank 추출 (differentiable)
        wav_lens = torch.ones(waveform.shape[0], device=waveform.device)
        fbank_feats = self.fbank(waveform)
        fbank_feats = self.normalizer(fbank_feats, wav_lens)        # (B, T', 80)

        # WavLM normalize (학습 시 적용된 경우)
        wav_in = waveform
        if self.wavlm.cfg.normalize:
            wav_in = F.layer_norm(waveform, waveform.shape[-1:])

        # WavLM 추출 — no_grad() 없이 호출 (gradient 흐름 보장)
        final_features, _ = self.wavlm.extract_features(
            wav_in,
            output_layer=self.wavlm.cfg.encoder_layers,
            ret_layer_results=True,
        )
        layer_reps = torch.stack(
            [x for x, _ in final_features[1][1:]], dim=0
        )                                                           # (24, B, T'', 1024)
        layer_reps = layer_reps.permute(2, 0, 1, 3)               # (T'', 24, B, 1024)

        # ECAPA_TDNN_test.forward(x, features) — features 포맷: (T'', 24, B, 1024)
        embedding = self.ecapa(fbank_feats, layer_reps)            # (B, 1, 192)
        return F.normalize(embedding.squeeze(1), dim=1)            # (B, 192)


# ============================================================
# Section 5: Entry Point
# ============================================================

def build_model(checkpoint_dir: Optional[str] = None) -> nn.Module:
    """
    체크포인트 없으면 Dummy 모델, 있으면 실제 VPC 모델 반환.
    checkpoint_dir: exp/asv_ssl/ 폴더 경로 (예: 'exp/asv_ssl')
    """
    if checkpoint_dir is None:
        print("⚠  체크포인트 없음 → Dummy 모델로 실행 (gradient 흐름 검증용)")
        return HybridASVModel()

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from pathlib import Path
    from evaluation.privacy.asv.WavLM import WavLM, WavLMConfig
    from evaluation.privacy.asv.ecapa_model_wavlm_24layer import ECAPA_TDNN_test
    from speechbrain.lobes.features import Fbank
    from speechbrain.processing.features import InputNormalization

    ckpt_dir = Path(checkpoint_dir)

    print("WavLM-Large 로드 중...")
    ckpt = torch.load(ckpt_dir / 'WavLM-Large.pt', map_location='cpu')
    cfg  = WavLMConfig(ckpt['cfg'])
    wavlm = WavLM(cfg)
    wavlm.load_state_dict(ckpt['model'])

    print("ECAPA_TDNN_test 로드 중...")
    ecapa = ECAPA_TDNN_test()
    ckpt2 = torch.load(ckpt_dir / 'embedding_model.ckpt', map_location='cpu')
    state = {k[len('module.'):] if k.startswith('module.') else k: v
             for k, v in ckpt2.get('state_dict', ckpt2).items()}
    ecapa.load_state_dict(state, strict=True)

    fbank      = Fbank(n_mels=80, left_frames=0, right_frames=0, deltas=False)
    normalizer = InputNormalization(norm_type='sentence', std_norm=False)

    print("실제 VPC ASV 모델 로드 완료.")
    return RealASVModelWrapper(wavlm, fbank, normalizer, ecapa)


def process_utterances(
    input_paths: list,
    output_dir: str,
    checkpoint_dir: str = 'exp/asv_ssl',
    config: Optional[dict] = None,
):
    """
    발화 목록을 받아 각 발화마다 독립적인 랜덤 target embedding으로 공격 수행.

    Args:
        input_paths  : 입력 wav 파일 경로 리스트
        output_dir   : 결과 wav 저장 디렉토리
        checkpoint_dir: VPC 체크포인트 폴더 (None이면 Dummy 모델)
        config       : 공격 하이퍼파라미터 (None이면 기본값 사용)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if config is None:
        config = {
            'eps':       0.005,
            'alpha':     1.0,
            'beta':      0.1,
            'gamma':     0.05,
            'n_iters':   300,
            'lr':        1e-3,
            'proj_type': 'linf',
            'verbose':   True,
            'log_every': 50,
        }

    model   = build_model(checkpoint_dir).to(device)
    attacker = AdversarialPerturbationOptimizer(model, config)

    for wav_path in input_paths:
        print(f"\n{'='*60}")
        print(f"처리 중: {wav_path}")

        x, sr = torchaudio.load(wav_path)
        x = x.to(device)                                     # (1, T)

        # 발화마다 독립적인 랜덤 target — 같은 화자라도 연결 불가
        with torch.no_grad():
            target_embedding = F.normalize(
                torch.randn(1, 192, device=device), dim=1
            )

        x_adv, delta = attacker.run(x, target_embedding)

        out_name = os.path.splitext(os.path.basename(wav_path))[0] + '_protected.wav'
        out_path = os.path.join(output_dir, out_name)
        torchaudio.save(out_path, x_adv.cpu(), sr)

        with torch.no_grad():
            emb_clean = model(x.clamp(-1, 1))
            emb_adv   = model(x_adv)

        print(f"저장: {out_path}")
        print(f"δ  L∞  = {delta.abs().max().item():.6f}  (제약: {config['eps']})")
        print(f"δ  L2  = {delta.norm(p=2).item():.4f}")
        print(f"δ  SNR = {10 * torch.log10(x.norm()**2 / delta.norm()**2).item():.2f} dB")
        print(f"cos_sim(clean, target) = {F.cosine_similarity(emb_clean, target_embedding).item():.4f}")
        print(f"cos_sim(adv,   target) = {F.cosine_similarity(emb_adv,   target_embedding).item():.4f}")
        print(f"cos_sim(clean, adv   ) = {F.cosine_similarity(emb_clean, emb_adv).item():.4f}")


def main():
    process_utterances(
        input_paths=[
            "/home/cnsl_intern/Workspace/VPC_sangjun/hello.wav",
        ],
        output_dir="protected_output",
        checkpoint_dir='exp/asv_ssl',
    )


if __name__ == '__main__':
    main()
