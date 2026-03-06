import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision import models
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from dataset import PCB_Dataset
from utils import set_random_seed


# ==========================
#  Config: 클래스 이름 정의
# ==========================
# 필요한 경우 여기만 바꿔서 사용
CLASS_NAMES = ["False Alarm", "Pollution"]  # 예: 0=False Alarm(정상), 1=Pollution(결함)


# ==============================================
#  ResNet Encoder (멀티 레이어 feature 반환)
# ==============================================
class ResNetEncoderMulti(nn.Module):
    """
    ResNet18에서 layer1~layer4 feature map을 모두 반환하는 Encoder
    return: dict { "l1": f1, "l2": f2, "l3": f3, "l4": f4 }
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # (B, 64, H1, W1)
        self.layer2 = backbone.layer2  # (B,128, H2, W2)
        self.layer3 = backbone.layer3  # (B,256, H3, W3)
        self.layer4 = backbone.layer4  # (B,512, H4, W4)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return {"l1": f1, "l2": f2, "l3": f3, "l4": f4}


# ==================================
#  Stage 1: 정상 쌍으로 Encoder 학습
#         (멀티 레이어 L1 loss)
# ==================================
def train_encoder_on_normal_pairs_multi(
    encoder: nn.Module,
    dataset: PCB_Dataset,
    device: torch.device,
    normal_class_name: str = "False Alarm",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    w1: float = 0.25,
    w2: float = 0.25,
    w3: float = 0.25,
    w4: float = 0.25,
):
    """
    정상(Current, Master) 쌍에 대해
    L = Σ_k w_k * L1( F^k_cur, F^k_mas ) (k ∈ {l1,l2,l3,l4}) 로 Encoder 학습
    """
    encoder = encoder.to(device)
    encoder.train()

    # 1) 정상 인덱스만 추출
    normal_label_idx = dataset.class_to_idx[normal_class_name]
    normal_indices = [
        i for i, (_, _, label) in enumerate(dataset.samples)
        if label == normal_label_idx
    ]
    print(f"[Stage1] Normal samples for training: {len(normal_indices)}")

    normal_loader = DataLoader(
        Subset(dataset, normal_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    criterion = nn.L1Loss()
    weights = {"l1": w1, "l2": w2, "l3": w3, "l4": w4}

    for epoch in range(1, epochs + 1):
        encoder.train()
        running_loss = 0.0
        n_samples = 0
        t0 = time.time()

        for current_img, master_img, paths, labels in normal_loader:
            current_img = current_img.to(device)
            master_img  = master_img.to(device)

            optimizer.zero_grad()

            f_cur = encoder(current_img)  # dict
            f_mas = encoder(master_img)   # dict

            loss = 0.0
            for k, w in weights.items():
                loss += w * criterion(f_cur[k], f_mas[k])

            loss.backward()
            optimizer.step()

            bs = current_img.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

        epoch_loss = running_loss / max(1, n_samples)
        dt = time.time() - t0
        print(f"[Stage1] Epoch {epoch}/{epochs}  Loss: {epoch_loss:.6f}  (time: {dt:.1f}s)")

    return encoder


# ===========================================
#  ΔF scalar 분포 계산 (정상 vs 결함 비교용)
#  - 여기서는 layer4 feature만 사용
# ===========================================
@torch.no_grad()
def compute_delta_scalar_stats_from_pcb(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device
):
    """
    PCB_Dataset batch (current, master, path, label)를 받아
    각 쌍에 대해 mean |ΔF| 를 하나의 스칼라로 계산.
    ΔF는 layer4(feature["l4"]) 기준으로 계산.
    """
    encoder.eval()
    encoder.to(device)

    all_vals = []

    for batch in dataloader:
        current_img, master_img, paths, labels = batch
        current_img = current_img.to(device)
        master_img  = master_img.to(device)

        f_cur = encoder(current_img)["l4"]  # (B,C,H4,W4)
        f_mas = encoder(master_img)["l4"]

        delta = torch.abs(f_cur - f_mas)      # (B,C,H4,W4)
        delta_scalar = delta.mean(dim=(1, 2, 3))    # (B,)
        all_vals.append(delta_scalar.cpu())

    if len(all_vals) == 0:
        return torch.empty(0)

    return torch.cat(all_vals, dim=0)


def plot_histograms(normal_vals, defect_vals, out_path: str):
    normal_vals = np.array(normal_vals)
    defect_vals = np.array(defect_vals)

    plt.figure()
    plt.hist(normal_vals, bins=30, alpha=0.5, label="Normal pairs")
    plt.hist(defect_vals, bins=30, alpha=0.5, label="Defect pairs")
    plt.xlabel("mean |ΔF| (layer4)")
    plt.ylabel("count")
    plt.legend()
    plt.title("Distribution of mean |ΔF| (normal vs defect, layer4)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Stage2] Saved histogram: {out_path}")


# ====================================
#  ΔF 기반 attention map / heatmap
#  - (1) layer4만 사용
#  - (2) l1~l4 모두 사용 (multi)
# ====================================

@torch.no_grad()
def compute_attention_map_layer4(
    encoder: nn.Module,
    insp: torch.Tensor,
    ref: torch.Tensor
) -> torch.Tensor:
    """
    layer4 feature만으로 ΔF 기반 attention map 생성.
    return: A_up (B,1,H,W)  - 입력 이미지 크기로 upsample된 heatmap
    """
    encoder.eval()
    feat_insp_l4 = encoder(insp)["l4"]
    feat_ref_l4  = encoder(ref)["l4"]

    delta = torch.abs(feat_insp_l4 - feat_ref_l4)  # (B,C,H4,W4)
    A = delta.mean(dim=1, keepdim=True)           # (B,1,H4,W4)

    B, _, H, W = insp.shape
    # 샘플별 0~1 normalize 후 upsample
    outs = []
    for b in range(B):
        att = A[b:b+1]
        mn, mx = att.min(), att.max()
        if (mx - mn) > 1e-6:
            att = (att - mn) / (mx - mn)
        else:
            att = torch.zeros_like(att)
        att_up = F.interpolate(att, size=(H, W), mode="bilinear", align_corners=False)
        outs.append(att_up)
    return torch.cat(outs, dim=0)  # (B,1,H,W)


@torch.no_grad()
def compute_attention_map_multi(
    encoder: nn.Module,
    insp: torch.Tensor,
    ref: torch.Tensor
) -> torch.Tensor:
    """
    l1~l4 모든 레이어에서 ΔF를 계산하고,
    각 레이어별 ΔF->채널 평균->0~1 normalize->upsample 후 합산한 multi-scale heatmap.
    return: A_final (B,1,H,W)  - 입력 이미지 크기로 upsample된 heatmap
    """
    encoder.eval()
    feat_insp = encoder(insp)  # dict
    feat_ref  = encoder(ref)

    B, _, H, W = insp.shape
    att_sum = 0.0
    n_layers = 0

    for k in ["l1", "l2", "l3", "l4"]:
        f_i = feat_insp[k]
        f_r = feat_ref[k]
        delta = torch.abs(f_i - f_r)          # (B,C,hk,wk)
        A_k = delta.mean(dim=1, keepdim=True) # (B,1,hk,wk)

        # 샘플별 0~1 normalize
        As = []
        for b in range(B):
            a = A_k[b:b+1]
            mn, mx = a.min(), a.max()
            if (mx - mn) > 1e-6:
                a = (a - mn) / (mx - mn)
            else:
                a = torch.zeros_like(a)
            As.append(a)
        A_k = torch.cat(As, dim=0)  # (B,1,hk,wk)

        # 원본 이미지 크기로 upsample
        A_k_up = F.interpolate(A_k, size=(H, W),
                               mode="bilinear", align_corners=False)
        att_sum = att_sum + A_k_up
        n_layers += 1

    A_final = att_sum / max(1, n_layers)

    # 한 번 더 샘플별 0~1 normalize
    outs = []
    for b in range(B):
        a = A_final[b:b+1]
        mn, mx = a.min(), a.max()
        if (mx - mn) > 1e-6:
            a = (a - mn) / (mx - mn)
        else:
            a = torch.zeros_like(a)
        outs.append(a)
    return torch.cat(outs, dim=0)  # (B,1,H,W)


@torch.no_grad()
def save_defect_heatmaps_from_pcb(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    out_dir_layer4: str,
    out_dir_multi: str,
    max_batches: int = 5
):
    """
    결함쌍 몇 배치에 대해:
    - layer4 기반 heatmap → out_dir_layer4
    - multi-layer 기반 heatmap → out_dir_multi
    각각 Current 이미지 위에 overlay해서 저장.
    """
    os.makedirs(out_dir_layer4, exist_ok=True)
    os.makedirs(out_dir_multi, exist_ok=True)

    encoder.to(device)
    encoder.eval()

    count_batches = 0

    # Unnormalize용 (PCB_Dataset의 Normalize 역연산)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    for batch in dataloader:
        current_img, master_img, paths, labels = batch
        current_img = current_img.to(device)
        master_img  = master_img.to(device)

        B, _, H, W = current_img.shape

        # (1) layer4 heatmap
        A_l4 = compute_attention_map_layer4(encoder, current_img, master_img)   # (B,1,H,W)
        # (2) multi-layer heatmap
        A_multi = compute_attention_map_multi(encoder, current_img, master_img) # (B,1,H,W)

        for i in range(B):
            # (1) 원본 이미지 복원
            base = current_img[i:i+1].detach().cpu()
            base_vis = base * std + mean
            base_vis = torch.clamp(base_vis, 0.0, 1.0)

            # (2-1) layer4 heatmap overlay
            att_l4 = A_l4[i:i+1].detach().cpu()
            overlay_l4 = base_vis.clone()
            overlay_l4[0, 0:1, :, :] = torch.clamp(
                overlay_l4[0, 0:1, :, :] + att_l4[0] * 0.7,
                0.0, 1.0
            )

            # (2-2) multi-layer heatmap overlay
            att_multi = A_multi[i:i+1].detach().cpu()
            overlay_multi = base_vis.clone()
            overlay_multi[0, 0:1, :, :] = torch.clamp(
                overlay_multi[0, 0:1, :, :] + att_multi[0] * 0.7,
                0.0, 1.0
            )

            fname = os.path.basename(paths[i])
            save_path_l4 = os.path.join(out_dir_layer4, f"l4_{fname}")
            save_path_multi = os.path.join(out_dir_multi, f"multi_{fname}")

            save_image(overlay_l4,    save_path_l4)
            save_image(overlay_multi, save_path_multi)

            print(f"[Stage2] Saved heatmap (l4):    {save_path_l4}")
            print(f"[Stage2] Saved heatmap (multi): {save_path_multi}")

        count_batches += 1
        if count_batches >= max_batches:
            break


# ===========================
#  Stage 2 전체 파이프라인
# ===========================
def quick_delta_check(
    device: torch.device,
    dataset: PCB_Dataset,
    encoder: nn.Module,
    normal_class_name: str = "False Alarm",
    batch_size: int = 32,
    out_dir_base: str = "./pcb_defect_heatmaps"
):
    """
    학습이 끝난 encoder를 그대로 이용해서:
    1) Normal vs Defect ΔF 분포 비교 (layer4 기준)
    2) 결함 heatmap 저장
       - out_dir_base_layer4 : layer4 heatmap
       - out_dir_base_multi  : multi-layer heatmap
    """
    # Normal / Defect 인덱스 분리
    normal_indices, defect_indices = [], []
    normal_label_idx = dataset.class_to_idx[normal_class_name]

    for idx, (_, _, label) in enumerate(dataset.samples):
        if label == normal_label_idx:
            normal_indices.append(idx)
        else:
            defect_indices.append(idx)

    print(f"[Stage2] Normal samples: {len(normal_indices)}")
    print(f"[Stage2] Defect samples: {len(defect_indices)}")

    normal_loader = DataLoader(
        Subset(dataset, normal_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    defect_loader = DataLoader(
        Subset(dataset, defect_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 1) ΔF scalar 분포 (layer4 기준)
    normal_vals = compute_delta_scalar_stats_from_pcb(encoder, normal_loader, device)
    defect_vals = compute_delta_scalar_stats_from_pcb(encoder, defect_loader, device)

    if len(normal_vals) > 0:
        print(f"[Stage2][ΔF] Normal: N={len(normal_vals)}, mean={normal_vals.mean():.6f}, std={normal_vals.std():.6f}")
    if len(defect_vals) > 0:
        print(f"[Stage2][ΔF] Defect: N={len(defect_vals)}, mean={defect_vals.mean():.6f}, std={defect_vals.std():.6f}")

    plot_histograms(
        normal_vals,
        defect_vals,
        out_path="delta_distribution_pcb_normal_vs_defect_layer4.png"
    )

    # 2) 결함 heatmap 저장 (두 폴더로 분리)
    out_dir_layer4 = out_dir_base + "_layer4"
    out_dir_multi  = out_dir_base + "_multi"

    save_defect_heatmaps_from_pcb(
        encoder=encoder,
        dataloader=defect_loader,
        device=device,
        out_dir_layer4=out_dir_layer4,
        out_dir_multi=out_dir_multi,
        max_batches=5,
    )


# =========
#  main
# =========
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_dir',
        type=str,
        default="/home/labs/lab_shpark/ce18d024/20251114/Data/20241112_p",
        help='Path to the dataset directory (Current/Master 이미지 루트)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=2025,
        help='Random seed'
    )
    parser.add_argument(
        '--normal_class',
        type=str,
        default="False Alarm",
        help='정상으로 간주할 클래스 이름 (CLASS_NAMES와 일치해야 함)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='학습/분석에 사용할 batch size'
    )
    parser.add_argument(
        '--encoder_epochs',
        type=int,
        default=20,
        help='정상쌍으로 Encoder 학습할 epoch 수'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Encoder 학습 learning rate'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default="./pcb_defect_heatmaps",
        help='heatmap base 경로 (layer4/multi 폴더 suffix로 나뉨)'
    )

    args = parser.parse_args()

    # 시드 고정
    set_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset: 학습용(train=True), 분석용(train=False)
    dataset_train = PCB_Dataset(args.data_dir, CLASS_NAMES, train=True)
    dataset_eval  = PCB_Dataset(args.data_dir, CLASS_NAMES, train=False)

    # Stage 1: 정상쌍으로 Encoder 학습 (멀티 레이어 L1 loss)
    encoder = ResNetEncoderMulti(pretrained=True)
    encoder = train_encoder_on_normal_pairs_multi(
        encoder=encoder,
        dataset=dataset_train,
        device=device,
        normal_class_name=args.normal_class,
        epochs=args.encoder_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        # 필요하면 레이어별 weight 조정 가능
        w1=0.25, w2=0.25, w3=0.25, w4=0.25,
    )

    # Encoder 가중치 fix
    for p in encoder.parameters():
        p.requires_grad = False

    # Stage 2: 학습된 encoder로 ΔF 분포 / heatmap 분석
    quick_delta_check(
        device=device,
        dataset=dataset_eval,
        encoder=encoder,
        normal_class_name=args.normal_class,
        batch_size=args.batch_size,
        out_dir_base=args.out_dir,
    )


if __name__ == '__main__':
    main()
