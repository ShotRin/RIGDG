import os
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from Encoders.encoders import ConvTransposeGenerator, nonBAGAN_Discriminator, nonBAGAN_Reconstructor, PCB_Dataset, Featuremap_Reconstructor, MapInputGenerator
from Methods.ProposedMethod0421 import ProposedGAN, CBAM2D

# 설정
channels = 3
defect_class_idx = 1

lambda_recon = 0.1

# 생성 이미지 저장 경로
save_dir = f'./results/generated_defect_only_{lambda_recon}'

# 기존에 학습 시킨 모델의 가중치 경로
g_path = f"./res_proposed_{lambda_recon}/MapInputGenerator_nonBAGAN_Discriminator/Weights/proposed_epoch_100_generator.pth"
r_path = f"./res_proposed_{lambda_recon}/MapInputGenerator_nonBAGAN_Discriminator/Weights/proposed_epoch_100_reconstructor.pth"
os.makedirs(save_dir, exist_ok=True)

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 선언 및 로드

# 모델 정의 및 로드
generator = MapInputGenerator(in_channels=512, out_channels=channels).to(device)
reconstructor = Featuremap_Reconstructor(channels=channels).to(device)
cbam = CBAM2D(512).to(device)

generator.load_state_dict(torch.load(g_path, map_location=device))
reconstructor.load_state_dict(torch.load(r_path, map_location=device))

generator.eval()
reconstructor.eval()
cbam.eval()

# 데이터셋 경로
data_path = ''

# 데이터셋 로드
dataset = PCB_Dataset(data_path=data_path, export_label=["False Alarm", "Pollution"], Train=False)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# 이미지 생성 루프
count = 0
with torch.no_grad():
    for ins_img, ref_img, _, label in dataloader:
        if label.item() != defect_class_idx:
            continue

        ins_img = ins_img.to(device)
        ref_img = ref_img.to(device)

        z_ins = reconstructor(ins_img)
        z_ref = reconstructor(ref_img)
        delta_z = z_ins - z_ref
        z_cond = z_ref + cbam(delta_z)

        generated = generator(z_cond)

        # 저장 경로
        gen_path = os.path.join(save_dir, f"gen_defect_{count:03d}.png")
        # origin_path = os.path.join(save_dir, f"gen_defect_{count:03d}_origin.png")

        # 저장
        save_image(generated, gen_path, normalize=True)
        # save_image(ins_img, origin_path, normalize=True)  # 💡 원본 이미지도 저장

        count += 1
        if count >= 200:
            break

print(f"[✓] 총 {count}장의 결함 이미지가가 '{save_dir}'에 저장되었습니다.")