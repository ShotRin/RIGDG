import os
import csv
import re
import numpy as np
from collections import defaultdict
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

import matplotlib.pyplot as plt
import os

from torchvision.utils import save_image

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # [B, C, 1, 1]
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM2D(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM2D, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_att(x) * x
        x = self.spatial_att(x) * x
        return x


# ---------------------------
# BalancingGAN (전체 모델 및 학습 루프 포함)
# ---------------------------
class ProposedGAN:
    def __init__(self, encoders, opts, n_classes, device, res_dir):
        self.generator, self.discriminator, self.reconstructor = encoders # 모델 생성
        self.opt_G, self.opt_D, self.opt_R = opts # 옵티마이저
        
        self.nclasses = n_classes
        self.res_dir = res_dir
        self.device = device
        # self.criterion_adv = nn.CrossEntropyLoss()
        self.criterion_adv = nn.BCEWithLogitsLoss() # 손실함수 # Discriminator 및 Generator adversarial loss
        self.criterion_recon = nn.MSELoss() # 손실함수 # Discriminator 및 Generator adversarial loss
        self.train_history= defaultdict(list) # Loss History 저장
        self.cbam = CBAM2D(512).to(self.device)
    
        self.f_name= os.path.join(self.res_dir,f"{self.generator.__class__.__name__}_{self.discriminator.__class__.__name__}")

    def generate_samples(self, c, samples): # 이미지 생성함수 # return fake image
        c_array = np.full(samples, c)       # train_one_epoch 함수 소속
        latent = self.generate_latent(c_array) # 각 클래스에 대해 학습된 latent 분포에서 샘플링
        # 이미지 생성
        self.generator.eval()
        with torch.no_grad():
            fake = self.generator(latent)
        return fake.cpu().numpy()
    
    
    def save_latent_histogram(self, z_cond, epoch):
        z_np = z_cond.detach().cpu().numpy().flatten()
        print("Saving histogram of z_cond")
        plt.figure(figsize=(6, 4))
        plt.hist(z_np, bins=100)
        plt.title(f"z_cond Distribution @ Epoch {epoch}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        histograms_fn = os.path.join(self.f_name, "histograms")
        os.makedirs(histograms_fn, exist_ok=True)
        plt.savefig(f"{histograms_fn}/z_cond_epoch_{epoch}.png")
        plt.close()  # plt 시작을 했으면 반드시 close 하기기
        
        
    def train_one_epoch(self, bg_train, epoch=None, save_dir=None, save_interval=10): # return np.mean(disc_losses), np.mean(gen_losses)
        # print(f"[epoch {epoch}] train_one_epoch 시작") 

        self.generator.train()
        self.discriminator.train()
        self.reconstructor.train()
        disc_losses, gen_losses, mse_losses = [], [], []
        
        for i, (ins_images, ref_images, _,  labels) in enumerate(bg_train):
            batch_size = ins_images.size(0)
            ins_images = ins_images.to(self.device)
            ref_images = ref_images.to(self.device)
            labels = labels.to(self.device)

            
            z_ins = self.reconstructor(ins_images)     # [B, C, H, W]
            z_ref = self.reconstructor(ref_images)     # [B, C, H, W]
            delta_z = z_ins - z_ref               # [B, C, H, W]
            # STEP 1. CBAM attention을 통해 결함 강조된 feature 생성
            attn_map = self.cbam(delta_z)              # 강조된 결함 위치
            z_cond = z_ref + attn_map             # 조건부 feature map
            

            # STEP 2. 생성된 이미지와 검사 이미지의 차이 구하기
            generated_image = self.generator(z_cond)  # Generator(z_cond) → 생성 이미지
            # print(f"[epoch {epoch}] ▶ 배치 {i} → 이미지 생성 완료")
            loss_mse = self.criterion_recon(ins_images, generated_image) # mse loss

            # STEP 3. Discriminator 학습
            # 클래스 분류 결과
            # fake_target = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            # loss_real = self.criterion_adv(self.discriminator(ins_images), labels)
            # loss_fake = self.criterion_adv(self.discriminator(generated_image.detach()), fake_target)
            

            # Discriminator 학습
            # real, fake 이미지에 대한 손실 계산
            fake_target = torch.zeros(batch_size, 1, device=self.device)  # 가짜 이미지 레이블
            real_target = torch.ones(batch_size, 1, device=self.device)  # 진짜 이미지 레이블
            loss_real = self.criterion_adv(self.discriminator(ins_images), real_target)
            loss_fake = self.criterion_adv(self.discriminator(generated_image.detach()), fake_target)
            # print(f"[epoch {epoch}] 배치 {i} → Discriminator 손실 계산 완료")  

            loss_D = (loss_real + loss_fake) / 2
            self.opt_D.zero_grad()
            loss_D.backward()
            self.opt_D.step()
            disc_losses.append(loss_D.item())
            # -------------------------------------------------------------------------

            # STEP 4. Generator 학습 
            out = self.discriminator(generated_image) # Generator는 Discriminator를 속여 실제 클래스(샘플링된 레이블)를 맞추도록 학습합니다.
            target = torch.ones(batch_size, 1, device=self.device)
            loss_G = self.criterion_adv(
                out, target)
            lambda_recon = 1.0
            loss_mse = lambda_recon* loss_mse
            loss_G_total = loss_G + loss_mse

            self.opt_G.zero_grad()
            self.opt_R.zero_grad()
            loss_G_total.backward()      
            self.opt_G.step()
            self.opt_R.step()
            # print(f"[epoch {epoch}] 배치 {i} → Generator 학습 완료")
            gen_losses.append(loss_G.item())
            mse_losses.append(loss_mse.item())
            # -------------------------------------------------------------------------

            
            # 이미지 저장 조건
            if epoch is not None and save_dir is not None and i == 0 and (
                (epoch + 1) % save_interval == 0 or (epoch + 1) in [1, 2]):
                with torch.no_grad():
                    img = generated_image[0]
                    print("Mean pixel value:", img.mean().item())
                    print("Min:", img.min().item(), "Max:", img.max().item())
                    print(f"z_cond mean: {z_cond.mean().item():.4f}, std: {z_cond.std().item():.4f}")

                os.makedirs(save_dir, exist_ok=True)
                self.save_latent_histogram(z_cond, epoch)

                for i in range(min(5, generated_image.size(0))):
                    save_path = os.path.join(save_dir, f"epoch_{epoch+1}_sample_{i}.png")
                    save_image(generated_image[i], save_path, normalize=True)
        return np.mean(disc_losses), np.mean(gen_losses), np.mean(mse_losses) # 평균 손실 반환


    def train(self, bg_train, epochs=50):
        for epoch in range(epochs):
            print(f"GAN train epoch: {epoch+1}/{epochs}")
            train_start = time.time()
            img_folder_path = f"{self.f_name}/results"
            d_loss, g_loss, m_loss = self.train_one_epoch(
                    bg_train,
                    epoch=epoch,
                    save_dir=img_folder_path,
                    save_interval=10
                )
            train_end = time.time()
            self.train_history['disc_loss'].append(d_loss)
            self.train_history['gen_loss'].append(g_loss)
            print(f"D loss: {d_loss:.5f}, G loss: {g_loss+m_loss:.5f} ({g_loss:.5f}+{m_loss:.5f}), Time: {train_end-train_start:.2f}")

            
            os.makedirs(img_folder_path,exist_ok=True)

            # 10 epoch마다 샘플 이미지 저장
            if (epoch + 1) % 10 == 0 or epoch + 1 == 1 or epoch + 1 == 2:
                self.save_models(self.generator, self.discriminator, self.reconstructor, epoch+1)

    def save_history(self):
        filename = os.path.join(self.f_name, f"score.csv")
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['train_gen_loss', 'train_disc_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for g, d in zip(self.train_history['gen_loss'], self.train_history['disc_loss']):
                writer.writerow({
                    'train_gen_loss' : g,
                    'train_disc_loss': d
                })
        
    def save_models(self, generator, discriminator, reconstructor, epoch):
        Weight_Fn = os.path.join(self.f_name, "Weights")
        os.makedirs(Weight_Fn, exist_ok=True)
        model= 'proposed'
        g_n = f"{model}_epoch_{epoch}_generator.pth"
        d_n = f"{model}_epoch_{epoch}_discriminator.pth"
        r_n = f"{model}_epoch_{epoch}_reconstructor.pth"
        # 모델 저장
        torch.save(    generator.state_dict(), os.path.join(Weight_Fn, g_n))
        torch.save(discriminator.state_dict(), os.path.join(Weight_Fn, d_n))
        torch.save(reconstructor.state_dict(), os.path.join(Weight_Fn, r_n))

    def load_models(self, fname_generator, fname_discriminator, fname_reconstructor, bg_train=None):
        self.generator.load_state_dict(torch.load(fname_generator, map_location=self.device, weights_only=True))
        self.discriminator.load_state_dict(torch.load(fname_discriminator, map_location=self.device, weights_only=True))
        self.reconstructor.load_state_dict(torch.load(fname_reconstructor, map_location=self.device, weights_only=True))
        if bg_train is not None:
            self.init_autoenc(bg_train)
