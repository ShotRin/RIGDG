import torch.nn as nn
import torch, glob
import PIL
from torchvision import transforms

#region Architecture

# ---------------------------
# Generator
# ---------------------------

class PCB_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path:str, export_label, Train=True):
        super().__init__()
        
        self.data_path = data_path 
        self.class_path = []
        for el in export_label:
            self.class_path.append(data_path+'/'+el)

        self.data = []
        
        self.Train=Train
        
        class_name_set = []
        
        for i_class_path in self.class_path:
            class_name = i_class_path.split("/")[-1]
            class_name_set.append(class_name)
            for img_path in glob.iglob(i_class_path + "/*.jpg"):
                if "Current" in img_path or "sample" in img_path:
                    self.data.append([img_path, class_name])
        
        class_name_set.sort()## sort class name by alphabet order
        
        self.class_to_idx = dict(zip(class_name_set, range(len(class_name_set))))            
                    
        self.Train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
                
        self.Valid_Test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.Transform = self.Train_transform if Train else self.Valid_Test_transform

                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx:int):
        
        Current_img_path, class_name = self.data[idx]
        Master_img_path = f"{Current_img_path[:-11]}Master.jpg"
        
        with PIL.Image.open(Current_img_path) as current_img:
            Current_img = self.Transform(current_img)
    
        with PIL.Image.open(Master_img_path) as master_img:
            Master_img = self.Transform(master_img)
        
        label = self.class_to_idx[class_name]
                        
        return Current_img, Master_img, Current_img_path,  label




class MapInputGenerator(nn.Module):
    def __init__(self, in_channels=512, out_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1),  # 10 → 20
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 20 → 40
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 40 → 80
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 80 → 160
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 160 → 320
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1),   # 320 유지
            nn.Upsample(size=(300, 300), mode='bilinear', align_corners=False),
            nn.Tanh()
        )

    def forward(self, z_map):
        return self.net(z_map)
    
# ---------------------------
# Discriminator
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        stride = 2 if downsample else 1

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),  # 🔸 dropout 추가
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.skip_connection = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.final_act = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.conv_block(x)
        out = out + identity
        return self.final_act(out)


class nonBAGAN_Discriminator(nn.Module):
    def __init__(self, channels=3, dropout_rate=0.5):
        super(nonBAGAN_Discriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        self.layer1 = ResidualBlock(64, 128, downsample=True, dropout_rate=dropout_rate)
        self.layer2 = ResidualBlock(128, 256, downsample=True, dropout_rate=dropout_rate)
        self.layer3 = ResidualBlock(256, 256, downsample=True, dropout_rate=dropout_rate)
        self.layer4 = ResidualBlock(256, 512, downsample=True, dropout_rate=dropout_rate)

        self.flatten = nn.Flatten()
        
        self.fc = nn.Sequential(
            nn.Linear(512 * 10 * 10, 1024),        # 🔸 차원 축소
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1)          # 🔸 클래스 + 1 (real/fake)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class nonBAGAN_Reconstructor(nn.Module):
    def __init__(self, channels=3, latent_dim=100):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
            # nn.LeakyReLU(0.2, inplace=False)
        )

        self.layer1 = ResidualBlock(64, 128, downsample=True)
        self.layer2 = ResidualBlock(128, 256, downsample=True)
        self.layer3 = ResidualBlock(256, 256, downsample=True)
        self.layer4 = ResidualBlock(256, 512, downsample=True)


        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 10 * 10, latent_dim)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.flatten(x)
        latent = self.fc(x)
        return latent
    

class Featuremap_Reconstructor(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
            # nn.LeakyReLU(0.2, inplace=False)
        )

        self.layer1 = ResidualBlock(64, 128, downsample=True)
        self.layer2 = ResidualBlock(128, 256, downsample=True)
        self.layer3 = ResidualBlock(256, 256, downsample=True)
        self.layer4 = ResidualBlock(256, 512, downsample=True)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x