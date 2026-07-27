import argparse, os, torch
import numpy as np

from torchvision import transforms
from torch.utils.data import DataLoader

# 변환된 BalancingGAN을 ResNet encoder 기반으로 구성 
from Encoders.encoders import ConvTransposeGenerator, nonBAGAN_Discriminator, nonBAGAN_Reconstructor, PCB_Dataset, Featuremap_Reconstructor, MapInputGenerator
from Methods.ProposedMethod0421 import ProposedGAN

from utils import set_random_seed

# python3 train_3.py
# torch.cuda.is_available() # True

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"  # Set the GPU 0 to use

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


set_random_seed(2023)

def init_models(device, args):
    generator     = MapInputGenerator().to(device)
    discriminator = nonBAGAN_Discriminator().to(device)
    reconstructor = Featuremap_Reconstructor().to(device)
    opt_R         = torch.optim.Adam(reconstructor.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    opt_G         = torch.optim.Adam(generator.parameters(),     lr=args.learning_rate, betas=(0.5, 0.999))   
    opt_D         = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    return [[generator, discriminator, reconstructor], [opt_G, opt_D, opt_R]]

def run_proposed(device, dataloader, encoders ,opts, args):
    proposed = ProposedGAN(encoders, opts, args.n_classes, device, res_dir="./res_proposed")
    proposed.train(dataloader, epochs=args.epochs)
    proposed.save_history()
    
DATA_DIR_MASTER     = ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   type=str,   default="")
                        # help='Path to the dataset directory')
    parser.add_argument('--latent_dim', type=int, default=100,help='Dimensionality of the latent vector z')
    parser.add_argument('--img_size', type=int, default=300,
                        help='Size of the input/output images (assumed square)')
    parser.add_argument('--channels', type=int, default=3, help='Number of image channels (e.g., 3 for RGB)')
    parser.add_argument('--n_classes', type=int, default=2, help='Number of data classes')
    # parser.add_argument('--n_classes', type=int, default=3, help='Number of data classes')
    parser.add_argument('--batch_size', type=int, default=40, help='Batch size for training')
    parser.add_argument('--lr', '--learning_rate', dest='learning_rate', type=float, default=0.00005,
                        help='Learning rate for optimizers')
    parser.add_argument('--epochs', type=int, default=100, help='Total number of GAN training epochs')
    parser.add_argument('--seed', type=int, default=2023, help='Random seed')
    
    args = parser.parse_args()

    
    encoders, opts = init_models(device, args)

    dataset     = PCB_Dataset(DATA_DIR_MASTER, ["False Alarm","Pollution"],)
    # dataset     = PCB_Dataset(DATA_DIR_MASTER, ["False Alarm","Pollution", "Foreign Particles"],)
    dataloader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)



    run_proposed(device, dataloader, encoders ,opts, args)

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    print("💡 Running with anomaly detection on")
    main()