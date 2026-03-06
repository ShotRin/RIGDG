from pathlib import Path

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import random
import torchvision.transforms.functional as TF


class PCB_Dataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 export_label: list[str], 
                 train: bool = True):
        
        super().__init__()

        self.root = Path(data_path)
        self.train = train

        # 클래스 이름: export_label 그대로 사용, 알파벳 순 정렬
        self.class_names = sorted(export_label)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        print(f"클래스 목록 : \n{self.class_to_idx}")

        # (Current 경로, Master 경로, label) 리스트 만들기
        self.samples = []
        # 루트 아래 모든 Current 파일을 한 번만 검색
        for current_path in self.root.rglob("*Current*.jpg"):
            class_name = current_path.parent.name

            master_name = current_path.name.replace("Current", "Master")
            master_path = current_path.with_name(master_name)
            if not master_path.exists():
                raise FileNotFoundError(f"참조 이미지 없음: {master_path}")

            label = self.class_to_idx[class_name]
            self.samples.append((current_path, master_path, label))

        # ----- 공통 transform: 300 -> 304 패딩 + ToTensor + Normalize -----
        # 패딩: 좌/우/상/하 2픽셀씩 → 300 + 2*2 = 304
        # pad_to_304 = transforms.Pad(
        #     padding=2,
        #     padding_mode="reflect"
        # )
        # Transform 정의
        self.base_transform = transforms.Compose([
            # pad_to_304,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        current_path, master_path, label = self.samples[idx]
        
        # 이미지 로드
        current_img = Image.open(current_path).convert("RGB")
        master_img  = Image.open(master_path ).convert("RGB")
        
        # ----- train일 때만, 검사/참조에 "같은" 랜덤 플립 적용 -----
        if self.train:
            if random.random() < 0.5:
                current_img = TF.hflip(current_img)
                master_img = TF.hflip(master_img)
                
        # ----- 공통 transform (Pad→Tensor→Normalize) -----
        current_img = self.base_transform(current_img)   # (3, 304, 304)
        master_img = self.base_transform(master_img)     # (3, 304, 304)

        return current_img, master_img, str(current_path), label