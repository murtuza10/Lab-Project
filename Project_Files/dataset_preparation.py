import os
import json
import random
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


def _resize_pil_image(img, long_edge_size):
    """Resize an image while preserving aspect ratio based on long edge size."""
    S = max(img.size)
    interp = Image.LANCZOS if S > long_edge_size else Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


class ImagePairDataset(Dataset):
    def __init__(self, image_paths, np_memmap_file1, np_memmap_file2, train_idx, test_idx, dtype=np.float32, shape=None, image_size=224, train_mode=True):
        self.image_paths = sorted(image_paths)
        self.image_size = image_size
        self.train_mode = train_mode
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.numpy_data1 = np.memmap(np_memmap_file1, dtype=dtype, mode='r', shape=shape)
        self.numpy_data2 = np.memmap(np_memmap_file2, dtype=dtype, mode='r', shape=shape)

    def __len__(self):
        return len(self.image_paths) - 1  

    def normalize(self, image):
        return ((image - 20000) / (25000 - 20000) * 255).astype(np.uint8)

    def apply_colormap(self, image):
        img_np = np.array(image)
        img_color = cv2.applyColorMap(img_np, cv2.COLORMAP_INFERNO)
        return Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))

    def clahe_equalization(self, image):
        img_np = np.array(image)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

    def process_image(self, img_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File {img_path} does not exist!")

        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img = self.normalize(img)
        img = Image.fromarray(img)
        img = exif_transpose(img)
        img = np.asarray(img)[:, 280:1730]
        img = Image.fromarray(img)

        W1, H1 = img.size
        img = _resize_pil_image(img, round(self.image_size * max(W1 / H1, H1 / W1)))

        W, H = img.size
        cx, cy = W // 2, H // 2
        half = min(cx, cy) if self.image_size == 224 else ((2 * cx) // 16) * 8
        img = img.crop((cx - half, cy - half, cx + half, cy + half))

        ImgNorm = transforms.Compose([
            transforms.Lambda(self.apply_colormap),
            transforms.Lambda(self.clahe_equalization),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return ImgNorm(img)

    def __getitem__(self, _):
        L_idx = random.randint(0, len(self.image_paths) - 3)
        idx = self.train_idx[L_idx] if self.train_mode else self.test_idx[L_idx]
        img1_path = self.image_paths[L_idx]
        img2_path = self.image_paths[L_idx + 1]

        img1 = self.process_image(img1_path)
        img2 = self.process_image(img2_path)
        np_val1 = torch.tensor(self.numpy_data1[idx], dtype=torch.float32)
        np_val2 = torch.tensor(self.numpy_data2[idx], dtype=torch.float32)

        return {"view1": img1, "view2": img2, "absolute_dept": np_val1, "relative_dept": np_val2}


def visualize(data, save_path=None):
    std, mean = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for i in range(4):
        img1 = std * (data['view1'][i].permute(1, 2, 0).numpy()) + mean
        img2 = std * (data['view2'][i].permute(1, 2, 0).numpy()) + mean
        depth1 = data['absolute_dept'][i].numpy().reshape(224, 224, 3)
        depth2 = data['relative_dept'][i].numpy().reshape(224, 224, 3)

        for j, (title, img) in enumerate([('view1', img1), ('view2', img2), ('abs_depth', depth1[:, :, 2]), ('rela_depth', depth2[:, :, 2])]):
            axes[j, i].imshow(img)
            axes[j, i].axis("off")
            axes[j, i].set_title(title)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def generate_data():
    """Load dataset and visualize sample images."""
    # Hardcoded file paths
    rgb_json = "rgb_path.json"
    thermal_json = "thermal_path.json"
    memmap1 = "abs_pos_3d.npy"
    memmap2 = "relative_pos_3d.npy"

    with open(rgb_json, "r") as f:
        rgb_paths = json.load(f)
    with open(thermal_json, "r") as f:
        thermal_paths = json.load(f)

    train_idx = list(range(0, int(len(thermal_paths) * 0.8)))
    test_idx = list(range(int(len(thermal_paths) * 0.8), len(thermal_paths)))
    shape = (len(rgb_paths) - 1, 224 * 224 * 3)

    train_dataset = ImagePairDataset(thermal_paths, memmap1, memmap2, train_idx, test_idx, shape=shape, train_mode=True)
    test_dataset = ImagePairDataset(thermal_paths, memmap1, memmap2, train_idx, test_idx, shape=shape, train_mode=False)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    sample_data = next(iter(test_loader))
    visualize(sample_data, save_path="sample_visualization.png")

    return train_loader, test_loader


if __name__ == "__main__":
    print("Generating dataset...")
    train_loader, test_loader = generate_data()
    print("Dataset preparation complete!")