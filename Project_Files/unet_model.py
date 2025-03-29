import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
import json
from torch.utils.data import Dataset, DataLoader
import wandb
import matplotlib.pyplot as plt
from train_model import train_model, ConfLoss, Regr3D
from PIL import Image
from PIL.ImageOps import exif_transpose
import random
import torchvision.transforms as transforms
import os
from tqdm import tqdm





class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.mp = nn.MaxPool2d(2, 2)

    def forward(self, x, apply_pool=True):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x1 = self.mp(x) if apply_pool else x  # Ensure x1 is always defined
        return x1, x

class CrossAttentionConv(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels,encoder_mode=True):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        if encoder_mode:
            self.key_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1)
            self.value_conv = nn.Conv2d(in_channels, value_channels, kernel_size=1)
        else:
            self.key_conv = nn.Conv2d(key_channels, key_channels, kernel_size=1)
            self.value_conv = nn.Conv2d(value_channels, value_channels, kernel_size=1)
            
        
        self.output_conv = nn.Conv2d(value_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        Q = self.query_conv(x1).view(B, -1, H * W)
        #print(x2.shape)
        K = self.key_conv(x2).view(B, -1, H * W)
        Q = Q.permute(0, 2, 1)
        attn = torch.bmm(Q, K) / (Q.shape[-1] ** 0.5)
        attn = self.softmax(attn)
        V = self.value_conv(x2).view(B, -1, H * W)
        attn_out = torch.bmm(V, attn.permute(0, 2, 1))
        attn_out = attn_out.view(B, -1, H, W)
        attn_out = self.output_conv(attn_out)
        return attn_out + x1

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, factor_channel=4):
        super().__init__()
        c1, c2, c3, c4 = 4 * factor_channel, 4 * (factor_channel**2), \
                         4 * (factor_channel**3), out_channels
        self.block1 = ConvBlock(in_channels, c1)
        self.block2 = ConvBlock(c1, c2)
        self.block3 = ConvBlock(c2, c3)
        self.block4 = ConvBlock(c3, c4)

        # Adjust Cross-Attention Channels to Match Concatenated Features
        self.cross_attn4 = CrossAttentionConv(c4 * 2, c3 * 2, c4 * 2)

    def forward(self, rgb, thermal):
        residuals = []
        
        rgb, r = self.block1(rgb)
        thermal, t = self.block1(thermal)
        residuals.append(torch.cat((r, t), dim=1))

        rgb, r = self.block2(rgb)
        thermal, t = self.block2(thermal)
        residuals.append(torch.cat((r, t), dim=1))

        rgb, r = self.block3(rgb)
        thermal, t = self.block3(thermal)
        residuals.append(torch.cat((r, t), dim=1))  

        rgb, r = self.block4(rgb, apply_pool=False)
        thermal, t = self.block4(thermal, apply_pool=False)

        # Ensure Consistency by Concatenating Before Attention
        fused = torch.cat((r, t), dim=1)
        cross_attn = self.cross_attn4(fused, fused)

        return cross_attn, residuals

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels+out_channels, out_channels, kernel_size=3, stride=1, padding=1)  # Fix channel mismatch
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, residual):
        x = self.up(x)
        #print(x.shape)
        x = torch.cat((x, residual), dim=1)  # Ensure channels match
        #print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, factor_channels=4):
        super().__init__()
        c1 = in_channels // factor_channels
        c2 = c1 // factor_channels
        c3 = c2 // factor_channels
        self.block1 = DeconvBlock(in_channels, c1)
        self.block2 = DeconvBlock(c1, c2)
        self.block3 = DeconvBlock(c2, c3)
        self.final_conv = nn.Conv2d(c3, out_channels, kernel_size=1)

    def forward(self, x, residual_arr):
        x = self.block1(x, residual_arr[-1])
        x = self.block2(x, residual_arr[-2])
        x = self.block3(x, residual_arr[-3])
        x = self.final_conv(x)
        return x

class model_unet_conv(nn.Module):
    def __init__(self,en_in_channels,en_out_channels,de_in_channels,de_out_channels):
        super().__init__()
        self.encoder = Encoder(in_channels=en_in_channels,out_channels=en_out_channels)
        self.decoder1 = Decoder(in_channels=de_in_channels,out_channels=de_out_channels)
        self.decoder2 = Decoder(in_channels=de_in_channels,out_channels=de_out_channels)
        self.conv1_depth = nn.Conv2d(in_channels=de_out_channels*2,out_channels=3,kernel_size=3,stride=1,padding=1)
        self.conv2_depth = nn.Conv2d(in_channels=de_out_channels*2,out_channels=3,kernel_size=3,stride=1,padding=1)
        self.conv1_conf = nn.Conv2d(in_channels=de_out_channels*2,out_channels=1,kernel_size=3,stride=1,padding=1)
        self.conv2_conf = nn.Conv2d(in_channels=de_out_channels*2,out_channels=1,kernel_size=3,stride=1,padding=1)

        self.cross_dec_attn1 = CrossAttentionConv(de_out_channels * 2, de_out_channels, de_out_channels,encoder_mode=False)
        self.cross_dec_attn2 = CrossAttentionConv(de_out_channels * 2, de_out_channels, de_out_channels,encoder_mode=False)
        self.up = nn.Upsample(size=(224,224))
        self.down = nn.Upsample(size=(64,64))
        
    def forward(self,rgb_view1,rgb_view2,thermal_view1,thermal_view2):
        b,_,w,h, = rgb_view1.shape
        encoded_view1,residual_view1 = self.encoder(rgb_view1,thermal_view1)
        encoded_view2,residual_view2 = self.encoder(rgb_view2,thermal_view2)
        output_view1 = self.decoder1(encoded_view1,residual_view1)
        output_view2 = self.decoder2(encoded_view2,residual_view2)
        
        inter_output_view1 = self.down(torch.cat((output_view1, output_view2), dim=1))
        inter_output_view2 = self.down(torch.cat((output_view2, output_view1), dim=1))

        output_view1 = self.down(output_view1)
        output_view2 = self.down(output_view2)
        #print(output_view1.shape,output_view2.shape)
        
        #print(self.down(inter_output_view2).shape)
        attn_output_view1 = self.cross_dec_attn1(inter_output_view1, output_view2)
        attn_output_view2 = self.cross_dec_attn2(inter_output_view2, output_view1)
        
        
        abs_depth = self.conv1_depth(self.up(attn_output_view1)).view(b,w,h,3)
        conf_view1 = 1 + torch.exp(self.conv1_conf(self.up(attn_output_view1)).view(b,w,h))
        rel_depth = self.conv2_depth(self.up(attn_output_view2)).view(b,w,h,3)
        conf_view2 = 1 + torch.exp(self.conv2_conf(self.up(attn_output_view2)).view(b,w,h))
        return ({'pts3d':abs_depth,'conf':conf_view1},{'conf':conf_view2,'pts3d_in_other_view':rel_depth})
    
def _resize_pil_image(img, long_edge_size):
    """ Resize an image while preserving aspect ratio based on long edge size. """
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    else:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)

class ImagePairDataset_rgb_thermal(Dataset):
    def __init__(self, rgb_paths, thermal_paths, np_memmap_file1, np_memmap_file2, train_idx, test_idx, dtype=np.float32, shape=None, image_size=224, square_ok=True, verbose=True,train_mode=True):
        """
        Args:
            rgb_paths (list): List of absolute rgb image paths.
            thermal_paths (list): List of absolute thermal image paths.
            np_memmap_file1 (str): Path to first `np.memmap` file.
            np_memmap_file2 (str): Path to second `np.memmap` file.
            dtype (type): Data type of the `memmap` file (default: np.float32).
            shape (tuple): Shape of the `memmap` array (e.g., `(num_samples, feature_dim)`).
            long_edge_size (int): Target size for long edge during resizing.
            square_ok (bool): Whether to allow square images.
            verbose (bool): Print debugging information.
            train_mode (bool) : Mode information.
            test_split (float) : Between 0-1 defining the train test split.
        """
        self.rgb_paths = sorted(rgb_paths)  # Ensure correct pairing order
        self.thermal_paths = sorted(thermal_paths) 
        self.image_size = image_size
        self.square_ok = square_ok
        self.verbose = verbose
        self.train_mode = train_mode
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.numpy_data1 = np.memmap(np_memmap_file1, dtype=dtype, mode='r', shape=shape)
        self.numpy_data2 = np.memmap(np_memmap_file2, dtype=dtype, mode='r', shape=shape)
            
        #     self.base_test_idx = int(len(self.image_paths)/self.test_split)
        #     self.base_test_idx = self.base_test_idx - len(self.image_paths)
            

        # Load np.memmap files
        

    def __len__(self):
        return len(self.rgb_paths) - 1  # Pair each image with the next

    def normalize(self,image):
        return ((image - 20000) / (25000 - 20000) * 255).astype(np.uint8)

    def apply_colormap(self,image):
        """ Apply colormap to a thermal image. """
        img_np = np.array(image)
        img_color = cv2.applyColorMap(img_np, cv2.COLORMAP_INFERNO)  # Other options: COLORMAP_HOT, COLORMAP_PARULA
        img_color = cv2.cvtColor(img_color,cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_color)
        
    def clahe_equalization(self,image):
        """ Apply CLAHE (Adaptive Histogram Equalization) to thermal images. """
        img_np = np.array(image)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[..., 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

    def rgb_process_image(self, img_path):
        """ Load, crop, resize, and normalize an image like DUSt3R """

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File {img_path} does not exist! Check the path.")
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        #plt.imshow(img)
        img = exif_transpose(img)  # Fix orientation
        img = np.asarray(img)[:, 280:1730]  # Crop width from 280 to 1730
        img = Image.fromarray(img)

        # Resize using `_resize_pil_image`
        W1, H1 = img.size  # Original size

        # Resize logic: Short side to 224 or long side to 512
        if self.image_size == 224:
            img = _resize_pil_image(img, round(self.image_size * max(W1 / H1, H1 / W1)))
        else:
            img = _resize_pil_image(img, self.image_size)

        W, H = img.size
        cx, cy = W // 2, H // 2  # Center coordinates

        # Apply central crop for square images
        if self.image_size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        # Convert to tensor & normalize with DUSt3R's ImgNorm
        #ImgNorm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5,0.5)])
        ImgNorm = transforms.Compose([
                                      transforms.ToTensor(), 
                                      transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
        img_tensor = ImgNorm(img)
        return img_tensor
    
    def thermal_process_image(self, img_path):
        """ Load, crop, resize, and normalize an image like DUSt3R """

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File {img_path} does not exist! Check the path.")
        
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img = self.normalize(img)
        img = Image.fromarray(img)
        #plt.imshow(img)
        img = exif_transpose(img)  # Fix orientation
        img = np.asarray(img)[:, 280:1730]  # Crop width from 280 to 1730
        img = Image.fromarray(img)

        # Resize using `_resize_pil_image`
        W1, H1 = img.size  # Original size

        # Resize logic: Short side to 224 or long side to 512
        if self.image_size == 224:
            img = _resize_pil_image(img, round(self.image_size * max(W1 / H1, H1 / W1)))
        else:
            img = _resize_pil_image(img, self.image_size)

        W, H = img.size
        cx, cy = W // 2, H // 2  # Center coordinates

        # Apply central crop for square images
        if self.image_size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        # Convert to tensor & normalize with DUSt3R's ImgNorm
        #ImgNorm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5,0.5)])
        ImgNorm = transforms.Compose([
                                      transforms.Lambda(lambda img: self.apply_colormap(img)),
                                      transforms.Lambda(lambda img: self.clahe_equalization(img)),
                                      transforms.ToTensor(), 
                                      transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
        img_tensor = ImgNorm(img)
        return img_tensor

    def __getitem__(self,_):
        # Get paths for image pairs
        if self.train_mode:
            L_idx = random.randint(0, len(self.rgb_paths) - 3)
            T_idx = self.train_idx[L_idx]
            rgb_img1_path = self.rgb_paths[L_idx]
            rgb_img2_path = self.rgb_paths[L_idx + 1]

            thermal_img1_path = self.thermal_paths[L_idx]
            thermal_img2_path = self.thermal_paths[L_idx + 1]
    
            # Process both images
            rgb_img1 = self.rgb_process_image(rgb_img1_path)
            rgb_img2 = self.rgb_process_image(rgb_img2_path)

            thermal_img1 = self.thermal_process_image(thermal_img1_path)
            thermal_img2 = self.thermal_process_image(thermal_img2_path)
    
    
            np_val1 = torch.tensor(self.numpy_data1[T_idx], dtype=torch.float32)
            np_val2 = torch.tensor(self.numpy_data2[T_idx], dtype=torch.float32)
    
            return {
                "rgb_view1": rgb_img1, 
                "rgb_view2": rgb_img2,
                "thermal_view1":thermal_img1,
                "thermal_view2":thermal_img2,
                "absolute_dept": np_val1,
                "relative_dept": np_val2,
            }
        else:
            L_idx = random.randint(0, len(self.rgb_paths) - 3)
            T_idx = self.test_idx[L_idx]
            
            rgb_img1_path = self.rgb_paths[L_idx]
            rgb_img2_path = self.rgb_paths[L_idx + 1]

            thermal_img1_path = self.thermal_paths[L_idx]
            thermal_img2_path = self.thermal_paths[L_idx + 1]
    
            # Process both images
            rgb_img1 = self.rgb_process_image(rgb_img1_path)
            rgb_img2 = self.rgb_process_image(rgb_img2_path)

            thermal_img1 = self.thermal_process_image(thermal_img1_path)
            thermal_img2 = self.thermal_process_image(thermal_img2_path)
    
    
            np_val1 = torch.tensor(self.numpy_data1[T_idx], dtype=torch.float32)
            np_val2 = torch.tensor(self.numpy_data2[T_idx], dtype=torch.float32)
    
            return {
                "rgb_view1": rgb_img1, 
                "rgb_view2": rgb_img2,
                "thermal_view1":thermal_img1,
                "thermal_view2":thermal_img2,
                "absolute_dept": np_val1,
                "relative_dept": np_val2,
            }
            
# Function to visualize RGB and thermal images
def visualize_rgb_thermal(data):
    std = np.array([0.5, 0.5, 0.5])
    mean = np.array([0.5, 0.5, 0.5])
    fig, axes = plt.subplots(6, 4, figsize=(10, 10))
    
    for i in range(4):    
        rgb_img1 = std * (data['rgb_view1'][i].permute(1, 2, 0).detach().numpy()) + mean
        rgb_img2 = std * (data['rgb_view2'][i].permute(1, 2, 0).detach().numpy()) + mean
        thermal_img1 = std * (data['thermal_view1'][i].permute(1, 2, 0).detach().numpy()) + mean
        thermal_img2 = std * (data['thermal_view2'][i].permute(1, 2, 0).detach().numpy()) + mean
        
        depth1 = data['absolute_dept'][i].detach().numpy().reshape(224, 224, 3)
        depth2 = data['relative_dept'][i].detach().numpy().reshape(224, 224, 3)
        
        axes[0, i].set_title('rgb_view1')
        axes[0, i].imshow(rgb_img1)
        axes[0, i].axis("off")
        
        axes[1, i].set_title('rgb_view2')
        axes[1, i].imshow(rgb_img2)
        axes[1, i].axis("off")
        
        axes[2, i].set_title('thermal_view1')
        axes[2, i].imshow(thermal_img1)
        axes[2, i].axis("off")
        
        axes[3, i].set_title('thermal_view2')
        axes[3, i].imshow(thermal_img2)
        axes[3, i].axis("off")
        
        axes[4, i].set_title('abs_depth')
        axes[4, i].imshow(depth1[:, :, 2])
        axes[4, i].axis("off")
        
        axes[5, i].set_title('rela_depth')
        axes[5, i].imshow(depth2[:, :, 2])
        axes[5, i].axis("off")
    
    plt.show()

# Training function
def train_rgb_thermal(NUM_EPOCHS, Model, criterion, train_loader, val_loader, optimizer, scheduler, device, use_wandb=True):
    loss_dic = {i: [] for i in range(NUM_EPOCHS)}
    val_dic = {i: [] for i in range(NUM_EPOCHS)}
    Model = Model.to(device)

    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        val_loss = 0.0
        num_train_batches = len(train_loader)
        num_val_batches = len(val_loader)

        Model.train()
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} Training")
        for data in train_progress_bar:
            rgb_view1, rgb_view2, thermal_view1, thermal_view2, gt1, gt2 = (
                data['rgb_view1'].to(device), data['rgb_view2'].to(device),
                data['thermal_view1'].to(device), data['thermal_view2'].to(device),
                data['absolute_dept'].to(device), data['relative_dept'].to(device)
            )
            gt1, gt2 = gt1.view(gt1.shape[0], 224, 224, 3), gt2.view(gt2.shape[0], 224, 224, 3)
            preds = Model(rgb_view1, rgb_view2, thermal_view1, thermal_view2)
            loss = criterion(gt1, gt2, preds[0], preds[1])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loss_dic[epoch].append(loss.item())
            if use_wandb:
                wandb.log({"training_loss": loss.item()})
            train_progress_bar.set_postfix({"loss": loss.item()})
        
        scheduler.step()

        Model.eval()
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} Validation")
            for data in val_progress_bar:
                rgb_view1, rgb_view2, thermal_view1, thermal_view2, gt1, gt2 = (
                    data['rgb_view1'].to(device), data['rgb_view2'].to(device),
                    data['thermal_view1'].to(device), data['thermal_view2'].to(device),
                    data['absolute_dept'].to(device), data['relative_dept'].to(device)
                )
                gt1, gt2 = gt1.view(gt1.shape[0], 224, 224, 3), gt2.view(gt2.shape[0], 224, 224, 3)
                preds = Model(rgb_view1, rgb_view2, thermal_view1, thermal_view2)
                loss_val = criterion(gt1, gt2, preds[0], preds[1])
                val_loss += loss_val.item()
                val_dic[epoch].append(loss_val.item())
                if use_wandb:
                    wandb.log({"validation_loss": loss_val.item()})
    
    return Model, loss_dic, val_dic

# Unnormalization function
def un_norm(data):
    std = np.array([0.5, 0.5, 0.5])
    mean = np.array([0.5, 0.5, 0.5])
    return std * data + mean

# UNet model training setup
def unet_model():
    with open("rgb_path.json", "r") as f:
        rgb_path = json.load(f)
    with open("thermal_path.json", "r") as f:
        thermal_path = json.load(f)
    
    filename1 = "abs_pos_3d.npy"
    filename2 = "relative_pos_3d.npy"
    shape = (len(rgb_path) - 1, 224 * 224 * 3)

    unet = model_unet_conv(3, 256, 512, 6)
    

    train_idx, test_idx = [i for i in range(0,int(len(thermal_path)*0.8))], [i for i in range(int(len(thermal_path)*0.8),len(thermal_path))]
    rgb_train_path,thermal_train_path = [rgb_path[i] for i in  train_idx],[thermal_path[i] for i in  train_idx]
    rgb_test_path,thermal_test_path = [rgb_path[i] for i in  test_idx],[thermal_path[i] for i in  test_idx]


    train_dataset = ImagePairDataset_rgb_thermal(rgb_train_path,thermal_train_path,filename1,filename2,shape=shape,train_idx=train_idx, test_idx=test_idx)
    test_dataset = ImagePairDataset_rgb_thermal(rgb_test_path,thermal_test_path,filename1,filename2,shape=shape, train_idx=train_idx, test_idx=test_idx, train_mode=False)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    
    visualize_rgb_thermal(next(iter(test_loader)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet.to(device)
    criterion = ConfLoss(Regr3D('L2', 'avg_dis'), alpha=0.2).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=1/3)
    
    wandb.login(key="ed7faaa7784428261467aee38c86ccc5c316f954")
    wandb.init(project="unet_thermal", name="unet_thermal", config={"epochs": 100})
    wandb.watch(unet, log_freq=5)
    train_rgb_thermal(100, unet, criterion, train_loader, test_loader, optimizer, scheduler, device)
