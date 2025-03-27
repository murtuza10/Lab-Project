import sys
sys.path.append("..")
sys.path.append("../dust3r")
from dust3r.model import AsymmetricCroCo3DStereo
from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from dust3r.utils.image import load_images
import os
from tqdm import tqdm
import numpy as np
import json
import torch


def load_file_paths(base_path):
    child_path1 = [os.path.join(base_path, i) for i in os.listdir(base_path)]
    child_path2 = [os.path.join(i, j) for i in child_path1 for j in os.listdir(i)]

    base_rgb_path = [os.path.join(i, 'fl_rgb') for i in child_path2]
    base_thermal_path = [os.path.join(i, 'fl_ir_aligned') for i in child_path2]

    rgb_path = sorted([os.path.join(i, j) for i in base_rgb_path for j in os.listdir(i)])
    thermal_path = sorted([os.path.join(i, j) for i in base_thermal_path for j in os.listdir(i)])

    # Remove corrupted or misaligned image
    del thermal_path[5156]
    del rgb_path[5156]

    with open("rgb_path.json", 'w') as f:
        json.dump(rgb_path, f, indent=2)

    with open("thermal_path.json", 'w') as f:
        json.dump(thermal_path, f, indent=2)

    return rgb_path, thermal_path


def run_inference_on_pair(image_arr, model, device):
    images = load_images(image_arr, size=224, verbose=False)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    pred1 = output['pred1']
    pred2 = output['pred2']

    pts1 = pred1['pts3d'].detach().cpu().numpy()
    pts2 = pred2['pts3d_in_other_view'].detach().cpu().numpy()
    return pts1, pts2


def generate_pseudo_gt(model, device):
    base_path = '/home/nfs/inf6/data/datasets/ThermalDBs/Freiburg/train/'
    rgb_path, _ = load_file_paths(base_path)

    shape = (len(rgb_path) - 1, 224 * 224 * 3)
    dtype = np.float32

    filename1 = "abs_pos_3d.npy"
    filename2 = "relative_pos_3d.npy"

    default_path = []

    mode = 'w+' if not os.path.exists(filename1) and not os.path.exists(filename2) else 'r+'
    fp1 = np.memmap(filename1, dtype=dtype, mode=mode, shape=shape)
    fp2 = np.memmap(filename2, dtype=dtype, mode=mode, shape=shape)

    for i in tqdm(range(len(rgb_path) - 1)):
        image_arr = [rgb_path[i], rgb_path[i + 1]]
        try:
            pred1, pred2 = run_inference_on_pair(image_arr, model, device)
            fp1[i, :] = pred1.flatten()
            fp2[i, :] = pred2.flatten()
        except Exception:
            default_path.append(image_arr)

    fp1.flush()
    fp2.flush()
    
def load_model(model_path,device):
    model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
    return model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "../mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"  # Update with actual path
    model = load_model(model_path, device)
    generate_pseudo_gt(model, device)