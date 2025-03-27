from generate_pseudo_gt import load_model, generate_pseudo_gt
from dataset_preparation import generate_data
import torch
import sys
sys.path.append("../dust3r") 
from dust3r.dust3r.model import load_model, AsymmetricCroCo3DStereo
import matplotlib.pyplot as plt
from train_model import train_model
from unet_model import unet_model


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "../mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"  # Update with actual path
    model = load_model(model_path, device)
    generate_pseudo_gt(model, device)
    print("Generating dataset...")
    train_loader, test_loader = generate_data()
    print("Dataset preparation complete!")
    model_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
    data = next(iter(test_loader))
    view1,view2 = {'img':data['view1'].to(device)},{'img':data['view2'].to(device)}
    a = model(view1,view2)
    plt.imshow(a[0]['pts3d'][2][:,:,2].detach().cpu().numpy())
    """Main function to execute training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_path = "checkpoints/dust3r_thermal.pth"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)

    if train_loader and test_loader:
        train_model(model, train_loader, test_loader)
    else:
        print("Error: Dataset not loaded!")  
    unet_model()