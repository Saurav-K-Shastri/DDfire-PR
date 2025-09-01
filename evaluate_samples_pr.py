import os
import cv2
import torch
import numpy as np
import lpips
from piq import LPIPS

# ----------------------
# Metric functions
# ----------------------
def compute_psnr(target, img2):
    target = np.clip(target, 0, 1)
    img2 = np.clip(img2, 0, 1)
    mse = np.mean((target - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def my_norm(x):
    return (x * 0.5 + 0.5).clip(0, 1)

# ----------------------
# Setup
# ----------------------
device_str = f"cuda:0" if torch.cuda.is_available() else 'cpu'
print(f"Using device {device_str}")
device = torch.device(device_str)

lpips_fn = LPIPS(replace_pooling=True, reduction='none').to(device)
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

# GT folders
gt_test_image_folder_ffhq = "/local/storage/FFHQ/ffhq256_firetest/0/" # TODO: CHANGE THIS TO DESIRED OUT PATH
gt_test_image_folder_imagenet = "/local/storage/imagenet_fire_testset/0/" # TODO: CHANGE THIS TO DESIRED OUT PATH

# Output folders to evaluate
output_folders = {
    "FFHQ_OSF": (gt_test_image_folder_ffhq, "/local/storage/DDfire_PR/results/FFHQ_dumm2/pr_osf/without_guidance/0/"), # TODO: CHANGE THIS TO DESIRED OUT PATH
    "FFHQ_CDP": (gt_test_image_folder_ffhq, "/local/storage/DDfire_PR/results/FFHQ_dummy2/pr_cdp/without_guidance/0/"), # TODO: CHANGE THIS TO DESIRED OUT PATH
    "ImageNet_OSF": (gt_test_image_folder_imagenet, "/local/storage/DDfire_PR/results/ImageNet_dummy2/pr_osf/with_guidance/0/"), # TODO: CHANGE THIS TO DESIRED OUT PATH
    "ImageNet_CDP": (gt_test_image_folder_imagenet, "/local/storage/DDfire_PR/results/ImageNet_dummy2/pr_cdp/without_guidance/0/"), # TODO: CHANGE THIS TO DESIRED OUT PATH
}

num_of_samples = 1000

# ----------------------
# Loop through all folders
# ----------------------
for label, (gt_folder, out_folder) in output_folders.items():
    PSNR_all_imgs = torch.zeros(num_of_samples, 1)
    LPIPS_all_imgs = torch.zeros(num_of_samples, 1)

    print("=====================================")
    print(f"Evaluating {label}")

    for test_image_number in range(num_of_samples):

        if "FFHQ" in label:
            num1 = 69000 + test_image_number
            file_name1 = f"{num1}.png"
        else:
            num1 = test_image_number
            file_name1 = f"{num1:05d}.png"

        num2 = 110000 + test_image_number
        file_name2 = f"{num2}.png"

        gt_img_path = os.path.join(gt_folder, file_name1)
        output_img_path = os.path.join(out_folder, file_name2)

        if not os.path.exists(gt_img_path) or not os.path.exists(output_img_path):
            raise FileNotFoundError(f"Missing file: {gt_img_path} or {output_img_path}")

        gt_image_0_255 = cv2.cvtColor(cv2.imread(gt_img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        recon_img_0_255 = cv2.cvtColor(cv2.imread(output_img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        gt_image = gt_image_0_255 / 255.0
        recon_img = recon_img_0_255 / 255.0

        PSNR_all_imgs[test_image_number, 0] = compute_psnr(gt_image, recon_img)

        x_m1t1 = torch.from_numpy(2 * gt_image - 1).permute(2, 0, 1).unsqueeze(0).float().to(device)
        sample_m1t1 = torch.from_numpy(2 * recon_img - 1).permute(2, 0, 1).unsqueeze(0).float().to(device)

        LPIPS_all_imgs[test_image_number, 0] = lpips_fn(my_norm(x_m1t1), my_norm(sample_m1t1))

    # Report averages
    print(f"[{label}] Average PSNR:  {PSNR_all_imgs.mean().item()}")
    print(f"[{label}] Average LPIPS: {LPIPS_all_imgs.mean().item()}")
    print("=====================================")
