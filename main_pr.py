# python main_pr.py \
# --model_config=configs_pr/ffhq_model_config.yaml \
# --diffusion_config=configs_pr/diffusion_config.yaml \
# --data_config=configs_pr/osf/ffhq/ffhq_osf_data_config.yaml \
# --problem_config=configs_pr/osf/ffhq/pr_osf_ffhq_config.yaml \
# --fire_config=configs_pr/fire_config_ffhq.yaml \
# --sig_y=0.05 --nfes=800 --gpu=0

# python main_pr.py \
# --model_config=configs_pr/imagenet_model_config.yaml \
# --diffusion_config=configs_pr/diffusion_config.yaml \
# --data_config=configs_pr/osf/imagenet/imagenet_osf_data_config.yaml \
# --problem_config=configs_pr/osf/imagenet/pr_osf_imagenet_config.yaml \
# --fire_config=configs_pr/fire_config_imagenet.yaml \
# --sig_y=0.05 --nfes=800 --gpu=0

# python main_pr.py \
# --model_config=configs_pr/ffhq_model_config.yaml \
# --diffusion_config=configs_pr/diffusion_config.yaml \
# --data_config=configs_pr/cdp/ffhq/ffhq_cdp_data_config.yaml \
# --problem_config=configs_pr/cdp/pr_cdp_config.yaml \
# --fire_config=configs_pr/fire_config_ffhq.yaml \
# --sig_y=0.05 --nfes=100 --gpu=0

# python main_pr.py \
# --model_config=configs_pr/imagenet_model_config.yaml \
# --diffusion_config=configs_pr/diffusion_config.yaml \
# --data_config=configs_pr/cdp/imagenet/imagenet_cdp_data_config.yaml \
# --problem_config=configs_pr/cdp/pr_cdp_config.yaml \
# --fire_config=configs_pr/fire_config_imagenet.yaml \
# --sig_y=0.05 --nfes=100 --gpu=0


from functools import partial
import argparse
import yaml
import types
import os

import torch
import matplotlib.pyplot as plt
import numpy as np

from guided_diffusion.unet import create_model
from guided_diffusion.DDfire import DDfire

from util.img_utils import clear_color
from util.logger import get_logger
from data.FFHQDataModule import FFHQDataModule
from data.PR_Measurement_Module import PRDataModule

from pytorch_lightning import seed_everything
from guided_diffusion.ddrm_svd import get_operator

import torchvision.transforms as transforms
from PIL import Image
from util.pr_utils import *

import lpips
from torchmetrics.functional import peak_signal_noise_ratio

def load_object(dct):
    return types.SimpleNamespace(**dct)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main(args):
    seed_everything(args.seed, workers=True)

    # logger
    logger = get_logger()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    problem_config = load_yaml(args.problem_config)
    data_config = load_yaml(args.data_config)
    fire_config = load_yaml(args.fire_config)

    if args.nfes < 100:
        diffusion_config["eta"] = 0.5

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    beta_start = 0.0001
    beta_end = 0.02
    model_betas = np.linspace(
        beta_start, beta_end, 1000, dtype=np.float64
    )

    if problem_config['guidance'] and problem_config['deg'] == 'pr_osf':
        os.makedirs(data_config["GLM_fire_out"] + f'/{problem_config["deg"]}/' + f'/with_guidance/0/', exist_ok=True)
    else:
        os.makedirs(data_config["GLM_fire_out"] + f'/{problem_config["deg"]}/' + f'/without_guidance/0/', exist_ok=True)  

    dm = FFHQDataModule(load_object(data_config))
    dm.setup()
    test_loader = dm.test_dataloader()

    dm_measurement = PRDataModule(load_object(data_config))
    dm_measurement.setup()
    test_loader_measurement = dm_measurement.test_dataloader()

       
    A = get_operator(problem_config, data_config, device)

    # Load diffusion sampler
    sampler = DDfire(fire_config, torch.ones(data_config["batch_size"], 3, 256, 256).to(device), model, model_betas, A, problem_config['K'], problem_config['delta'], problem_config['eta'], N_tot=args.nfes, quantize_ddim=True, guidance = problem_config['guidance'], guidance_t = problem_config['guidance_t'], pr_channel=problem_config['deg'])
    
    if problem_config["deg"] == 'pr_osf':

        num_alg_runs = 4

        my_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if problem_config['guidance']:
            final_HIO_image_inp_all_alg = torch.zeros(num_alg_runs,len(test_loader.dataset), 3, 256, 256)
            for alg_run in range(num_alg_runs):
                for image_idx in range(len(test_loader.dataset)):
                    HIO_image_location = data_config['guidance_image_path']+ f"image_"+str(image_idx)+"_"+str(alg_run)+".png"
                    final_HIO_image_inp_all_alg[alg_run, image_idx,:,:,:] = my_transform(Image.open(HIO_image_location).convert('RGB'))   


        for i, (data_test, data_measurement) in enumerate(zip(test_loader, test_loader_measurement)):
        
            logger.info(f"Inference for batch {i+1}/{len(test_loader)}")
            x = data_test[0]
            x = x.to(device)

            y_n = data_measurement[0].to(device)
            sig = data_measurement[1].to(device)
            
            x0 = 255*0.5*(data_test[0]+1)
            x0 = x0.to(device)
                    

            DDfire_recon_all_alg_runs_0_255 = torch.zeros((num_alg_runs, x.shape[0], x.shape[1], x.shape[2], x.shape[3]), device=device)
            true_image_number_start = i * x.shape[0]

            for alg_run in range(num_alg_runs):
                logger.info(f"Algorithm run {alg_run+1}/{num_alg_runs}")
                # Sampling
                if problem_config['guidance']:
                    final_HIO_image_inp = final_HIO_image_inp_all_alg[alg_run, true_image_number_start:true_image_number_start+x.shape[0],:,:,:].to(device)
                else:
                    final_HIO_image_inp = None

                with torch.no_grad():

                    torch.random.manual_seed(alg_run)
                    torch.manual_seed(alg_run)

                    x_start = torch.randn(x.shape, device=device)

                    sample = sampler.p_sample_loop(x_start, y_n, sig, guidance_clean_image_m1t1 = final_HIO_image_inp)

                    DDfire_recon_all_alg_runs_0_255[alg_run] = (255*(1*sample*0.5 + 0.5)).clamp(0,255)

            for image_idx in range(x.shape[0]):
        
                mse_DDfire = torch.tensor(float('inf'))
                best_DDfire = torch.zeros((1, x.shape[1], x.shape[2], x.shape[3]), device=device)
                for alg_run in range(num_alg_runs):
                    recon_foo = DDfire_recon_all_alg_runs_0_255[alg_run,image_idx,:,:,:].unsqueeze(0).contiguous()
                    # if recon foo has nan values, then skip this run
                    if torch.isnan(recon_foo).any():
                        continue
                    z_n = A.H(recon_foo)
                    z_n_abs = torch.abs(1*z_n)

                    mse_DDfire_foo = torch.mean((z_n_abs - y_n[image_idx].reshape(1,-1))**2)
                    if mse_DDfire_foo < mse_DDfire:
                        mse_DDfire = mse_DDfire_foo
                        best_DDfire = recon_foo

                true_image_number = i * x.shape[0] + image_idx

                fixed_image_np = correct_2d_shift_and_flip(x0[image_idx,:,:,:].cpu().numpy(), best_DDfire[0,:,:,:].cpu().numpy())
                fixed_image = torch.from_numpy(fixed_image_np)
                
                if problem_config['guidance']:
                    print("Using guidance...")
                    plt.imsave(data_config["GLM_fire_out"] + f'/{problem_config["deg"]}/' + f'/with_guidance/0/{110000 + true_image_number}.png', clear_color(fixed_image))
                else:
                    plt.imsave(data_config["GLM_fire_out"] + f'/{problem_config["deg"]}/' + f'/without_guidance/0/{110000 + true_image_number}.png', clear_color(fixed_image))

    elif problem_config["deg"] == 'pr_cdp':

        num_alg_runs = 1
        for i, (data_test, data_measurement) in enumerate(zip(test_loader, test_loader_measurement)):

            logger.info(f"Inference for batch {i+1}/{len(test_loader)}")
            x = data_test[0]
            x = x.to(device)

            y_n = data_measurement[0].to(device)
            sig = data_measurement[1].to(device)
            
            x0 = 255*0.5*(data_test[0]+1)
            x0 = x0.to(device)

            DDfire_recon_all_alg_runs_0_255 = torch.zeros((num_alg_runs, x.shape[0], x.shape[1], x.shape[2], x.shape[3]), device=device)
            for alg_run in range(num_alg_runs):
                logger.info(f"Algorithm run {alg_run+1}/{num_alg_runs}")
                final_HIO_image_inp = None

                with torch.no_grad():

                    torch.random.manual_seed(alg_run)
                    torch.manual_seed(alg_run)

                    x_start = torch.randn(x.shape, device=device)

                    sample = sampler.p_sample_loop(x_start, y_n, sig)

                    DDfire_recon_all_alg_runs_0_255[alg_run] = (255*(1*sample*0.5 + 0.5)).clamp(0,255)

            for image_idx in range(x.shape[0]):
                best_DDfire = torch.zeros((1, x.shape[1], x.shape[2], x.shape[3]), device=device)
                recon_foo = DDfire_recon_all_alg_runs_0_255[alg_run,image_idx,:,:,:].unsqueeze(0).contiguous()
                best_DDfire = recon_foo
                true_image_number = i * x.shape[0] + image_idx
                fixed_image_np = best_DDfire[0,:,:,:].cpu().numpy()
                fixed_image = torch.from_numpy(fixed_image_np)


                plt.imsave(data_config["GLM_fire_out"] + f'/{problem_config["deg"]}/' + f'/without_guidance/0/{110000 + true_image_number}.png', clear_color(fixed_image))

    else:

        raise NotImplementedError("Problem configuration not implemented.")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--data_config', type=str)
    parser.add_argument('--problem_config', type=str)
    parser.add_argument('--fire_config', type=str)
    parser.add_argument('--noiseless', action='store_true')
    parser.add_argument('--sig_y', type=float, default=0.05)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--nfes', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    
    main(args)

