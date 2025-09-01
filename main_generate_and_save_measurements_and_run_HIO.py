# python main_generate_and_save_measurements_and_run_HIO.py --data_config=configs_pr/osf/ffhq/ffhq_osf_config_save_measurement.yaml --poisson-alpha=8
# python main_generate_and_save_measurements_and_run_HIO.py --data_config=configs_pr/osf/imagenet/imagenet_osf_config_save_measurement.yaml --poisson-alpha=8

# python main_generate_and_save_measurements_and_run_HIO.py --data_config=configs_pr/cdp/ffhq/ffhq_cdp_config_save_measurement.yaml --poisson-alpha=45
# python main_generate_and_save_measurements_and_run_HIO.py --data_config=configs_pr/cdp/imagenet/imagenet_cdp_config_save_measurement.yaml --poisson-alpha=45

from functools import partial
import argparse
import yaml
import types
import lpips
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from util.logger import get_logger
from data.FFHQDataModule import FFHQDataModule
from data.ImageDataModule import ImageDataModule2
from pytorch_lightning import seed_everything
from guided_diffusion.ddrm_svd import OSF, CDP
from util.pr_utils import *
from util.img_utils import clear_color

import torchvision.transforms as transforms
from PIL import Image


def load_object(dct):
    return types.SimpleNamespace(**dct)


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    # torch.set_default_dtype(torch.float64)

    parser = argparse.ArgumentParser()
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--data_config', type=str)
    parser.add_argument('--noiseless', action='store_true')
    parser.add_argument('--verbose', action='store_true') # Whether or not to print IRE iteration values
    parser.add_argument('--sig_y', type=float, default=0.05)
    parser.add_argument('--noise-type', type=str, default='Poisson')
    parser.add_argument('--poisson-alpha', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    seed_everything(args.seed, workers=True)

    # logger
    logger = get_logger()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'

    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    data_config = load_yaml(args.data_config)

    image_type = data_config.get("image_type")


    if image_type == 'ffhq':
        dm = FFHQDataModule(load_object(data_config))
    elif image_type == 'imagenet':
        dm = ImageDataModule2(data_config)
    else:
        raise ValueError(f"Unknown image type: {image_type}")

    dm.setup()
    test_loader = dm.test_dataloader()

    chosen_deg = data_config.get("deg")

    accepted_operators = ['pr_osf', 'pr_cdp']
    if chosen_deg not in accepted_operators:
        raise RuntimeError('Unknown degradation.')

    sig_y = float(args.sig_y)
    if args.noiseless:
        sig_y = 0.001 # Imperceptible noise, but enough so that FIRE plays nicely


    if chosen_deg == 'pr_osf':
        print("Using OSF for phase retrieval.")
        imsize = 256
        d = imsize*imsize
        p = 4
        m = p*d
        oversampled_image_size = int(np.sqrt(m))       
        support = torch.zeros(1,1,3,oversampled_image_size,oversampled_image_size)
        pad = (oversampled_image_size - imsize)
        pad_left = pad//2 # same amount at top
        pad_right = pad - pad_left # same amount at bottom
        support[:,:,:,pad_left:pad_left+imsize,pad_left:pad_left+imsize] = 1
        support = support.to(device)
        H = OSF(3,imsize, p, device)
    elif chosen_deg == 'pr_cdp':
        print("Using CDP for phase retrieval.")
        imsize = 256
        d = imsize*imsize
        p = 4
        m = p*d
        SamplingRate = p

        fixed_rand_mat = torch.load('./guided_diffusion/fixed_rand_mat_CDP.pt')
        fixed_rand_mat_3_chan_6_dim = torch.zeros(1,SamplingRate,3,imsize,imsize,2)
        for i in range(SamplingRate):
            for k in range(3):
                fixed_rand_mat_3_chan_6_dim[:,i,k,:,:,:] = fixed_rand_mat[:,i,:,:,:]

        H = CDP(3,imsize, fixed_rand_mat_3_chan_6_dim.to(device), device)

    else:
        raise NotImplementedError


    os.makedirs(data_config["measurement_save_path_test"] + f'/{chosen_deg}', exist_ok=True)

    y_mat = torch.zeros(len(test_loader), int(3*imsize*imsize*p))
    sig_mat = torch.zeros(len(test_loader), 1)
    
    print(" ")
    print("Image type: ", image_type)
    print("Alpha value: ", args.poisson_alpha)
    print("Noise type: ", args.noise_type)
    print("Degradation: ", chosen_deg)
    print(" ")

    for i, data in enumerate(test_loader):
        
        if i%50 == 0:
            logger.info(f"Measurement collection for images {i} - {i+50}.")
            
        x = data[0]
        x_0_255 = 255*0.5*(data[0]+1)
        x = x.to(device)
        x_0_255 = x_0_255.to(device)
        # Sampling
        with torch.no_grad():

            if chosen_deg == 'pr_osf':
                if args.noise_type == 'Poisson':

                    torch.random.manual_seed(i)
                    torch.manual_seed(i)

                    z_n = H.H(x_0_255)
                    z_n_abs = torch.abs(1*z_n)
                    noise_mat = torch.randn_like(z_n_abs)
                    z_n_abssq = z_n_abs**2
                    intensity_noise = args.poisson_alpha*z_n_abs*noise_mat

                    y_n_sq = torch.clamp(z_n_abssq + intensity_noise, min=0)

                    y_n = torch.abs(torch.sqrt(y_n_sq))
                    
                    err = y_n - z_n_abs
                    sig_y = (torch.std(err)).reshape(-1,1)
                else:
                    raise NotImplementedError("Noise type not implemented for OSF.")

            elif chosen_deg == 'pr_cdp':
                if args.noise_type == 'Poisson':
    
                    z_n = H.H(x_0_255)
                    z_n_abs = torch.abs(1*z_n)
                    noise_mat = torch.randn_like(z_n_abs)
                    z_n_abssq = z_n_abs**2
                    intensity_noise = args.poisson_alpha*z_n_abs*noise_mat

                    y_n_sq = torch.abs(z_n_abssq + intensity_noise)

                    y_n = torch.abs(torch.sqrt(y_n_sq))
                    
                    err = y_n - z_n_abs
                    sig_y = (torch.std(err)).reshape(-1,1)         
                else:
                    raise NotImplementedError("Noise type not implemented for CDP.")       

            else:
                raise NotImplementedError


            y_mat[i] = y_n[0,:].cpu()
            sig_mat[i] = sig_y[0].cpu()


    torch.save(y_mat, data_config["measurement_save_path_test"] + f'/{chosen_deg}/y_mat_alpha_{args.poisson_alpha}.pt')
    torch.save(sig_mat, data_config["measurement_save_path_test"] + f'/{chosen_deg}/sig_mat_alpha_{args.poisson_alpha}.pt')

    print(" ")
    print(" ")
    print("Measurement Saved!")

    if image_type == 'ffhq':
        ref_avg_img = torch.load('FFHQ_avg_of_7000_images_m1t1.pt').to(device)
        ref_avg_img_0_255 = (255*0.5*(1*ref_avg_img + 1))
    elif image_type == 'imagenet':
        ref_avg_img = torch.load('ImageNet_avg_of_7000_images_m1t1.pt').to(device)
        ref_avg_img_0_255 = (255*0.5*(1*ref_avg_img + 1)).unsqueeze(0)  # Add batch dimension
    else:
        raise ValueError(f"Unknown image type: {image_type}")
    
    print(" ")
    print(" ")
    print("Running HIO Algorithm...")
    print(" ")
    print(" ")

    if chosen_deg == 'pr_osf':
        os.makedirs(data_config["hio_result_save_path"] + f'/{chosen_deg}/alpha_{args.poisson_alpha}/samples/', exist_ok=True)
        os.makedirs(data_config["hio_result_save_path"] + f'/{chosen_deg}/alpha_{args.poisson_alpha}/samples_final/0/', exist_ok=True)
    elif chosen_deg == 'pr_cdp':
        os.makedirs(data_config["hio_result_save_path"] + f'/{chosen_deg}/alpha_{args.poisson_alpha}/samples_final/0/', exist_ok=True)
    else:
        raise NotImplementedError("Unknown degradation type for saving samples.")



    for alg_run in range(data_config.get("total_num_hio_runs", 1)):
        print("++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++")
        print(" ")
        print(f"HIO Run # {alg_run+1} of {data_config.get('total_num_hio_runs')}")
        print(" ")
        print("++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++")


        for i in range(len(test_loader)):
                
            if i%50 == 0:
                logger.info(f"Inference for images {i} - {i+50}.")

            # HIO Initialization
            if chosen_deg == 'pr_osf':
                # HIO realted
                y_2D = y_mat[i].to(device).reshape(1, 1, 3, 512, 512)
                y_2D_complex = torch.view_as_complex(torch.stack([y_2D, 0*y_2D], dim=-1)).contiguous()
                A_op_complex_foo = A_op_complex_HIO_OSF(p,imsize)
                beta_HIO = data_config.get("beta_HIO")
                HIO_iter_trials = data_config.get("HIO_iter_trials")
                HIO_iter_final = data_config.get("HIO_iter_final")
                number_of_trials = data_config.get("number_of_trials")

                x_rec_HIO_best = get_the_best_HIO_recon(1*y_2D_complex, A_op_complex_foo, beta_HIO, HIO_iter_trials ,HIO_iter_final,  support, d, m, pad_left, imsize, number_of_trials = number_of_trials, algo_trial_num = alg_run)
                x_rec_HIO_best_orientation_corrected_correlation = fix_channel_orientation_using_correlation(1*x_rec_HIO_best)

                final_HIO_image = torch.clamp(fix_flip_ambiguity(x_rec_HIO_best_orientation_corrected_correlation, ref_avg_img_0_255)[[0],0,:,:,:], 0, 255)

                sample = (2/255)*final_HIO_image - 1

                # Save image for FID...
                for j in range(sample.shape[0]):
                    plt.imsave(f'{data_config["hio_result_save_path"]}/{chosen_deg}/alpha_{args.poisson_alpha}/samples/image_{i+j}_{alg_run}.png', clear_color(sample[j].unsqueeze(0)))

            elif chosen_deg == 'pr_cdp':
                # HIO realted
                y_2D = y_mat[i].to(device).reshape(1, 4, 3, 256, 256)
                y_2D_complex = torch.view_as_complex(torch.stack([y_2D, 0*y_2D], dim=-1)).contiguous()

                A_op_complex_foo = A_op_complex_HIO_CDP(fixed_rand_mat_3_chan_6_dim.to(device))

                beta_HIO = data_config.get("beta_HIO")
                HIO_iter_final = data_config.get("HIO_iter_final")

                x_rec_HIO_best = HIO_CDP(1*y_2D_complex, A_op_complex_foo, beta_HIO, HIO_iter_final, device = device)

                final_HIO_image = torch.clamp(x_rec_HIO_best, 0, 255)

                # psnr_vals.append(peak_signal_noise_ratio(1*final_HIO_image, x).mean().detach().cpu().numpy())
                sample = (2/255)*final_HIO_image - 1

                # Save image for FID...
                for j in range(sample.shape[0]):
                    plt.imsave(f'{data_config["hio_result_save_path"]}/{chosen_deg}/alpha_{args.poisson_alpha}/samples_final/0/{110000 + i}.png', clear_color(sample[j].unsqueeze(0))) # works since batch size is 1

    # We need to select the best HIO image from the multiple algorithm runs for OSF
    print(" ")
    print(" ")
    print("Selecting the best HIO image from multiple algorithm runs for OSF-PR...")
    print(" ")
    print(" ")
     
    if chosen_deg == 'pr_osf':

        for i, data_test in enumerate(test_loader):

            x = 255*0.5*(data_test[0]+1)
            x = x.to(device)
            y_n = y_mat[i].to(device)

            test_image_number = 1*i

            x0 = torch.zeros(1,2,3,imsize,imsize).to(device)
            x0[:,0,:,:,:] = 1*x

            if i%50 == 0:
                logger.info(f"Selecting and saving best HIO image for test images {i} - {i+50}.")

            HIO_recon_all_alg_runs = torch.zeros(4,1,3,imsize,imsize)

            for alg_run in range(data_config.get("total_num_hio_runs", 1)):

                HIO_image_location = data_config["hio_result_save_path"]+f"/{chosen_deg}/alpha_{args.poisson_alpha}/samples/image_"+str(test_image_number)+"_"+str(alg_run)+".png"

                my_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

                HIO_image_m1t1 = my_transform(Image.open(HIO_image_location).convert('RGB'))

                x_rec_HIO_tensor = torch.zeros_like(x0)
                x_rec_HIO_tensor[0,0,:,:,:] = 255*0.5*(HIO_image_m1t1+1)

                HIO_recon_all_alg_runs[alg_run,0,:,:,:] = x_rec_HIO_tensor[0,0,:,:,:]

            mse_HIO = torch.tensor(float('inf'))
            best_HIO_recon = torch.zeros_like(x0)

            for alg_run in range(4):
                z_n = H.H(HIO_recon_all_alg_runs[alg_run,:,:,:,:].to(device))
                z_n_abs = torch.abs(1*z_n)

                mse_HIO_foo = torch.mean((z_n_abs - y_n)**2)
                if mse_HIO_foo < mse_HIO:
                    mse_HIO = mse_HIO_foo
                    best_HIO_recon = HIO_recon_all_alg_runs[alg_run,:,:,:,:]
                            

            fixed_image_np = correct_2d_shift_and_flip(x0[0,0,:,:,:].cpu().numpy(), best_HIO_recon[0,:,:,:].numpy())
            fixed_image = torch.from_numpy(fixed_image_np)

            plt.imsave(f'{data_config["hio_result_save_path"]}/{chosen_deg}/alpha_{args.poisson_alpha}/samples_final/0/{110000 + test_image_number}.png', clear_color(fixed_image)) # works since batch size is 1


if __name__ == '__main__':
    main()
