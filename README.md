# Solving Inverse Problems using Diffusion with Iterative Colored Renoising [[TMLR]](https://openreview.net/pdf?id=RZv8FcQDPW)[[arXiv]](https://arxiv.org/pdf/2501.17468)

Official PyTorch implementation of the **Phase Retrieval (PR) experiments** from  
*Solving Inverse Problems using Diffusion with Iterative Colored Renoising*.  

This code is adapted for PR from [DDfire](https://github.com/matt-bendel/DDfire).

**Authors:** Matthew C. Bendel, Saurav K. Shastri, Rizwan Ahmad, and Philip Schniter

## Generate Measurements and Guidance Data

Run the following commands to generate the required measurements and guidance data:

```
python main_generate_and_save_measurements_and_run_HIO.py \
    --data_config=configs_pr/osf/ffhq/ffhq_osf_config_save_measurement.yaml \
    --poisson-alpha=8

python main_generate_and_save_measurements_and_run_HIO.py \
    --data_config=configs_pr/osf/imagenet/imagenet_osf_config_save_measurement.yaml \
    --poisson-alpha=8

python main_generate_and_save_measurements_and_run_HIO.py \
    --data_config=configs_pr/cdp/ffhq/ffhq_cdp_config_save_measurement.yaml \
    --poisson-alpha=45

python main_generate_and_save_measurements_and_run_HIO.py \
    --data_config=configs_pr/cdp/imagenet/imagenet_cdp_config_save_measurement.yaml \
    --poisson-alpha=45
```

## Running the Image Recovery Code

To run PR-OSF on FFHQ data:
```
python main_pr.py \
--model_config=configs_pr/ffhq_model_config.yaml \
--diffusion_config=configs_pr/diffusion_config.yaml \
--data_config=configs_pr/osf/ffhq/ffhq_osf_data_config.yaml \
--problem_config=configs_pr/osf/ffhq/pr_osf_ffhq_config.yaml \
--fire_config=configs_pr/fire_config_ffhq.yaml \
--sig_y=0.05 --nfes=800 --gpu=0
```

To run PR-CDP on FFHQ data:
```
python main_pr.py \
--model_config=configs_pr/ffhq_model_config.yaml \
--diffusion_config=configs_pr/diffusion_config.yaml \
--data_config=configs_pr/cdp/ffhq/ffhq_cdp_data_config.yaml \
--problem_config=configs_pr/cdp/pr_cdp_config.yaml \
--fire_config=configs_pr/fire_config_ffhq.yaml \
--sig_y=0.05 --nfes=100 --gpu=0
```

To run PR-OSF on ImageNet data:
```
python main_pr.py \
--model_config=configs_pr/imagenet_model_config.yaml \
--diffusion_config=configs_pr/diffusion_config.yaml \
--data_config=configs_pr/osf/imagenet/imagenet_osf_data_config.yaml \
--problem_config=configs_pr/osf/imagenet/pr_osf_imagenet_config.yaml \
--fire_config=configs_pr/fire_config_imagenet.yaml \
--sig_y=0.05 --nfes=800 --gpu=0
```

To run PR-CDP on ImageNet data:
```
python main_pr.py \
--model_config=configs_pr/imagenet_model_config.yaml \
--diffusion_config=configs_pr/diffusion_config.yaml \
--data_config=configs_pr/cdp/imagenet/imagenet_cdp_data_config.yaml \
--problem_config=configs_pr/cdp/pr_cdp_config.yaml \
--fire_config=configs_pr/fire_config_imagenet.yaml \
--sig_y=0.05 --nfes=100 --gpu=0
```

## Evaluating Performance
To evaluate PSNR and LPIPS performance, run:
```
python evaluate_samples_pr.py
```

To evaluate FID, we use the pytorch_fid module, which can be run by
```
python -m pytorch_fid /path/to/gt/images /path/to/reconstructed/images
```
