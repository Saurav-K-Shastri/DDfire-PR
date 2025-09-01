import yaml
import numpy as np
import torch
from tqdm.auto import tqdm
from guided_diffusion.fire import FIRE_PR


class DDfire:
    def __init__(self, fire_config, ref_tensor, model, model_betas, A, K, delta, eta_ddim=1.0, N_tot=1000,
                 quantize_ddim=False, guidance = False, guidance_t = 1000, pr_channel = "pr_osf"):
        self.K = K
        self.delta = delta
        self.N_tot = N_tot
        self.quantize_ddim = quantize_ddim
        self.eta = eta_ddim

        self.A = A
        self.model = model
        self.model_alphas = 1 - model_betas
        self.model_alpha_bars = np.cumprod(self.model_alphas)
        self.model_vars = (1 - self.model_alpha_bars) / self.model_alpha_bars

        vp_prec = 1 / self.model_vars  # Precision is inverse variance TODO: Update to variance..
        vp_prec_log = np.log10(vp_prec)

        vp_ddim_prec_log = np.linspace(np.max(vp_prec_log), np.min(vp_prec_log), self.K)
        vp_ddim_prec = 10 ** vp_ddim_prec_log

        all_steps = []
        for i in range(K):
            t = np.argmin(np.abs(vp_prec - vp_ddim_prec[i]), axis=0)
            all_steps.append(t)

        t_list = list(set(all_steps))
        t_list.sort()
        if quantize_ddim:
            # raise NotImplementedError("Quantized DDIM is not implemented for PR problems.")
            vp_ddim_prec = vp_prec[np.array(t_list)]
            self.K = len(vp_ddim_prec)

        self.N_k = []

        # set gam_tgt using delta, the fraction of 1-iter steps
        self.K_ddim1 = 1 + int(self.delta * (self.K - 1))  # number of 1-iter steps, in 1,...,K_ddim-1

        self.guidance = guidance
        if self.guidance:
            self.vp_prec_guidance = vp_prec[guidance_t-1]
            iters_ire_, rho = self.run_bisection_search(vp_ddim_prec + self.vp_prec_guidance)
        else:
            self.vp_prec_guidance = 0
            iters_ire_, rho = self.run_bisection_search(vp_ddim_prec)

        self.rho = rho
        self.N_k = iters_ire_.tolist()


        self.vp_ddim_prec = vp_ddim_prec
        self.alpha_bars = 1 / (1 + 1 / self.vp_ddim_prec)
        self.fire_runner = FIRE_PR(ref_tensor, vp_prec, model, A, rho, 1 / vp_prec[0], self.model_alpha_bars, fire_config, pr_channel)
        

    def run_bisection_search(self, vp_ddim_prec):
        # set gam_tgt using delta, the fraction of 1-iter steps
        log_gam_tgt = np.log10(vp_ddim_prec[self.K_ddim1 - 1])  # log-precision of first 1-iter step

        # determine rho and FIRE waterfilling schedule to determine rho
        log_gam_ddim_ = np.log10(vp_ddim_prec)
        gam_ddim_ = 10 ** log_gam_ddim_
        log_rho_min = 1e-5  # initialize bisection
        log_rho_max = 1e6  # initialize bisection
        quant = lambda inp: np.ceil(inp - 1e-5)  # small correction due to imperfect bisection
        NFE_min = np.sum(quant(np.maximum((log_gam_tgt - log_gam_ddim_) / log_rho_min, 0) + 1))
        NFE_max = np.sum(quant(np.maximum((log_gam_tgt - log_gam_ddim_) / log_rho_max, 0) + 1))
        NFE_tgt = self.N_tot
        max_bisection_iters = 80

        if NFE_max > NFE_tgt:
            raise ValueError('NFE_tgt must be > ' + str(NFE_max))

        # bisection
        it = 1
        while it < max_bisection_iters:
            it = it + 1
            log_rho_mid = 0.5 * (log_rho_min + log_rho_max)
            NFE_mid = np.sum(quant(np.maximum((log_gam_tgt - log_gam_ddim_) / log_rho_mid, 0) + 1))

            if NFE_mid > NFE_tgt:
                log_rho_min = log_rho_mid
            elif NFE_mid < NFE_tgt:
                log_rho_max = log_rho_mid
            else:
                log_rho_max = log_rho_mid  # okay, but try to improve...

        iters_ire_ = quant(np.maximum((log_gam_tgt - log_gam_ddim_) / log_rho_mid, 0) + 1).astype(int)

        rho = 10 ** log_rho_mid

        return iters_ire_, rho

    def p_sample_loop(self, x_start, y, sig_y=0.001, guidance_clean_image_m1t1 = None):
        """
        The function used for sampling from noise.
        """
        self.fire_runner.reset()

        x_t = x_start
        sig_prev = None

        pbar = tqdm(list(range(self.K))[::-1])
        for k in pbar:
            fire_iters = int(self.N_k[k])
            fire_prec = self.vp_ddim_prec[k]
            E_x_0_g_x_t_y = self.fire_runner.run_fire_pr(x_t, y, sig_y, fire_prec, fire_iters, guidance = self.guidance, guidance_clean_image_m1t1=guidance_clean_image_m1t1, vp_prec_guidance=self.vp_prec_guidance, quantized_t = self.quantize_ddim)
            x_t = self.ddim_update(x_t, E_x_0_g_x_t_y, k)

        return x_t.clamp(min=-1., max=1.)

    def ddim_update(self, x_t, E_x_0_g_x_t_y, k):
        alpha_bar = self.alpha_bars[k]
        alpha_bar_prev = 1. if k - 1 < 0 else self.alpha_bars[k - 1]
        sigma = (
                self.eta
                * np.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * np.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        c = np.sqrt((1 - alpha_bar_prev - sigma ** 2) / (1 - alpha_bar))

        # Equation 12.
        noise = torch.randn_like(x_t)

        x_t = c * x_t + (np.sqrt(alpha_bar_prev) - c * np.sqrt(alpha_bar)) * E_x_0_g_x_t_y
        if k != 0:
            x_t += sigma * noise

        return x_t
