import torch
import numpy as np

from numpy.polynomial import Polynomial


class FIRE_PR:
    def __init__(self, ref_tensor, gamma_model, model, A, rho, v_min, model_alpha_bars, fire_config, pr_channel = "pr_osf"):
        self.A = A
        self.v_min = v_min
        self.rho = rho
        self.singular_match = A.s_max
        self.gamma_model = gamma_model
        self.var_model = 1 / gamma_model
        self.model = model
        self.model_alpha_bars = model_alpha_bars

        # Ablation params
        self.use_stochastic_denoising = fire_config['use_stochastic_denoising']
        self.use_colored_noise = fire_config['use_colored_noise']
        self.estimate_nu = fire_config['estimate_nu']

        # CG params
        self.gam_w_correct = float(fire_config['gam_w_correct'])
        self.max_cg_iters = int(fire_config['max_cg_iters'])
        self.cg_tolerance = float(fire_config['cg_tolerance'])

        self.cg_initialization = torch.zeros_like(ref_tensor)

        with open(fire_config['nu_lookup'], 'rb') as f:
            self.scale_factor = np.load(f)

        # Quantized nu
        nu = np.sqrt(self.var_model) * self.scale_factor

        # find first t where sequence decreases
        first = np.argmin(np.diff(nu) >= 0)

        # start polynomial fit a bit earlier
        earlier = 100
        t_nofit = np.arange(first - earlier)
        t_fit = np.arange(first - earlier, 1000)

        # try polynomial fit in log domain
        logPoly_ffhq = Polynomial.fit(t_fit, np.log(nu[t_fit]), deg=10)
        self.nu_predict = lambda t: np.exp(logPoly_ffhq(t))
        self.channel = pr_channel

    def uncond_denoiser_function(self, noisy_im, noise_var, quantized_t):
        delta = np.minimum(noise_var / self.v_min, 1.)
        noise_var_clip = np.maximum(noise_var, self.v_min)

        if quantized_t:
            # raise NotImplementedError("Quantized t is not implemented for PR problems.")
            alpha_bars_model = 1 / (1 + 1 / self.gamma_model)
            diff = torch.abs(
                noise_var - (1 - torch.tensor(alpha_bars_model).to(noisy_im.device)) / torch.tensor(
                    alpha_bars_model).to(noisy_im.device))
            t = torch.argmin(diff).repeat(noisy_im.shape[0])
        else:
            t = torch.tensor(self.get_t_from_var(noise_var)).unsqueeze(0).repeat(noisy_im.shape[0]).to(noisy_im.device) # Need timestep for denoiser input

        alpha_bar = 1 / (1 + noise_var_clip)
        scaled_noisy_im = noisy_im * np.sqrt(alpha_bar)

        noise_predict = self.model(scaled_noisy_im, t)

        if noise_predict.shape[1] == 2 * noisy_im.shape[1]:
            noise_predict, _ = torch.split(noise_predict, noisy_im.shape[1], dim=1)

        noise_est = np.sqrt(noise_var_clip) * noise_predict

        x_0 = (1 - delta ** 0.5) * noisy_im + (delta ** 0.5) * (noisy_im - noise_est)

        return x_0, noise_var_clip, t

    

    def denoising_pr(self, r_0_255, gamma, quantized_t=False):
        # Max var
        noise_var = 1 / gamma
        noise_var_new = (4/(255**2))*noise_var

        q = (2/255)*r_0_255 - 1

        # Denoise
        x_bar_m1t1, noise_var, t = self.uncond_denoiser_function(q.float(), noise_var_new, quantized_t)
        # x_bar_m1t1 = x_bar_m1t1.clamp(min=-1, max=1)

        if quantized_t:
            # raise NotImplementedError("Quantized t is not implemented for PR problems.")
            lookup_t = np.argmin(np.abs(self.gamma_model - gamma))
            one_over_nu = 1 / (self.scale_factor[lookup_t] * np.sqrt(noise_var))
        else:
            one_over_nu = 1 / self.nu_predict(t[0].cpu().numpy())

        return x_bar_m1t1, one_over_nu



    def renoising_pr(self, r_und_cap, zeta, gamma_r):

        noise = torch.randn_like(r_und_cap)
        zeros = torch.zeros(r_und_cap.shape[0], 1).to(r_und_cap.device)

        gamma_r = self.rho * gamma_r
        max_prec = 4*self.model_alpha_bars[0] / ((255**2)*(1 - self.model_alpha_bars[0]))
        # gamma_r = torch.minimum(gamma_r, torch.ones(gamma_r.shape).to(gamma_r.device) * max_prec)
        gamma_r = gamma_r if gamma_r < max_prec else max_prec

        v_1 = 1 / gamma_r - 1 / zeta
        v_1 = torch.maximum(v_1, zeros)

        r_und = 1*r_und_cap + noise * v_1.sqrt()

        return r_und, gamma_r
    
    def get_t_from_var(self, noise_var):
        return np.minimum(999 * (np.sqrt(0.1 ** 2 + 2 * 19.9 * np.log(1 + noise_var)) - 0.1) / 19.9, 999)

    def reset(self):
        self.cg_initialization = torch.zeros_like(self.cg_initialization)


    def non_linear_estimation(self, mu, v, y, noise_sig, ext = False): 


        if self.channel == 'pr_osf':

            z_inp_complex = mu.reshape(mu.shape[0],3,self.A.oversampled_image_size,self.A.oversampled_image_size)
            z_inp = torch.stack([1*torch.real(z_inp_complex[:,:,:,:]), 1*torch.imag(z_inp_complex[:,:,:,:])], dim=1).contiguous()
            y_inp = y.reshape(mu.shape[0],3,self.A.oversampled_image_size,self.A.oversampled_image_size).contiguous()
            v_sig = (noise_sig**2)
            z_complex = torch.view_as_complex(torch.stack([1*z_inp[:,0,:,:,:], 1*z_inp[:,1,:,:,:]], dim=-1).contiguous())
            z_angle_inp = torch.angle(z_complex)
            z_abs_inp = torch.abs(z_complex)
            v_bar = 1*v

            z_foo = torch.view_as_real(torch.polar(y_inp, z_angle_inp)).permute(0,4,1,2,3).contiguous()
            z_hat = ((v_bar)/(v_bar + 2*v_sig))[:,:, None, None, None]*(z_foo) + ((2*v_sig)/(v_bar + 2*v_sig))[:,:, None, None, None]*(z_inp)

            v_hat_real_axis = ((v_sig)*v_bar)/(v_bar + 2*v_sig)
            v_hat_imag_axis = (v_bar[:,:, None, None]/(2*z_abs_inp))*(((v_bar)/(v_bar + 2*v_sig))[:,:, None, None]*(y_inp) + ((2*v_sig)/(v_bar + 2*v_sig))[:,:, None, None]*(z_abs_inp))
            v_hat_full = v_hat_real_axis[:,:, None, None] + v_hat_imag_axis
            mask_pos_ext = (z_abs_inp>(y_inp/2))
            v_hat = (torch.sum(v_hat_full*mask_pos_ext, dim = (1,2,3))/(torch.sum(mask_pos_ext, dim = (1,2,3)) + 1e-18)).unsqueeze(1)

            # eta_z = (1/v_hat)*torch.ones_like(v)
            eta_z = (1/v_hat)
            z_hat_complex = z_hat[:,0,:,:,:] + 1j*z_hat[:,1,:,:,:]
            z_cap = 1*z_hat_complex.reshape(mu.shape[0],-1).contiguous()
        
        elif self.channel == 'pr_cdp':
            
            z_inp_complex = mu.reshape(mu.shape[0],self.A.Sampling_Rate,3,self.A.image_size,self.A.image_size)
            z_inp = torch.cat([1*torch.real(z_inp_complex[:,:,:,:,:]), 1*torch.imag(z_inp_complex[:,:,:,:,:])], dim=1).contiguous()
            y_inp = y.reshape(mu.shape[0],self.A.Sampling_Rate,3,self.A.image_size,self.A.image_size).contiguous()
            v_sig = (noise_sig**2)
            z_complex = torch.view_as_complex(torch.stack([1*z_inp[:,0:self.A.Sampling_Rate,:,:,:], 1*z_inp[:,self.A.Sampling_Rate:,:,:,:]], dim=-1).contiguous())
            z_angle_inp = torch.angle(z_complex)
            z_abs_inp = torch.abs(z_complex)
            v_bar = 1*v

            z_foo_1 = torch.view_as_real(torch.polar(y_inp, z_angle_inp)).contiguous()
            z_foo = torch.cat([z_foo_1[:,:,:,:,:,0], z_foo_1[:,:,:,:,:,1]], dim=1).contiguous()

            z_hat = ((v_bar)/(v_bar + 2*v_sig))[:,:, None, None, None]*(z_foo) + ((2*v_sig)/(v_bar + 2*v_sig))[:,:, None, None, None]*(z_inp)

            v_hat_real_axis = ((v_sig)*v_bar)/(v_bar + 2*v_sig)
            v_hat_imag_axis = (v_bar[:,:, None, None, None]/(2*z_abs_inp))*(((v_bar)/(v_bar + 2*v_sig))[:,:, None, None, None]*(y_inp) + ((2*v_sig)/(v_bar + 2*v_sig))[:,:, None, None, None]*(z_abs_inp))
            v_hat_full = v_hat_real_axis[:,:, None, None, None] + v_hat_imag_axis
            mask_pos_ext = (z_abs_inp>(y_inp/2))
            v_hat = (torch.sum(v_hat_full*mask_pos_ext, dim = (1,2,3,4))/(torch.sum(mask_pos_ext, dim = (1,2,3,4)) + 1e-18)).unsqueeze(1)

            # eta_z = (1/v_hat)*torch.ones_like(v)
            eta_z = (1/v_hat)
            z_hat_complex = z_hat[:,0:self.A.Sampling_Rate,:,:,:] + 1j*z_hat[:,self.A.Sampling_Rate:,:,:,:]
            z_cap = 1*z_hat_complex.reshape(mu.shape[0],-1).contiguous()

        else:
            raise NotImplementedError

        if ext==True:

            eta_bar_z = eta_z - (1/v)
            z_bar = ((eta_z*z_cap) - (mu/v))*(1/eta_bar_z)

        else:
            eta_bar_z = 1*eta_z
            z_bar = 1*z_cap


        z_bar_und = self.A.Ht(z_bar)

        image_domain_noise_var = (1/eta_bar_z/2)            
        eta_bar_z_und = (1/image_domain_noise_var)


        return z_bar_und, eta_bar_z_und, z_bar, eta_bar_z
    

    def linear_estimation_pr(self, x_cap_und, z_bar_und, eta_x, eta_bar_z):
        
        right_term = eta_x[:, 0, None] * x_cap_und + eta_bar_z[:, 0, None] * z_bar_und
        
        nonzero_singular_mult = (eta_x[:, 0, None] + eta_bar_z[:, 0, None]) ** -1

        r_und_cap = nonzero_singular_mult * right_term

        zeta = eta_x[:, 0, None] + eta_bar_z[:, 0, None]
      
        return r_und_cap, zeta
    

    def run_fire_pr(self, x_t, y, sig_y, gamma_init, fire_iters, guidance=False, guidance_clean_image_m1t1=None, vp_prec_guidance=0, quantized_t = False):
        # 0. Initialize Values

        alpha_bar = 1 / (1 + 1 / gamma_init)

        x_hat = (x_t + np.sqrt(alpha_bar)) / ((2/255)*np.sqrt(alpha_bar)) # We need to convert everything in 0-255 scale since measurements are in 0-255 scale
        gamma_hat = 4*alpha_bar / ((255**2)*(1 - alpha_bar))


        if guidance:
            # guidance_noisy = guidance_clean_image_m1t1 + torch.randn_like(guidance_clean_image_m1t1) * (1/np.sqrt(vp_prec_guidance))
            # alpha_bar_guidance = 1 / (1 + 1 / vp_prec_guidance)
            # x_hat_guidance = (guidance_noisy + np.sqrt(alpha_bar_guidance)) / ((2/255)*np.sqrt(alpha_bar_guidance))
            # gamma_hat_guidance = 4*alpha_bar_guidance / ((255**2)*(1 - alpha_bar_guidance))

            # gamma = gamma_hat + gamma_hat_guidance
       
            # r = (gamma_hat* x_hat + gamma_hat_guidance * x_hat_guidance) / gamma

            # print("Using new guidance implementation")
            alpha_bar_guidance = 1 / (1 + 1 / vp_prec_guidance)
            gamma_hat_guidance = 4*alpha_bar_guidance / ((255**2)*(1 - alpha_bar_guidance))

            x_hat_guidance = 255*0.5*(guidance_clean_image_m1t1 + 1) + torch.randn_like(guidance_clean_image_m1t1) * (1/np.sqrt(gamma_hat_guidance))

            gamma = gamma_hat + gamma_hat_guidance
       
            r = (gamma_hat* x_hat + gamma_hat_guidance * x_hat_guidance) / gamma

        else:
            
            r = x_hat.clone()
            gamma = 1*gamma_hat

        for i in range(fire_iters):
            # print(i)
            # 1. Denoising

            x_bar_m1t1, one_over_nu = self.denoising_pr(r, gamma, quantized_t=quantized_t)

            if self.use_stochastic_denoising:
                x_bar_m1t1 = 1*x_bar_m1t1 + torch.randn_like(x_bar_m1t1) / np.sqrt(one_over_nu)
                one_over_nu = torch.tensor(one_over_nu / 2).unsqueeze(0).repeat(x_bar_m1t1.shape[0]).unsqueeze(1).to(x_bar_m1t1.device)


            x_cap = 255*0.5*(x_bar_m1t1 + 1)
            x_cap = x_cap.reshape(x_cap.shape[0], -1).contiguous()
            eta_x = (4/(255**2))*one_over_nu

            # 2. Non-Linear Estimation
            z_bar = self.A.H(x_cap) 
            v_bar_z = (1/4)*(1/eta_x) # simplification; specific for PR forward operators used

            z_bar_und, eta_bar_z_und,z_bar,eta_bar_z = self.non_linear_estimation(1*z_bar, 1*v_bar_z, 1*y, 1*sig_y, ext = True)

            if self.channel == 'pr_cdp': # not tested on OSF yet

                z_check = torch.view_as_real(self.A.H(x_cap))
                z_bar_new = torch.view_as_real(z_bar)
                my_m = z_bar_new.shape[1]
                eta_x_corrected = (1/((torch.sum((z_bar_new - z_check)**2, dim = (1,2)) - my_m/eta_bar_z[:,0])*(4/my_m))).reshape(eta_x.shape)
                
                #if any eta_x_corrected is negative, set it to eta value
                if torch.sum(eta_x_corrected<0)>0:
                    eta_x_corrected[eta_x_corrected<0] = eta_x[eta_x_corrected<0]

                eta_x = 1*eta_x_corrected

            # 3. Linear Estimation # simplification; specific for PR forward operators used
            r_und_cap, zeta_r_new  = self.linear_estimation_pr(x_cap, z_bar_und, eta_x, eta_bar_z_und)

            # 3. Re-Noising # simplification; specific for PR forward operators used
            r_und, gamma = self.renoising_pr(r_und_cap, zeta_r_new, gamma)
            r = r_und.view(r_und.shape[0], 3, 256, 256)


        return_val = x_cap.view(x_cap.shape[0], 3, 256, 256)
        return_val_m1t1 =  ((2/255)*return_val - 1)

        return return_val_m1t1.float()