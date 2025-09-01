
import torch
import numpy as np
from scipy import signal

from fastmri_utils2.fftc import fft2c_new as fft2c
from fastmri_utils2.fftc import ifft2c_new as ifft2c
from fastmri_utils2.math import (
    complex_abs,
    complex_abs_sq,
    complex_conj,
    complex_mul,
    tensor_to_complex_np,
)

import time

from scipy.fft import fft2, ifft2, fftshift


def HIO_CDP(y, A_op_complex_foo, beta, iters, x_init=None, seed_num=0, device = 'cuda:0'):
    # HIO works best when dealing with images scaled from 0-255, so scale input accordingly
    
    # Random initial guess
    if x_init is None:
        torch.manual_seed(seed_num)
        x = (torch.rand(A_op_complex_foo.H(y).shape) + 0*1j).to(device)
    else:
        x = x_init

    # Define projection operator
    def Pm(x):
        return A_op_complex_foo.H(y * torch.exp(1j * torch.angle(A_op_complex_foo.A(x))))

    # Main loop
    for k in range(iters):
        Pmx = Pm(x)
        inds = ((Pmx.real > 0)).bool()

        x[inds] = Pmx[inds]
        x[~inds] = x[~inds] - beta * Pmx[~inds]
  
    return torch.real(x)

class A_op_complex_HIO_CDP:
    
    def __init__(self,fixed_rand_mat):
        self.fixed_rand_mat = fixed_rand_mat
        self.Sampling_Rate = self.fixed_rand_mat.shape[1]

    def A(self,X):
        X1 = torch.view_as_real(X.contiguous())

        X1_new = X1.repeat(1,self.Sampling_Rate,1,1,1,1).contiguous()
        fixed_rand_mat_new = self.fixed_rand_mat.repeat(X1_new.shape[0],1,1,1,1,1).contiguous()
        # add repeat map
        X2 = complex_mul(X1_new,fixed_rand_mat_new)
        out = fft2c(X2)*np.sqrt(1/self.Sampling_Rate)

        return torch.view_as_complex(out.contiguous())

    def H(self,X):
        X1 = torch.view_as_real(X.contiguous())

        X2 = ifft2c(X1)
        fixed_rand_mat_new = self.fixed_rand_mat.repeat(X2.shape[0],1,1,1,1,1)
        out = (torch.sum(complex_mul(X2, complex_conj(fixed_rand_mat_new)),dim = 1))*np.sqrt(1/self.Sampling_Rate)
        out2 = out.unsqueeze(1).contiguous()
        out2[:,:,:,:,:,1] = 0
        return torch.view_as_complex(out2.contiguous())
    


class A_op_CDP:
    
    def __init__(self,fixed_rand_mat):
        self.fixed_rand_mat = fixed_rand_mat
        self.Sampling_Rate = self.fixed_rand_mat.shape[1]

    def A(self,X):
        X1 = X.permute(0,2,3,4,1).unsqueeze(1)
        X1_new = X1.repeat(1,self.Sampling_Rate,1,1,1,1).contiguous()
        fixed_rand_mat_new = self.fixed_rand_mat.repeat(X1_new.shape[0],1,1,1,1,1).contiguous()
        # add repeat map
        X2 = complex_mul(X1_new,fixed_rand_mat_new)
        out = fft2c(X2)*np.sqrt(1/self.Sampling_Rate)
        return torch.cat([out[:,:,:,:,:,0], out[:,:,:,:,:,1]], dim=1).contiguous()

    def H(self,X):
        X1 = torch.stack([X[:,0:self.Sampling_Rate,:,:,:], X[:,self.Sampling_Rate:,:,:,:]], dim=-1).contiguous()
        X2 = ifft2c(X1)
        fixed_rand_mat_new = self.fixed_rand_mat.repeat(X2.shape[0],1,1,1,1,1)
        out = (torch.sum(complex_mul(X2, complex_conj(fixed_rand_mat_new)),dim = 1))*np.sqrt(1/self.Sampling_Rate)
        return out.permute(0,4,1,2,3).contiguous()


class A_op_OSF:
    
    def __init__(self,oversampling_factor, image_size):
        self.oversampling_factor = oversampling_factor
        self.image_size = image_size
        self.n = self.image_size*self.image_size
        self.m = int(self.n*self.oversampling_factor)
        self.oversampled_image_size = int(np.sqrt(self.m))
        self.pad = (self.oversampled_image_size - self.image_size)
        self.pad_left = self.pad//2 # same amount at top
        self.pad_right = self.pad - self.pad_left # same amount at bottom

    def A(self,X):
        X1  = torch.nn.functional.pad(X, (self.pad_left, self.pad_right, self.pad_left, self.pad_right))
        X2 = X1.permute(0,2,3,4,1).unsqueeze(1)
        out = fft2c(X2)
        return torch.cat([out[:,:,:,:,:,0], out[:,:,:,:,:,1]], dim=1).contiguous()

    def H(self,X):
        X1 = torch.stack([X[:,0,:,:,:], X[:,1,:,:,:]], dim=-1).contiguous()
        X2 = ifft2c(X1)
        out = X2.permute(0,4,1,2,3).contiguous()
        return out[:,:,:,self.pad_left:self.pad_left+self.image_size,self.pad_left:self.pad_left+self.image_size].contiguous()

class A_op_Fourier:
    
    def __init__(self,oversampling_factor, image_size):
        self.oversampling_factor = oversampling_factor
        self.image_size = image_size

    def A(self,X):
        X2 = X.permute(0,2,3,4,1).unsqueeze(1)
        out = fft2c(X2)
        return torch.cat([out[:,:,:,:,:,0], out[:,:,:,:,:,1]], dim=1).contiguous()

    def H(self,X):
        X1 = torch.stack([X[:,0,:,:,:], X[:,1,:,:,:]], dim=-1).contiguous()
        X2 = ifft2c(X1)
        out = X2.permute(0,4,1,2,3).contiguous()
        return out.contiguous()
    
    
class A_op_complex_HIO_OSF:
    
    def __init__(self, oversampling_factor, image_size):
        self.oversampling_factor = oversampling_factor
        self.image_size = image_size
        self.n = self.image_size*self.image_size
        self.m = int(self.n*self.oversampling_factor)

        self.oversampled_image_size = int(np.sqrt(self.m))
        self.pad = (self.oversampled_image_size - self.image_size)
        self.pad_left = self.pad//2 # same amount at top
        self.pad_right = self.pad - self.pad_left # same amount at bottom

    def A(self,X):
        X1 = torch.view_as_real(X.contiguous())
        out = fft2c(X1)
        return torch.view_as_complex(out.contiguous())

    def H(self,X):
        X1 = torch.view_as_real(X.contiguous())
        out = ifft2c(X1)
        out[:,:,:,:,:,1] = 0
        return torch.view_as_complex(out.contiguous())


def HIO(y, A_op_complex_foo, beta, iters, support, x_init=None, seed_num=0):
    # HIO works best when dealing with images scaled from 0-255, so scale input accordingly
    
    # Random initial guess
    if x_init is None:
        torch.manual_seed(seed_num)
        x = 255*torch.rand(A_op_complex_foo.H(y).shape, device = y.device ) + 0*1j
        # x = x.to(y.device)
        inds = (support).bool()
        x[~inds] = 0
        # x[0] = 1
    else:
        x = x_init

    # Define projection operator
    def Pm(x):
        return A_op_complex_foo.H(y * torch.exp(1j * torch.angle(A_op_complex_foo.A(x))))

    # Main loop
    for k in range(iters):
        Pmx = Pm(x)

        inds = (support * (Pmx.real > 0)).bool()

        x[inds] = Pmx[inds]
        # x[~inds] = x[~inds] - betaPmx[~inds]
        x[~inds] = x[~inds] - beta * Pmx[~inds]

    inds = (support).bool()
    x[~inds] = 0
    
    return x


def get_the_best_HIO_recon(y_complex, A_op_complex_foo, beta_HIO, HIO_iter_trials ,HIO_iter_final,  support, d, m, pad_left, resize_size, number_of_trials = 50, algo_trial_num = 0):
    
    resid_best_r = torch.inf
    resid_best_g = torch.inf
    resid_best_b = torch.inf

    x_init_best = torch.zeros_like(y_complex)
    # print("Running {} trials for HIO ({} iterations each)...".format(number_of_trials, HIO_iter_trials))
    for seed_num in range(number_of_trials):
        # print('seed_num = ', seed_num)
        x_init_i = HIO(y_complex, A_op_complex_foo, beta_HIO, HIO_iter_trials, support, seed_num = int(seed_num + (algo_trial_num*number_of_trials))) 
        
        error_resid = torch.abs(y_complex) - torch.abs(A_op_complex_foo.A(x_init_i))

        resid_i_r = torch.norm(error_resid[0,0,0,:,:])
        resid_i_g = torch.norm(error_resid[0,0,1,:,:])
        resid_i_b = torch.norm(error_resid[0,0,2,:,:])



        if resid_i_r<resid_best_r:
            resid_best_r=resid_i_r
            x_init_best[0,0,0,:,:]=x_init_i[0,0,0,:,:]
        if resid_i_g<resid_best_g:
            resid_best_g=resid_i_g
            x_init_best[0,0,1,:,:]=x_init_i[0,0,1,:,:]
        if resid_i_b<resid_best_b:
            resid_best_b=resid_i_b
            x_init_best[0,0,2,:,:]=x_init_i[0,0,2,:,:]

    x_hat_HIO_foo = HIO(y_complex, A_op_complex_foo, beta_HIO, HIO_iter_final, support, x_init = x_init_best)


    x_hat_HIO = torch.real((x_hat_HIO_foo[0,0,:,pad_left:pad_left+resize_size,pad_left:pad_left+resize_size]))


    x_hat_HIO_new_unflipped = torch.zeros(1,2,3,256,256)
    x_hat_HIO_new_unflipped[0,0,:,:,:] = x_hat_HIO


    return x_hat_HIO_new_unflipped.to(y_complex.device)


def fix_channel_orientation_using_correlation(x_rec_HIO_best):
    
    x_rec_HIO_best_orientation_corrected = torch.zeros_like(x_rec_HIO_best)

    HIO_r_channel = x_rec_HIO_best[0,0,0,:,:].cpu()
    HIO_g_channel = x_rec_HIO_best[0,0,1,:,:].cpu()
    HIO_b_channel = x_rec_HIO_best[0,0,2,:,:].cpu()

    rg_corr = signal.correlate2d(HIO_r_channel.numpy(), HIO_g_channel.numpy(), mode='valid')
    rg_corr_flip = signal.correlate2d(HIO_r_channel.numpy(), HIO_g_channel.flip(0).flip(1).numpy(), mode='valid')

    if np.abs(rg_corr_flip) > np.abs(rg_corr):
        HIO_g_channel = HIO_g_channel.flip(0).flip(1)

    rb_corr = signal.correlate2d(HIO_r_channel.numpy(), HIO_b_channel.numpy(), mode='valid')
    rb_corr_flip = signal.correlate2d(HIO_r_channel.numpy(), HIO_b_channel.flip(0).flip(1).numpy(), mode='valid')

    if np.abs(rb_corr_flip) > np.abs(rb_corr):
        HIO_b_channel = HIO_b_channel.flip(0).flip(1)
    
    x_rec_HIO_best_orientation_corrected[0,0,0,:,:] = HIO_r_channel
    x_rec_HIO_best_orientation_corrected[0,0,1,:,:] = HIO_g_channel
    x_rec_HIO_best_orientation_corrected[0,0,2,:,:] = HIO_b_channel

    return x_rec_HIO_best_orientation_corrected

    

def fix_sign_ambiguity(inp_image, ref_avg_img):
    
    img_1 = inp_image[0,0,:,:,:].cpu()
    img_2 = -1*inp_image[0,0,:,:,:].cpu()
    img_3 = inp_image[0,0,:,:,:].flip(-2).flip(-1).cpu()
    img_4 = -1*inp_image[0,0,:,:,:].flip(-2).flip(-1).cpu()

    ref_avg_img = ref_avg_img[0,:,:,:].cpu() #torch.load('my_avg_image.pt')[0,:,:,:]
    img_1_corr = 0
    img_2_corr = 0
    img_3_corr = 0
    img_4_corr = 0

    for i in range(3):
        img_1_corr+=signal.correlate2d(img_1[i,:,:].numpy(), ref_avg_img[i,:,:].numpy(), mode='valid')
        img_2_corr+=signal.correlate2d(img_2[i,:,:].numpy(), ref_avg_img[i,:,:].numpy(), mode='valid')
        img_3_corr+=signal.correlate2d(img_3[i,:,:].numpy(), ref_avg_img[i,:,:].numpy(), mode='valid')
        img_4_corr+=signal.correlate2d(img_4[i,:,:].numpy(), ref_avg_img[i,:,:].numpy(), mode='valid')

    correlation_values = np.array([img_1_corr.max(), img_2_corr.max(), img_3_corr.max(), img_4_corr.max()])
    max_index = np.argmax(correlation_values)

    if max_index == 0:
        final_HIO_image = inp_image
    elif max_index == 1:
        final_HIO_image = -1*inp_image
    elif max_index == 2:
        final_HIO_image = inp_image.flip(-2).flip(-1)
    else:
        final_HIO_image = -1*inp_image.flip(-2).flip(-1)

    return final_HIO_image

def fix_flip_ambiguity(inp_image, ref_avg_img):
    
    img_1 = inp_image[0,0,:,:,:].cpu()
    img_3 = inp_image[0,0,:,:,:].flip(-2).flip(-1).cpu()

    ref_avg_img = ref_avg_img[0,:,:,:].cpu() #torch.load('my_avg_image.pt')[0,:,:,:]
    img_1_corr = 0
    img_3_corr = 0

    for i in range(3):
        img_1_corr+=signal.correlate2d(img_1[i,:,:].numpy(), ref_avg_img[i,:,:].numpy(), mode='valid')
        img_3_corr+=signal.correlate2d(img_3[i,:,:].numpy(), ref_avg_img[i,:,:].numpy(), mode='valid')

    correlation_values = np.array([img_1_corr.max(), img_3_corr.max()])
    max_index = np.argmax(correlation_values)

    if max_index == 0:
        final_HIO_image = inp_image
    else:
        final_HIO_image = inp_image.flip(-2).flip(-1)

    return final_HIO_image




def correct_2d_shift_and_flip(image_3chan, shifted_image_3chan):
    

    fixed_image_3chan = np.zeros_like(image_3chan)

    for chan in range(3):
        
        image = image_3chan[chan,:,:]
        shifted_image = shifted_image_3chan[chan,:,:]
        best_mse = float('inf')
        fixed_image = 1*shifted_image

        for my_flip in [0,1]:

            if my_flip == 1:
                shifted_image_foo = np.flip(1*shifted_image, axis=(0, 1))
            else:
                shifted_image_foo = 1*shifted_image

            # Step 1: Compute the FFT of both images
            fft_image = fft2(image)
            fft_shifted_image = fft2(shifted_image_foo)
            
            # Step 2: Compute the cross-power spectrum
            cross_power_spectrum = fft_image * np.conj(fft_shifted_image)
            
            # Normalize to prevent scaling differences
            cross_power_spectrum /= np.abs(cross_power_spectrum)
            
            # Step 3: Compute the inverse FFT of the cross-power spectrum to get the cross-correlation
            cross_correlation = ifft2(cross_power_spectrum)
            
            # Step 4: Shift the zero-frequency component to the center
            cross_correlation_shifted = fftshift(np.abs(cross_correlation))
            
            # Step 5: Find the location of the peak in the shifted cross-correlation
            max_index = np.unravel_index(np.argmax(cross_correlation_shifted), cross_correlation_shifted.shape)
            
            # Step 6: Calculate the shifts in both directions
            shifts = np.array(max_index)
            
            # Since the cross-correlation is shifted, we need to adjust the result by subtracting half the image size
            shift_x, shift_y = shifts - np.array(image.shape) // 2
            
            # fixed_image_foo = torch.roll(shifted_image_foo, shifts=(shift_x, shift_y), dims=(0, 1)) 
            fixed_image_foo = np.roll(shifted_image_foo, shift=(shift_x, shift_y), axis=(0, 1))

            mse = np.mean((image - fixed_image_foo)**2)

            if mse < best_mse:
                best_mse = mse
                fixed_image = 1*fixed_image_foo

        fixed_image_3chan[chan,:,:] = fixed_image

    return fixed_image_3chan