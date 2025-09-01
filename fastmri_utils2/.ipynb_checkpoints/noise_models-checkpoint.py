import torch

# Need to fix the device issue
# def get_noisy_data_with_SD_map_complex(data, noise_std = float(25)/255.0, mode='S', 
#                     min_noise = float(5)/255., max_noise = float(55)/255.):
    
#     print(data.shape)

#     noise = torch.randn_like(data);
    
#     result_data = torch.zeros(data.shape[0],3,data.shape[2],data.shape[3]).to(device);
# #     print('hi')
#     if mode == 'B':
#         n = noise.shape[0];
#         noise_tensor_array = (1/torch.sqrt(torch.tensor(2)))*((max_noise - min_noise) * torch.rand(n) + min_noise);
#         for i in range(n):
#             noise.data[i] = noise.data[i] * noise_tensor_array[i];
#             result_data[i,0:2,:,:] = data[i,0:2,:,:] + noise.data[i]
#             result_data[i,2,:,:] = noise_tensor_array[i]*torch.ones(data[i,0,:,:].shape)
#     else:
#         noise.data = (1/torch.sqrt(torch.tensor(2))) * noise.data * noise_std;
#         n = noise.shape[0];
#         for i in range(n):
#             result_data[i,0:2,:,:] = data[i,0:2,:,:] + noise.data[i]
#             result_data[i,2,:,:] = (1/torch.sqrt(torch.tensor(2)))*noise_std*torch.ones(data[i,0,:,:].shape)

#     return result_data


def get_noisy_data_complex(data, noise_std = float(25)/255.0, mode='S', 
                    min_noise = float(5)/255., max_noise = float(55)/255.):
    
    noise = torch.randn_like(data);
    
    result_data = torch.zeros_like(data);
#     print('hi')
    if mode == 'B':
        n = noise.shape[0];
        noise_tensor_array = (1/torch.sqrt(torch.tensor(2)))*((max_noise - min_noise) * torch.rand(n) + min_noise);
        for i in range(n):
            noise.data[i] = noise.data[i] * noise_tensor_array[i];
            result_data[i,:,:,:] = data[i,:,:,:] + noise.data[i]
    else:
        noise.data = (1/torch.sqrt(torch.tensor(2))) * noise.data * noise_std;
        n = noise.shape[0];
        for i in range(n):
            result_data[i,:,:,:] = data[i,:,:,:] + noise.data[i]
    return result_data



def add_noise_to_complex_measurements(y,wvar,idx1_complement,idx2_complement,device, seed, is_complex):

    torch.manual_seed(seed)
    
    if is_complex:
        my_std_new = (torch.sqrt(wvar.clone()))/torch.sqrt(torch.tensor(2))
    else:
        my_std_new = (torch.sqrt(wvar.clone()))
        
    noise = my_std_new*(torch.randn(y.shape, device = device))
    
    result = y.clone() + noise

    result[:,:,idx1_complement,idx2_complement] = 0
    
    ## Testing added noise

    noise[:,:,idx1_complement,idx2_complement] = 0
    
    pow_1 = torch.sum(y**2)
    pow_2 = torch.sum(noise**2)
    ratio_snr = torch.sqrt(pow_1)/torch.sqrt(pow_2)
    SNRdB_test = 20*torch.log10(ratio_snr)
    # print('SNR in dB for this run:')
    # print(SNRdB_test)
    
    ## Done Testing
    
    return result, ratio_snr



def add_noise_to_complex_measurements_no_seed(y,wvar,idx1_complement,idx2_complement,device, is_complex):

    # torch.manual_seed(seed)
    
    if is_complex:
        my_std_new = (torch.sqrt(wvar.clone()))/torch.sqrt(torch.tensor(2))
    else:
        my_std_new = (torch.sqrt(wvar.clone()))
        
    noise = my_std_new*(torch.randn(y.shape, device = device))
    
    result = y.clone() + noise

    result[:,:,idx1_complement,idx2_complement] = 0
    
    ## Testing added noise

    noise[:,:,idx1_complement,idx2_complement] = 0
    
    pow_1 = torch.sum(y**2)
    pow_2 = torch.sum(noise**2)
    ratio_snr = torch.sqrt(pow_1)/torch.sqrt(pow_2)
    SNRdB_test = 20*torch.log10(ratio_snr)
    # print('SNR in dB for this run:')
    # print(SNRdB_test)
    
    ## Done Testing
    
    return result, ratio_snr