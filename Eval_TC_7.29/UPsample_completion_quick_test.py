# Completion of a 256*256*3 image by TRLRF
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from TR_functions import gen_W, RSE_fun
import TRLRF
from utils import str2bool, get_accuracy, get_image_classifier, load_data






class arguments():
    def __init__(self, exp, config,decomposition_method,SNR,data_seed,torch_seed,classifier_name,verbose,image_folder,attack_version,attack_type,lp_norm,adv_batch_size,domain,
                 num_sub,adv_eps):
        self.exp = exp
        self.config = config
        self.decomposition_method = decomposition_method
        self.SNR = SNR
        self.data_seed = data_seed
        self.torch_seed = torch_seed
        self.classifier_name = classifier_name
        self.verbose = verbose
        self.image_folder = image_folder
        self.attack_version = attack_version
        self.attack_type = attack_type
        self.lp_norm = lp_norm
        self.adv_batch_size = adv_batch_size
        self.domain = domain
        self.num_sub = num_sub
        self.adv_eps = adv_eps
exp = "./exp_results"
config = "cifar10.yml"
decomposition_method = "TRLRF"
SNR = 0.5
data_seed = 54
torch_seed = 54
classifier_name = "cifar10-wideresnet-28-10"
verbose = 'info'
image_folder = 'cifar10-robust_adv'
attack_version = "standard"
lp_norm = 'Linf'
adv_batch_size = 64
domain = "cifar10"
num_sub = 1000
adv_eps = 0.07
attack_type = 'PGD'        
args = arguments(exp, config,decomposition_method,SNR,data_seed,torch_seed,classifier_name,verbose,image_folder,attack_version,attack_type,lp_norm,adv_batch_size,domain,
                 num_sub,adv_eps)


device = "cuda"
mr = 0.2  # missing rate
r = 5 * np.ones(3, dtype='int')  # TR-rank
maxiter = 300  # maxiter 300~500
tol = 1e-6  # 1e-6~1e-8
Lambda = 5  # usually 1~10
ro = 1.1  # 1~1.5
K = 1e0  # 1e-1~1e0

x_val, y_val = load_data(args, 64,0)
mask = []
for x in x_val:
    W = gen_W(x.shape, mr)
    W = torch.from_numpy(W)
    W = W.to(device)
    mask.append(W)
mask = torch.stack(mask)

x_val_256 = F.interpolate(x_val, size=(256, 256), mode='bilinear', align_corners=False)
mask_256 = F.interpolate(mask, size=(256, 256), mode='bilinear', align_corners=False)

for image_index in range(10):
    print(image_index)
    
    
    img_o = x_val[image_index]
    img_o = img_o.clone().detach().cpu().numpy()
    img_o_256 = x_val_256[image_index]
    img_o_256 = img_o_256.clone().detach().cpu().numpy()
    W = mask[image_index]
    W = W.clone().detach().cpu().numpy()
    W_256 = mask_256[image_index]
    W_256 = W_256.clone().detach().cpu().numpy()
    
    img_o = np.transpose(img_o, (1,2,0))
    img_o_256 = np.transpose(img_o_256, (1,2,0))
    W = np.transpose(W, (1,2,0))
    W_256 = np.transpose(W_256, (1,2,0))
    
    image_ori = np.array(255 *img_o, dtype='uint8')
    image_ori_256 = np.array(255 *img_o_256, dtype='uint8')
    image_noise = np.array( W * image_ori, dtype='uint8')
    image_noise_256 = np.array( W_256 * image_ori_256, dtype='uint8')
    
    
    
    
    image_temp_256, _, __ = TRLRF.TRLRF(img_o_256, W_256, r, maxiter, K, ro, Lambda, tol)
    image_temp, _, __ = TRLRF.TRLRF(img_o, W, r, maxiter, K, ro, Lambda, tol)
    
    image_hat = np.array(image_temp*255, dtype='uint8')
    image_hat_256 = np.array(image_temp_256*255, dtype='uint8')
    
    

    
    plt.figure()
    plt.subplot(2,4,1)
    plt.imshow(image_ori)
    plt.title('Original')
    

    plt.subplot(2,4,2)
    plt.imshow(image_noise)
    plt.title("%s %% MISSING" % str(mr*100))




    plt.subplot(2,4,3)
    #plt.imshow(np.array(image_hat, dtype='double'))
    plt.imshow(image_hat)
    plt.title('Completion')

    plt.subplot(2,4,4)
    plt.imshow(W)
    plt.title('debug')
    
    

    plt.subplot(2,4,5)
    plt.imshow(image_ori_256)
    plt.title('Original_256')
    

    plt.subplot(2,4,6)
    plt.imshow(image_noise_256)
    plt.title("%s %% MISSING_256" % str(mr*100))




    plt.subplot(2,4,7)
    #plt.imshow(np.array(image_hat, dtype='double'))
    plt.imshow(image_hat_256)
    plt.title('Completion_256')

    plt.subplot(2,4,8)
    plt.imshow(W_256)
    plt.title('debug_256')
    
    
    
    
    
    
    
    
    name = 'image' + str(image_index)
    plt.savefig(name)
    '''
    image_temp = torch.from_numpy(image_temp)
    image_temp = image_temp.requires_grad_(True)
    image_temp = image_temp.to(device)
    '''
    



'''
W = gen_W(X.shape, mr)
X_hat, _, __ = TRLRF.TRLRF(X, W, r, maxiter, K, ro, Lambda, tol)














# Evaluation
RSE = RSE_fun(X, X_hat, W)

print('Completion RSE is %s' % RSE[0])

image_o = np.array(image, dtype='uint8')
image_noise = np.array(255 * W * X, dtype='uint8')
image_hat = np.array(X_hat*255, dtype='uint8')
# print(image_hat.shape)


plt.figure()
plt.subplot(141)
plt.imshow(im)
plt.title('Original')

plt.subplot(142)
plt.imshow(image_noise)
plt.title("%s %% MISSING" % str(mr*100))

plt.subplot(143)
#plt.imshow(np.array(image_hat, dtype='double'))
plt.imshow(image_hat)
plt.title('Completion')

plt.subplot(144)
plt.imshow(W)
plt.title('debug')

plt.show()
'''