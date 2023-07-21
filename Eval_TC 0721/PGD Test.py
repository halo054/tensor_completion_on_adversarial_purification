import sys
from PIL import Image
import matplotlib.pyplot as plt
from TR_functions import gen_W, RSE_fun
import TRLRF
import torch
import torch.nn as nn
import utils
from utils import str2bool, get_accuracy, get_image_classifier, load_data
import numpy as np
import torchattacks
import matplotlib.pyplot as plt
import robustbench
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
import torch.nn.functional as F

#images, labels = load_cifar10(n_examples=5)
print('[Data loaded]')



from torchattacks import PGD

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
resnet = get_image_classifier(args.classifier_name).to(device)
mr = 0.5  # missing rate
r = 5 * np.ones(3, dtype='int')  # TR-rank
maxiter = 300  # maxiter 300~500
tol = 1e-6  # 1e-6~1e-8
Lambda = 5  # usually 1~10
ro = 1.1  # 1~1.5
K = 1e0  # 1e-1~1e0

x_val, y_val = load_data(args, 64,0)

atk = PGD(resnet, eps=8/255, alpha=2/225, steps=10, random_start=True)



init_acc = get_accuracy(resnet, x_val[0:64,:,:,:], y_val[0:64], bs=adv_batch_size)

print("initial accuracy:",init_acc)




adv_images = atk(x_val, y_val)
robust_acc = get_accuracy(resnet, adv_images[0:64,:,:,:], y_val[0:64], bs=adv_batch_size)

print("robust accuracy:",robust_acc)


TC_tag = y_val[0:64]

TC_image = []
      
x_val_256 = F.interpolate(x_val, size=(256, 256), mode='bilinear', align_corners=False)
for image_index in range(64):
    print(image_index)
    adv_temp = x_val_256[image_index]
    adv_temp = adv_temp.clone().detach().cpu().numpy()
    #image_temp = np.transpose(image_temp, (1,2,0))
          
    W = gen_W(adv_temp.shape, mr)
    image_temp, _, __ = TRLRF.TRLRF(adv_temp, W, r, maxiter, K, ro, Lambda, tol)
    image_temp = torch.from_numpy(image_temp)
    image_temp = image_temp.requires_grad_(True)
    image_temp = image_temp.to(device)

    TC_image.append(image_temp)
            
TC_image = torch.stack(TC_image)
TC_image =(TC_image + 1) * 0.5
TC_tag = y_val[0:64]
TC_image = TC_image.type(torch.FloatTensor).to(device)    
TC_image = F.interpolate(TC_image, size=(32, 32), mode='bilinear', align_corners=False)

print("image:",TC_image.shape)
print("tag:",TC_tag.shape)
TC_acc = get_accuracy(resnet, TC_image, TC_tag, bs=adv_batch_size)
print("TC_acc:",TC_acc)

      
robust_TC_image = []
      
adv_images_256 = F.interpolate(adv_images, size=(256, 256), mode='bilinear', align_corners=False)
for image_index in range(64):
    print(image_index)
    adv_temp = adv_images_256[image_index]
    adv_temp = adv_temp.clone().detach().cpu().numpy()
    #image_temp = np.transpose(image_temp, (1,2,0))
          
    W = gen_W(adv_temp.shape, mr)
    image_temp, _, __ = TRLRF.TRLRF(adv_temp, W, r, maxiter, K, ro, Lambda, tol)
    image_temp = torch.from_numpy(image_temp)
    image_temp = image_temp.requires_grad_(True)
    image_temp = image_temp.to(device)

    robust_TC_image.append(image_temp)
            
robust_TC_image = torch.stack(robust_TC_image)
robust_TC_image =(robust_TC_image + 1) * 0.5

robust_TC_image = robust_TC_image.type(torch.FloatTensor).to(device)    
robust_TC_image = F.interpolate(robust_TC_image, size=(32, 32), mode='bilinear', align_corners=False)

print("image:",robust_TC_image.shape)
print("tag:",TC_tag.shape)
robust_TC_acc = get_accuracy(resnet, robust_TC_image, TC_tag, bs=adv_batch_size)
print("robust_TC_acc:",robust_TC_acc)