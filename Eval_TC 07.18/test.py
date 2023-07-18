import argparse
import logging
import yaml
import os
import time

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoattack import AutoAttack
from stadv_eot.attacks import StAdvAttack

import utils
from utils import str2bool, get_accuracy, get_image_classifier, load_data

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from TR_functions import gen_W, RSE_fun
import TRLRF






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


class SDE_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args

        # image classifier
        self.classifier = get_image_classifier(args.classifier_name).to(config.device)

        # diffusion model
        print(f'Decomposition_method: {args.decomposition_method}')
#        if args.diffusion_type == 'ddpm':
#            self.runner = TRLRF
#        elif args.diffusion_type == 'sde':
#            self.runner = RevGuidedDiffusion(args, config, device=config.device)
        
#        else:
#            raise NotImplementedError('unknown diffusion type')

        # use `counter` to record the the sampling time every 5 NFEs (note we hardcoded print freq to 5,
        # and you may want to change the freq)
        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=config.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x):
        counter = self.counter.item()
        if counter % 5 == 0:
            print(f'diffusion times: {counter}')

        # imagenet [3, 224, 224] -> [3, 256, 256] -> [3, 224, 224]
        if 'imagenet' in self.args.domain:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        start_time = time.time()
        
        
        
        
        mr = 0.1  # missing rate
        r = 5 * np.ones(3, dtype='int')  # TR-rank
        maxiter = 300  # maxiter 300~500
        tol = 1e-6  # 1e-6~1e-8
        Lambda = 5  # usually 1~10
        ro = 1.1  # 1~1.5
        K = 1e0  # 1e-1~1e0
        
        
        x = x.cpu().detach().numpy()
        x_re = []
        length = np.shape(x)
        for image_index in range(length[0]):
            print(image_index)
            image_temp = x[image_index]

            

            image_temp = np.transpose(image_temp, (1,2,0))
            W = gen_W(image_temp.shape, mr)


            image_o = np.array(255 *image_temp, dtype='uint8')
            image_noise = np.array( W * image_o, dtype='uint8')




            image_temp, _, _ = TRLRF.TRLRF(image_temp, W, r, maxiter, K, ro, Lambda, tol)


            image_hat = np.array(image_temp*255, dtype='uint8')

            image_temp = np.transpose(image_temp, (2,0,1))


            plt.figure()
            plt.subplot(141)
            plt.imshow(image_o)
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
            name = 'image' + str(image_index)
            plt.savefig(name)
            
            


            image_temp = torch.from_numpy(image_temp)
#            image_temp = image_temp.requires_grad_(True)
            image_temp = image_temp.to(config.device)
            
            x_re.append(image_temp)
            
        x_re = torch.stack(x_re)
        x_re =(x_re + 1) * 0.5
    
        x_re = x_re.type(torch.FloatTensor).to(config.device)
        minutes, seconds = divmod(time.time() - start_time, 60)





        if 'imagenet' in self.args.domain:
            x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear', align_corners=False)

        if counter % 5 == 0:
            print(f'x shape (before diffusion models): {x.shape}')
            print(f'x shape (before classifier): {x_re.shape}')
            print("Sampling time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))
        
        
        
        out = self.classifier(x_re)
        

        self.counter += 1

        return out
    




def parse_args_and_config():


    '''


    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # diffusion models



    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde]')
    parser.add_argument('--eot_iter', type=int, default=20, help='only for rand version of autoattack')
    parser.add_argument('--use_bm', action='store_true', help='whether to use brownian motion')

    # LDSDE
    parser.add_argument('--sigma2', type=float, default=1e-3, help='LDSDE sigma2')
    parser.add_argument('--lambda_ld', type=float, default=1e-2, help='lambda_ld')
    parser.add_argument('--eta', type=float, default=5., help='LDSDE eta')
    parser.add_argument('--step_size', type=float, default=1e-3, help='step size for ODE Euler method')

    # adv
    parser.add_argument('--domain', type=str, default='celebahq', help='which domain: celebahq, cat, car, imagenet')
    parser.add_argument('--partition', type=str, default='val')
    parser.add_argument('--adv_batch_size', type=int, default=64)
    parser.add_argument('--attack_type', type=str, default='square')
    parser.add_argument('--lp_norm', type=str, default='Linf', choices=['Linf', 'L2'])
    parser.add_argument('--attack_version', type=str, default='custom')

    parser.add_argument('--num_sub', type=int, default=1000, help='imagenet subset')
    parser.add_argument('--adv_eps', type=float, default=0.07)

    args = parser.parse_args()
    '''





        

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
    
    #
    attack_type = 'square'
    
    
    args = arguments(exp, config, decomposition_method, SNR, data_seed, torch_seed, classifier_name, verbose, image_folder, attack_version, attack_type, lp_norm, adv_batch_size,
                     domain,num_sub,adv_eps)
    

    
    
    # parse config file
    with open(os.path.join('configs', config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = utils.dict2namespace(config)
    
    level = getattr(logging, verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    image_folder_path = os.path.join(exp, image_folder)
    os.makedirs(image_folder_path, exist_ok=True)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(torch_seed)
    random.seed(torch_seed)
    np.random.seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)

    torch.backends.cudnn.benchmark = True
    
    

    
    
    return args, new_config





args,config = parse_args_and_config()
model = SDE_Adv_Model(args, config)
x_val, y_val = load_data(args, 64,0)
result = model.forward(x_val[0:10,:,:,:])