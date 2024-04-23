#  ----------------Import--------------------------
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.hub import load_state_dict_from_url
from torch.utils.data.dataset import Subset,ConcatDataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
#  ----------------Import Data Loader--------------------------
import utils

try:
    from apex.parallel import DistributedDataParallel 
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    print("Please install apex from https://www.github.com/nvidia/apex to run this example.")

#  ----------------Import Models--------------------------
from models import *
# from models import DNS
# from models import DNS_att2
# from models import DnCNN
# from models import DnCNN_CA
# from models import DnCNN_Condition
# from models import DnCNN_Subnet
# from models import DNS_wVGG
# from models import DnCNN_Subnet_residual
# from models import OWAN
# from models import DNS_nosub
# from models import model_del_attn
#  ----------------Import End--------------------------

# debug 

def debug(hyper_params):
    Experiment_Confs = hyper_params
    Model_conf =  hyper_params.model
    Data_conf =  hyper_params.dataset
    hyper_params = hyper_params.experiment

    gpuID = hyper_params['device_ids'][0]
    torch.cuda.set_device(gpuID)
    checkpoint_resume = hyper_params['checkpoint_resume']
    checkpoint_PATH = hyper_params['checkpoint_PATH']
    opt_level = hyper_params['opt_level']
    epoch_num = hyper_params['epoch_num']
    device_ids = hyper_params['device_ids']
    trained_model = hyper_params['best_model']
    lr = hyper_params['lr']
    scheduler = hyper_params['scheduler']

    handle_test_size = False
    test_mod_size = 1
    Seg_Train = False
    handle_test_size = hyper_params['handle_test_size']
    test_mod_size = hyper_params['test_mod_size']
    torch.cuda.empty_cache()
    print('GPUID    :', gpuID)
    print('epoch_num:', epoch_num)

    best_psnr=0
    best_model = None
    # model
    np.random.seed(seed=hyper_params['seed'])  # for reproducibility
    torch.manual_seed(hyper_params['seed'])
    torch.cuda.manual_seed(hyper_params['seed'])
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
#----------------------------------------------------------------------------------------------------------------
#-----------------------------------setup model--------------------------------------
#----------------------------------------------------------------------------------------------------------------
    model = eval(Model_conf.name)(Model_conf)
    model = model.cuda(gpuID)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    if opt_level=="O1" or opt_level=="O2":
        model, optimizer = amp.initialize(model, optimizer,
                                            opt_level=opt_level
                                            )
    batch=hyper_params['batchsize']//len(hyper_params['device_ids'])
    train_data = torch.randn(batch, hyper_params['color'], hyper_params['imgSize'], hyper_params['imgSize']).float().cuda(gpuID)
    true_data = torch.randn(batch, hyper_params['color'], hyper_params['imgSize'], hyper_params['imgSize']).float().cuda(gpuID)
    test_data = torch.randn(1, hyper_params['color'],1280, 1280).float().cuda(gpuID)
    criterion = nn.MSELoss().cuda(gpuID)
    print('Param:', utils.count_parameters_in_MB(model))

    print('debug start')
    print('train_data',train_data.shape)
    loss = criterion(model(train_data), true_data)
    if opt_level=="O1" or opt_level=="O2":
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    print('train_data debug is finished')
    model = eval(Model_conf.name)(Model_conf)

    model = model.cuda(gpuID)
    
    model.eval()
    with torch.no_grad():

        print('test_data',test_data.shape)
        model(test_data)
        print('test_data debug is finished')