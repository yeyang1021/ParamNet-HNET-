import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable

#from core import test, train
from misc import params
from misc.utils import get_data_loader, init_model, init_random_seed
from models import HNet


from datasets.TuSimple import get_tuSimple
torch.load('./checkpoint.pth.tar')


init_random_seed(params.manual_seed)
src_data_loader = get_tuSimple('train')
print src_data_loader
hnet = init_model(net=HNet.HNet(), restore=None)
model = nn.DataParallel(hnet).cuda() 
for i, (input, target) in enumerate(src_data_loader):
    input, target = input.cuda(), target.cuda()
    input_var = Variable(input)
    output = model(input_var)
    print output
    
