import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from torch.autograd import Variable

from misc import params
from misc.utils import get_data_loader, init_model, init_random_seed
from models import HNet

from datasets.TuSimple import get_tuSimple

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=50, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning-rate', default=0.000001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int, metavar='N', help='print frequency (default: 10)')



class MaxLossFunc(nn.Module):
    def __init__(self):
        super(MaxLossFunc, self).__init__()
    def forward(self, pred_y, gt_y):

        pred_value, pred_index = torch.max(pred_y,1)
        notEq_ = torch.zeros(pred_y.size())
        max_values = torch.ones(pred_y.size())
        max_value = torch.max(pred_y)
        max_val = torch.mul(max_values, -1*np.asscalar((max_value.data.cpu().numpy())))
        gt_y = gt_y.view(-1,1)
        gt_y = gt_y.data.cpu()
        mask = torch.gt(torch.zeros(pred_y.size()).scatter_(1,gt_y,1.0),notEq_).cuda()
        c = torch.masked_select(pred_y, Variable(mask))
        d = torch.masked_select(max_val.cuda(), mask)
        loss = torch.div(torch.sum(torch.pow(torch.div(torch.add(c,Variable(d)),100),2)), 1280)
        return loss





class ParamLossFunc(nn.Module):
    def __init__(self):
        super(ParamLossFunc, self).__init__()

        
    def forward(self, trans_param, gt_):   
        
        trans_gt = gt_.data       
        #print 'gt_: ', gt_
        count = 0
        loss = 0
        h  =  20
        gt =Variable(torch.Tensor([[0],[0],[0]]), requires_grad=True ).cuda()
        #trans_ = Variable( torch.Tensor([[0],[0],[0]]), requires_grad=True)
        back = Variable(torch.Tensor([[0],[0]]), requires_grad=True).cuda()
        #print 'trans_gt.size: ', trans_gt.size()
        for i in range(trans_gt.size()[0]):
            #print trans_param
            #trans_param[i,6].data = torch.zeros(1).cuda()
            #trans_param[i,7].data = torch.ones(1).cuda()
            #print trans_param
            fx = 160 * trans_param[i,0]
            fy = 120 * trans_param[i,1]
            c1 = torch.cos(trans_param[i,2])
            
            c2 = torch.ones(1).cuda()
            cx = 160 * trans_param[i,3]
            cy = 120 * trans_param[i,4]
            s1 = torch.sin(trans_param[i,2])
            
            s2 = torch.zeros(1).cuda()   
            zeros_ = torch.zeros(1).cuda()
            ones_ = torch.ones(1).cuda()            
            #H = torch.cat(((-h*c2/fx).view(1,1), (h*s1*s2/fy).view(1,1)))
            #print H
            
            
            #print fx, fy, c1, cx, cy, s1
            
            
            H = torch.cat(((-h*c2/fx).view(1,1), (h*s1*s2/fy).view(1,1), (h*c2*cx/fx-h*s1*s2*cy/fy-h*c1*s2).view(1,1), (h*s2/fx).view(1,1), (h*s1*c2/fy).view(1,1), (-h*s2*cx/fx-h*s1*c2*cy/fy-h*c1*c2).view(1,1), zeros_.view(1,1), (h*c1/fy).view(1,1), (-h*c1*cy/fy+h*s1).view(1,1), zeros_.view(1,1),  (-c1/fy).view(1,1), (c1*cy/fy-s1).view(1,1) ))

            H_1 = torch.cat(((fx*c2+c1*s2*cx).view(1,1), (-fx*s2+c1*c2*cx).view(1,1), (-s1*cx).view(1,1), (s2*(-fy*s1+c1*cy)).view(1,1), (c2*(-fy*s1+c1*cy)).view(1,1), (-fy*c1-s1*cy).view(1,1), (c1*s2).view(1,1), (c1*c2).view(1,1), -s1.view(1,1)))
            
            
            for j in range(4):
                
                each_points = gt_[i,j]
                #print torch.sum(each_points)
                if torch.sum(each_points.data) == 0:
                   continue

                each_points = each_points[each_points > 0]
                points_trans_gt = Variable(torch.ones([3,each_points.cpu().size()[0]/2]).cuda())
                #ones = np.ones([1,points_y.shape[0]], dtype = np.float32)
                points_trans_gt[0:2, :] = torch.t(each_points.view (-1,2))
                #points_trans_gt = Variable(points_trans_gt)
                transed_point = torch.mm(H.view(4,-1), points_trans_gt)
                
                ver = torch.reciprocal(transed_point[3,:])
                x = transed_point[0,:]*ver
                y = transed_point[1,:]*ver
                
                
                #print transed_point
                Y = torch.ones([y.size()[0] ,3]).cuda()
                
                Y[:, 0] = torch.pow(y,2)
                Y[:, 1] = y

                X = x
                
                a1 = torch.inverse(torch.mm(torch.t(Y), Y) + torch.Tensor([[0.001, 0 , 0], [0, 0.001, 0], [0, 0, 0.001]]).cuda())
                a2  = torch.mm(a1, torch.t(Y))
                W = torch.mm(a2 , X.view(-1,1))
                

                X_ = W[0,0]*y*y + W[1,0]*y + W[2,0]                                  # f(y)=ay2+by+c
                
                
                transed_point_pred = Variable(torch.ones([3, y.size()[0]]).cuda())
                transed_point_pred[0,:] = X_
                transed_point_pred[1,:] = y
                transed_point_pred[2,:] = -h*transed_point_pred[2,:]
                
                #transed_point_pred = np.vstack((X_, y, -h*np.ones([1, y.shape[0]]))
                
                transed_point_back = torch.mm(H_1.view(3,-1) , Variable(transed_point_pred))
                
                back0 = transed_point_back[0,:]* torch.reciprocal(transed_point_back[2,:])
                back1 = transed_point_back[1,:]* torch.reciprocal(transed_point_back[2,:])
                
                
                back_ = torch.cat((back0.view(1,-1),back1.view(1,-1)),0)                 
                gt = torch.cat([gt, points_trans_gt], 1)
                #trans_.data = torch.cat([trans_.data, transed_point_pred],1)
                back = torch.cat([back, back_],1)

        loss = torch.mean(torch.pow((back[0:2, 1::] - gt[0:2,1::]), 2))
        #loss = torch.mean(torch.abs(back[0:2, 1::] - gt[0:2,1::]))
        return loss
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(trainloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        #loss = criterion(output, target_var)  + criterion1(output, target_var)
        loss = criterion(output, target_var)
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print 'losses: ', loss.item()
            print 'Epoch: ', epoch
            if epoch %10 == 0:
                save_checkpoint(model.state_dict())

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)
    args = parser.parse_args()
    
    hnet = init_model(net=HNet.HNet(), restore=None)
    print("=== Training models ===")
    print(">>> hnet <<<")                            
    print(hnet)                       
    
    model = nn.DataParallel(hnet).cuda()  

    criterion = ParamLossFunc()
  
    optimizer = optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99))
    cudnn.benchmark = True


    
    src_data_loader = get_tuSimple('train')
    print src_data_loader
    
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(src_data_loader, model, criterion, optimizer, epoch)    

    # evaluate models
    #print("=== Evaluating models ===")
    #print(">>> on source domain <<<")
    #test(classifier, generator, src_data_loader, params.src_dataset)
    #print(">>> on target domain <<<")
    #test(classifier, generator, tgt_data_loader, params.tgt_dataset)
