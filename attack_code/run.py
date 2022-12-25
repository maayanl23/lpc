from __future__ import print_function
import argparse
# import pickle

import numpy as np
import torch
import torch.optim as optim
from gcommand_loader import GCommandLoader
from model import LeNet, VGG
from train import train, test
from attacks import attack

import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pynvml import *
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total / 10 ** 9}')
print(f'free     : {info.free / 10 ** 9}')
print(f'used     : {info.used / 10 ** 9}')
class args():

    def _init_(self, name):
        pass
    def cuda(self,x):
        pass         
    def train_path(self,x):
        pass
    def test_path(self,x):
        pass
    def test_batch_size(self,x):
        pass
    def arc(self,x):
        pass
    def epochs(self,x):
        pass
    def lr(self,x):
        pass
    def momentum(self,x):
        pass
    def optimizer(self,x):
        pass    
    def log_interval(self,x):
        pass
    def seed(self,x):
        pass
    def patience(self,x):
        pass
    def window_size(self,x):
        pass
    def window_stride(self,x):
        pass
    def window_type(self,x):
        pass    
    def normalize(self,x):
        pass
    def chkpt_path(self,x):
        pass
    def chkpt_path_train(self,x):
        pass
    def n_iter(self,x):
        pass
    def eps(self,x):
        pass    
    def alpha(self,x):
        pass
    def test_mode(self,x):
        pass
args.train_path='/home/maayanl/data/gcommands/train'
args.test_path='/home/maayanl/data/gcommands/test'
args.valid_path='/home/maayanl/data/gcommands/valid'
args.batch_size=25
args.test_batch_size=25
args.arc='VGG11'#'network architecture: LeNet, VGG11, VGG13, VGG16, VGG19'
args.epochs=2#TODO:default=100
args.lr=0.001
args.momentum=0.9#help='SGD momentum, for SGD only'
args.optimizer='adam'#' help=optimization method: sgd | adam'
args.cuda=True
args.log_interval=10
args.cuda = args.cuda and torch.cuda.is_available()
args.seed="1234"
args.patience=5
args.window_size=0.2
args.window_stride=0.1
args.window_type='hamming'
args.normalize=True

args.test_mode=False
#parser.add_argument('--test_mode', action="store_true", help='Whether to run model for test only or not')
args.chkpt_path=""
args.chkpt_path_train=""
args.n_iter=10
args.eps=0.01
args.alpha=0.01
'''# Training settings

parser = argparse.ArgumentParser(
    description='ConvNets for Speech Commands Recognition')
parser.add_argument('--train_path', default='/home/maayanl/data/gcommands/train',
                    help='path to the train data folder')
parser.add_argument('--test_path', default='/home/maayanl/data/gcommands/test',
                    help='path to the test data folder')
parser.add_argument('--valid_path', default='/home/maayanl/data/gcommands/valid',
                    help='path to the valid data folder')
parser.add_argument('--batch_size', type=int, default=80,
                    metavar='N', help='training and valid batch size')
parser.add_argument('--test_batch_size', type=int, default=100,
                    metavar='N', help='batch size for testing')
parser.add_argument('--arc', default='VGG11',
                    help='network architecture: LeNet, VGG11, VGG13, VGG16, VGG19')
parser.add_argument('--epochs', type=int, default=2,
                    metavar='N', help='number of epochs to train')  # TODO:default=100
parser.add_argument('--lr', type=float, default=0.001,
                    metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    metavar='M', help='SGD momentum, for SGD only')
parser.add_argument('--optimizer', default='adam',
                    help='optimization method: sgd | adam')
parser.add_argument('--cuda', default=True, help='enable CUDA')
parser.add_argument('--seed', type=int, default=1234,
                    metavar='S', help='random seed')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='num of batches to wait until logging train status')
parser.add_argument('--patience', type=int, default=5, metavar='N',
                    help='how many epochs of no loss improvement should we wait before stop training')

# feature extraction options
parser.add_argument('--window_size', default=.02,
                    help='window size for the stft')
parser.add_argument('--window_stride', default=.01,
                    help='window stride for the stft')
parser.add_argument('--window_type', default='hamming',
                    help='window type for the stft')
parser.add_argument('--normalize', default=True,
                    help='boolean, whether or not to normalize the spect')


parser.add_argument('--test_mode', action="store_true", help='Whether to run model for test only or not')
parser.add_argument('--chkpt_path', default="", help='checkpoint path to load')
parser.add_argument('--chkpt_path_train', default="", help='checkpoint path name to save')
parser.add_argument('--n_iter', type=int, default=10, metavar='N',
                    help='')
parser.add_argument('--eps', type=float, default=0.01, metavar='N',
                    help='')
parser.add_argument('--alpha', type=float, default=0.01, metavar='N',
                    help='')
args = parser.parse_args()'''

args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# build model
if args.arc == 'LeNet':
    model = LeNet()
elif args.arc.startswith('VGG'):
    model = VGG(args.arc)
else:
    model = LeNet()

if args.cuda:
    model = model.cuda()

# loading data
train_dataset = GCommandLoader(args.train_path, window_size=args.window_size, window_stride=args.window_stride,
                             window_type=args.window_type, normalize=args.normalize)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=8, pin_memory=args.cuda, sampler=None)

valid_dataset = GCommandLoader(args.valid_path, window_size=args.window_size, window_stride=args.window_stride,
                             window_type=args.window_type, normalize=args.normalize)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=args.batch_size, shuffle=None,
    num_workers=8, pin_memory=args.cuda, sampler=None)

test_dataset = GCommandLoader(args.test_path, window_size=args.window_size, window_stride=args.window_stride,
                            window_type=args.window_type, normalize=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=None,
    num_workers=8, pin_memory=args.cuda, sampler=None)



# define optimizer
if args.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

best_valid_loss = np.inf
iteration = 0
epoch = 1
nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total / 10 ** 9}')
print(f'free     : {info.free / 10 ** 9}')
print(f'used     : {info.used / 10 ** 9}')
if args.test_mode:
    checkpoint = torch.load(args.chkpt_path)
    model.load_state_dict(checkpoint['net'].state_dict())
    model.eval()

# train int with early stopping
while (epoch < args.epochs + 1) and (iteration < args.patience) and not args.test_mode:
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'epoch   : {epoch}')
    print(f'total    : {info.total  /10** 9}')
    print(f'free     : {info.free / 10 **9}')
    print(f'used     : {info.used / 10 ** 9}')
    train(train_loader, model, optimizer, epoch, args.cuda, args.log_interval)
    valid_loss = test(valid_loader, model, args.cuda)
    if valid_loss > best_valid_loss:
        if epoch > 11:
           for param_group in optimizer.param_groups:
              param_group['lr'] = args.lr*0.2
        iteration += 1
        print('Loss was not improved, iteration {0}'.format(str(iteration)))
    else:
        print('Saving model...')
        iteration = 0
        best_valid_loss = valid_loss
        state = {
            'net': model,
            'acc': valid_loss,
            'epoch': epoch,
            'optimizer_state_dict:': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/{args.chkpt_path_train}.t7')
    epoch += 1


test(test_loader, model, args.cuda)
attack(test_loader, 0, model, n_iter=10,eps=args.eps, alpha =args.alpha, rand_init=True)