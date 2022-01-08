#!/usr/bin/env python

"""
    bkj_pretrain.py
"""

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae
import util.lr_sched as lr_sched

from engine_pretrain import train_one_epoch

torch.backends.cudnn.benchmark = True

def to_numpy(x):
    return x.detach().cpu().numpy()

def parse_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=512,  type=int)
    parser.add_argument('--epochs',     default=4000, type=int)
    
    parser.add_argument('--model',         default='mae_vit_base_patch16', type=str, metavar='MODEL')
    parser.add_argument('--input_size',    default=32, type=int)
    parser.add_argument('--mask_ratio',    default=0.75, type=float)
    parser.add_argument('--norm_pix_loss', action='store_true')
    parser.set_defaults(norm_pix_loss=False)
    
    parser.add_argument('--weight_decay',  type=float, default=0.05)
    
    parser.add_argument('--lr',            type=float, default=2e-3)
    parser.add_argument('--min_lr',        type=float, default=0.)
    parser.add_argument('--warmup_epochs', type=int,   default=10)
    
    parser.add_argument('--output_dir', default='./output_dir/run1')
    parser.add_argument('--log_dir',    default='./output_dir/run1')
    parser.add_argument('--seed',       default=123, type=int)
    
    return parser.parse_args()

# --
# Init

args = parse_args()

assert args.output_dir is not None
os.makedirs(args.output_dir, exist_ok=True)

device = torch.device('cuda')

# fix the seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

os.makedirs(args.log_dir, exist_ok=True)
log_writer = SummaryWriter(log_dir=args.log_dir)

# --
# Data

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(args.input_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ds_train = datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_train
)
ds_valid = datasets.CIFAR10(
    root='./data', train=True, download=False, transform=transform_test
)
ds_test = datasets.CIFAR10(
    root='./data', train=False, download=False, transform=transform_test
)

dl_train = torch.utils.data.DataLoader(
    ds_train, 
    shuffle=True,
    batch_size=args.batch_size,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
)
dl_valid = torch.utils.data.DataLoader(
    ds_valid, 
    shuffle=False,
    batch_size=args.batch_size,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
)
dl_test = torch.utils.data.DataLoader(
    ds_test, 
    shuffle=False,
    batch_size=args.batch_size,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
)

# --
# Model

from functools import partial
from models_mae import MaskedAutoencoderViT
from torch import nn

model = MaskedAutoencoderViT(
    img_size=32,
    patch_size=8,
    embed_dim=512,
    depth=8,
    num_heads=8,
    decoder_embed_dim=256,
    decoder_depth=4,
    decoder_num_heads=8,
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6)
)

model = model.to(device)

# following timm: set wd as 0 for bias and norm layers
param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
opt          = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

# --
# Run

for epoch in range(args.epochs):
    # --
    # Train
    
    _ = model.train()
    gen = tqdm(dl_train)
    for batch_idx, (x, _) in enumerate(gen):
        
        lr = lr_sched.adjust_learning_rate(opt, batch_idx / len(dl_train) + epoch, args)
        
        x    = x.to(device, non_blocking=True)
        loss = model(x, mask_ratio=args.mask_ratio)
        
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        log_writer.add_scalar('loss/train', float(loss), epoch * len(dl_train) + batch_idx)
        log_writer.add_scalar('cfg/lr',   float(lr),   epoch * len(dl_train) + batch_idx)
        gen.set_postfix(loss=float(loss))
    
    # --
    # Eval
    
    if epoch % 10 == 0:
        with torch.inference_mode():
            _ = model.eval()
            X_valid, y_valid = [], []
            
            loss = 0
            for batch_idx, (x, y) in enumerate(tqdm(dl_valid)):        
                x   = x.to(device, non_blocking=True)
                fts = model.forward_encoder(x, mask_ratio=0)[0]
                
                X_valid.append(to_numpy(fts.mean(axis=1)))
                y_valid.append(to_numpy(y))
                
                loss += float(model(x, mask_ratio=args.mask_ratio))
            
            X_valid = normalize(np.row_stack(X_valid))
            y_valid = np.hstack(y_valid)
            loss /= len(dl_valid)
            
            _ = model.eval()
            X_test, y_test = [], []
            for batch_idx, (x, y) in enumerate(tqdm(dl_test)):        
                x   = x.to(device, non_blocking=True)
                fts = model.forward_encoder(x, mask_ratio=0)[0]
                
                X_test.append(to_numpy(fts.mean(axis=1)))
                y_test.append(to_numpy(y))
            
            X_test = normalize(np.row_stack(X_test))
            y_test = np.hstack(y_test)
            
            if (epoch == 0) or (epoch % 100 != 0):
                sel = np.random.choice(X_valid.shape[0], int(X_valid.shape[0] // 10), replace=False)
            else:
                sel = np.arange(X_valid.shape[0])
            
            clf       = LinearSVC().fit(X_valid[sel], y_valid[sel])
            valid_acc = (clf.predict(X_valid) == y_valid).mean()
            test_acc  = (clf.predict(X_test) == y_test).mean()
            
            log_writer.add_scalar('acc/valid', float(valid_acc), epoch * len(dl_train))
            log_writer.add_scalar('acc/test',  float(test_acc),  epoch * len(dl_train))
            log_writer.add_scalar('loss/test', float(loss),  epoch * len(dl_train))
            log_writer.flush()
            print(valid_acc, test_acc)

# !! learning rate schedule
# ?? super convergence
# !! change model capacity

state_dict = model.cpu().state_dict()
torch.save(state_dict, 'cifar10_model1.pt')
