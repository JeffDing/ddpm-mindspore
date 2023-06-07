import os
import argparse

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init

import time

from ddm import Unet, GaussianDiffusion, Trainer


parser = argparse.ArgumentParser(description='train afhq dataset')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--local_data_root', default='/cache/data/',
                    help='a directory used for transfer data between local path and OBS path')
parser.add_argument('--data_url', metavar='DIR',
                    default='', help='path to dataset')
parser.add_argument('--train_url', metavar='DIR',
                    default='', help='save output')
parser.add_argument('--multi_data_url',help='path to multi dataset',
                    default= '/cache/data/')
parser.add_argument('-b', '--batch_size', default=2, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--image_size', default=512, type=int,
                    metavar='N', help='img size')
parser.add_argument('--sampling_timesteps', default=250, type=int,
                    metavar='N', help='')
parser.add_argument('--timesteps', default=1000, type=int,
                    metavar='N', help='')
parser.add_argument('--train_num_steps', default=50001, type=int,
                    metavar='N', help='')
parser.add_argument('--save_and_sample_every', default=2000, type=int,
                    metavar='N', help='')
parser.add_argument('--num_samples', default=25, type=int,
                    metavar='N', help='')    
parser.add_argument('--pic_num', default=100, type=int,
                    metavar='N', help='')                  
parser.add_argument('--gradient_accumulate_every', default=2, type=int,
                    metavar='N', help='')
parser.add_argument('--ckpt_url', type=str, default=None,
                    help='load ckpt file path')
parser.add_argument('--ckpt_path', type=str, default=None,
                    help='load ckpt file path')
parser.add_argument('--pretrain_url', type=str, default=None,
                    help='load ckpt file path')
parser.add_argument('--use_qizhi', type=bool, default=False,
                    help='use qizhi')
parser.add_argument('--use_zhisuan', type=bool, default=False,
                    help='use zhisuan')
args = parser.parse_args()

if args.use_qizhi:
    from openi import openi_multidataset_to_env as DatasetToEnv  
    from openi import pretrain_to_env as PretrainToEnv
    from openi import env_to_openi as EnvToOpeni
    data_dir = '/cache/data/'  
    train_dir = '/cache/output/'
    pretrain_dir = '/cache/pretrain/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)      
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(pretrain_dir):
        os.makedirs(pretrain_dir)
    DatasetToEnv(args.multi_data_url,data_dir)


if args.use_zhisuan:
    from openi import c2net_multidataset_to_env as DatasetToEnv  
    from openi import pretrain_to_env as PretrainToEnv
    from openi import env_to_openi as EnvToOpeni
    data_dir = '/cache/data/'  
    train_dir = '/cache/output/'
    pretrain_dir = '/cache/pretrain/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)      
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(pretrain_dir):
        os.makedirs(pretrain_dir)
    DatasetToEnv(args.multi_data_url,data_dir)

path = args.local_data_root

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size=args.image_size,
    timesteps=args.timesteps,             # number of steps
    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    sampling_timesteps=args.sampling_timesteps,
    loss_type='l1'            # L1 or L2
)

if args.use_qizhi == False and args.use_zhisuan == False:
    train_dir=args.train_url
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    
trainer = Trainer(
    diffusion,
    path,
    train_batch_size=args.batch_size,
    train_lr=8e-5,
    train_num_steps=args.train_num_steps,         # total training steps
    gradient_accumulate_every=args.gradient_accumulate_every,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp_level='O1',                        # turn on mixed precision
    save_and_sample_every=args.save_and_sample_every,
    num_samples=args.num_samples,
    pic_num = args.pic_num,
    results_folder=os.path.join(train_dir, 'results'),
    train_url=train_dir
)

if args.use_qizhi or args.use_zhisuan:
    PretrainToEnv(args.pretrain_url, pretrain_dir)
trainer.load(args.ckpt_path)
print('load ckpt successfully')

img_list = trainer.inference()
print(img_list)
trainer.save_images(img_list, train_dir)
print('inference successfully')

if args.use_qizhi or args.use_zhisuan:
    EnvToOpeni(train_dir,args.train_url)