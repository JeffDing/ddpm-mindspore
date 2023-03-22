import ddpm as ddpm
import argparse
import os
from mindspore import context
from mindspore.communication.management import init
from mindspore.context import ParallelMode
import time


IMAGE_SIZE = 128


def parse_args():
    parser = argparse.ArgumentParser(description="train ddpm",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--pretrain_path',
                        type=str,
                        default=None,
                        help='the pretrain model path')

    parser.add_argument('--data_url',
                        type=str,
                        default="",
                        help='training data file path')

    parser.add_argument('--train_url',
                        default='./results',
                        type=str,
                        help='the path model and fig save path')

    parser.add_argument('--epochs',
                        default=10,
                        type=int,
                        help='training epochs')

    parser.add_argument('--num_samples',
                        default=4,
                        type=int,
                        help='num_samples must have a square root, like 4, 9, 16 ...')

    parser.add_argument('---device_target',
                        default="Ascend",
                        type=str,
                        help='device target')
    args, _ = parser.parse_known_args()
    return args


def train_ddpm():
    epoch = args_opt.epochs
    data_dir = '../dataset/dog/'
    train_dir = './output'
    ckpt_url = './ckpt/checkpoint.ckpt'
    try:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
    except Exception as e:
        print("path already exists")

    print("List dataset:  ", os.listdir(data_dir))
    model = ddpm.Unet(
        dim=IMAGE_SIZE,
        out_dim=3,
        dim_mults=(1, 2, 4, 8)
    )

    diffusion = ddpm.GaussianDiffusion(
        model,
        image_size=IMAGE_SIZE,
        timesteps=20,  # number of time steps
        sampling_timesteps=10,
        loss_type='l1'  # L1 or L2
    )

    trainer = ddpm.Trainer(
        diffusion,
        os.path.join(data_dir),
        train_batch_size=1,
        train_lr=8e-5,
        train_num_steps=epoch,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        save_and_sample_every=5,  # image sampling and step
        num_samples=4,
        results_folder=train_dir
    )
    if args_opt.pretrain_path:
        trainer.load(args_opt.pretrain_path)
    trainer.train()

if __name__ == '__main__':
    args_opt = parse_args()
    train_ddpm()
