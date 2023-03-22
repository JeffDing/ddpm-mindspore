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
    
    parser.add_argument('--data_dir',
                        type=str,
                        default="",
                        help='training data file path')

    parser.add_argument('--train_url',
                        default='./results',
                        type=str,
                        help='the path model and fig save path')
    
    parser.add_argument('--checkpoint_url',
                        default='./results',
                        type=str,
                        help='the path model and fig save path')
    
    parser.add_argument('--train_dir',
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
    parser.add_argument('--use_qizhi',
                        type=bool, default=False,
                        help='use qizhi')
    parser.add_argument('--use_zhisuan', 
                        type=bool, default=False,
                        help='use zhisuan')
    args, _ = parser.parse_known_args()
    return args


def train_ddpm():
    epoch = args_opt.epochs
    data_dir = args_opt.data_dir
    train_dir = args_opt.train_dir
    ckpt_url = args_opt.checkpoint_url
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
    if args_opt.use_qizhi:
        import moxing as mox
        ### Copy single dataset from obs to training image###
        def ObsToEnv(obs_data_url, data_dir):
            try:     
                mox.file.copy_parallel(obs_data_url, data_dir)
                print("Successfully Download {} to {}".format(obs_data_url, data_dir))
            except Exception as e:
                print('moxing download {} to {} failed: '.format(obs_data_url, data_dir) + str(e))
            #Set a cache file to determine whether the data has been copied to obs. 
            #If this file exists during multi-card training, there is no need to copy the dataset multiple times.
            f = open("/cache/download_input.txt", 'w')    
            f.close()
            try:
                if os.path.exists("/cache/download_input.txt"):
                    print("download_input succeed")
            except Exception as e:
                print("download_input failed")
            return 
        ### Copy the output to obs###
        def EnvToObs(train_dir, obs_train_url):
            try:
                mox.file.copy_parallel(train_dir, obs_train_url)
                print("Successfully Upload {} to {}".format(train_dir,obs_train_url))
            except Exception as e:
                print('moxing upload {} to {} failed: '.format(train_dir,obs_train_url) + str(e))
            return      
        def DownloadFromQizhi(obs_data_url, data_dir):
            device_num = int(os.getenv('RANK_SIZE'))
            if device_num == 1:
                ObsToEnv(obs_data_url,data_dir)
                context.set_context(mode=context.GRAPH_MODE,device_target=args_opt.device_target)
            if device_num > 1:
                # set device_id and init for multi-card training
                context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=int(os.getenv('ASCEND_DEVICE_ID')))
                context.reset_auto_parallel_context()
                context.set_auto_parallel_context(device_num = device_num, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, parameter_broadcast=True)
                init()
                #Copying obs data does not need to be executed multiple times, just let the 0th card copy the data
                local_rank=int(os.getenv('RANK_ID'))
                if local_rank%8==0:
                    ObsToEnv(obs_data_url,data_dir)
                #If the cache file does not exist, it means that the copy data has not been completed,
                #and Wait for 0th card to finish copying data
                while not os.path.exists("/cache/download_input.txt"):
                    time.sleep(1)  
            return
        def UploadToQizhi(train_dir, obs_train_url):
            device_num = int(os.getenv('RANK_SIZE'))
            local_rank=int(os.getenv('RANK_ID'))
            if device_num == 1:
                EnvToObs(train_dir, obs_train_url)
            if device_num > 1:
                if local_rank%8==0:
                    EnvToObs(train_dir, obs_train_url)
            return

        data_dir = '/cache/data/'  
        train_dir = '/cache/output/'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        ###Initialize and copy data to training image
        DownloadFromQizhi(args_opt.data_url, data_dir)

    if args_opt.use_zhisuan:
        import moxing as mox
        ### Copy multiple datasets from obs to training image and unzip###  
        def C2netMultiObsToEnv(multi_data_url, data_dir):
            #--multi_data_url is json data, need to do json parsing for multi_data_url
            multi_data_json = json.loads(multi_data_url)  
            for i in range(len(multi_data_json)):
                zipfile_path = data_dir + "/" + multi_data_json[i]["dataset_name"]
                try:
                    mox.file.copy(multi_data_json[i]["dataset_url"], zipfile_path) 
                    print("Successfully Download {} to {}".format(multi_data_json[i]["dataset_url"],zipfile_path))
                    #get filename and unzip the dataset
                    filename = os.path.splitext(multi_data_json[i]["dataset_name"])[0]
                    filePath = data_dir + "/" + filename
                    if not os.path.exists(filePath):
                        os.makedirs(filePath)
                    os.system("unzip {} -d {}".format(zipfile_path, filePath))

                except Exception as e:
                    print('moxing download {} to {} failed: '.format(
                        multi_data_json[i]["dataset_url"], zipfile_path) + str(e))
            #Set a cache file to determine whether the data has been copied to obs. 
            #If this file exists during multi-card training, there is no need to copy the dataset multiple times.
            f = open("/cache/download_input.txt", 'w')    
            f.close()
            try:
                if os.path.exists("/cache/download_input.txt"):
                    print("download_input succeed")
            except Exception as e:
                print("download_input failed")
            return 
        ### Copy the output model to obs ###  
        def EnvToObs(train_dir, obs_train_url):
            try:
                mox.file.copy_parallel(train_dir, obs_train_url)
                print("Successfully Upload {} to {}".format(train_dir,
                                                            obs_train_url))
            except Exception as e:
                print('moxing upload {} to {} failed: '.format(train_dir,
                                                            obs_train_url) + str(e))
            return                                                       
        def DownloadFromQizhi(multi_data_url, data_dir):
            device_num = int(os.getenv('RANK_SIZE'))
            if device_num == 1:
                C2netMultiObsToEnv(multi_data_url,data_dir)
                context.set_context(mode=context.GRAPH_MODE,device_target=args_opt.device_target)
            if device_num > 1:
                # set device_id and init for multi-card training
                context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, device_id=int(os.getenv('ASCEND_DEVICE_ID')))
                context.reset_auto_parallel_context()
                context.set_auto_parallel_context(device_num = device_num, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, parameter_broadcast=True)
                init()
                #Copying obs data does not need to be executed multiple times, just let the 0th card copy the data
                local_rank=int(os.getenv('RANK_ID'))
                if local_rank%8==0:
                    C2netMultiObsToEnv(multi_data_url,data_dir)
                #If the cache file does not exist, it means that the copy data has not been completed,
                #and Wait for 0th card to finish copying data
                while not os.path.exists("/cache/download_input.txt"):
                    time.sleep(1)  
            return
        def UploadToQizhi(train_dir, obs_train_url):
            device_num = int(os.getenv('RANK_SIZE'))
            local_rank=int(os.getenv('RANK_ID'))
            if device_num == 1:
                EnvToObs(train_dir, obs_train_url)
            if device_num > 1:
                if local_rank%8==0:
                    EnvToObs(train_dir, obs_train_url)
            return

        data_dir = '/cache/data/'  
        train_dir = '/cache/output/'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        ###Initialize and copy data to training image
        DownloadFromQizhi(args_opt.multi_data_url, data_dir)
    train_ddpm()
