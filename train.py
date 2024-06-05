import datetime
from os import path
import math
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset,distributed
import torch.distributed as distributed

from model.model import STCNModel
from dataset.static_dataset import StaticTransformDataset
from dataset.vos_dataset import VOSDataset

from util.logger import TensorboardLogger
from util.hyper_para import HyperParameters
from util.load_subset import load_sub_davis#, load_sub_yv
from IMAGE_QUADRUPLE_SEPARATE import split_images,destroy_temporal_split_folder
import matplotlib.pyplot as plt
from TRAIN_PARAS import train_parameters as train_parameters
"""
training commands:
CUDA_VISIBLE_DEVICES=[a,b] OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port [11708] --nproc_per_node=2 train.py --id [karlexp] --stage [0]

 Replace a, b with the GPU ids, cccc with an unused port number, defg with a unique experiment identifier, and h with the training stage (0/1/2/3).

The model is trained progressively with different stages (0: static images; 1: BL30K; 2: 300K main training; 3: 150K main training). After each stage finishes, we start the next stage by loading the latest trained weight.

(Models trained on stage 0 only cannot be used directly. See model/model.py: load_network for the required mapping that we do.)

The .pth with _checkpoint as suffix is used to resume interrupted training (with --load_model) which is usually not needed. Typically you only need --load_network and load the last network weights (without checkpoint in its name).

So, to train a s012 model, we launch three training steps sequentially as follows:

And to train a s03 model, we launch two training steps sequentially as follows:
Pre-training on static images: 
###CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 10708 --nproc_per_node=2 /root/data/videoanno/STCN-main/train.py --id retrain_s0 --stage 0
###OR:CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 10708 --use_env --nproc_per_node=2 /root/data/videoanno/STCN-main/train.py --id retrain_s0 --stage 0
###OR:CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.run --nproc_per_node=2 /root/data/videoanno/STCN-main/train.py --id retrain_s0 --stage 0
Main training: 
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9842 --nproc_per_node=2 /root/data/videoanno/STCN-main/train.py --id retrain_s03 --load_network /root/data/Model_saves_lvlanjiejing/Mar17_22.26.17_retrain_s03/Mar17_22.26.17_retrain_s03_1500.pth  --stage 3

"""
"""
Initial setup
"""
#训练文件夹格式：
# Data format:
#     self_dataset/
#         JPEGImages/
#             video1/
#                 00000.jpg
#                 00001.jpg
#                 ...
#             video2/
#                 ...
#         Annotations/
#             video1/
#                 00000.png
#             video2/
#                 00000.png
#             ...
# """


# Init distributed environment
distributed.init_process_group(backend="nccl")
# Set seed to ensure the same initialization
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)


# Parse command line arguments
para = HyperParameters()
para.parse()
##########################################################################################################
#保存的训练好的网络的文件夹，这里改变文件夹名称即可
train_parameters= train_parameters()
saved_models_Are_at=train_parameters.save_model_at
cudas=torch.cuda.device_count()
print('CUDA Device count: ', cudas)
separate_flag=train_parameters.separate_flag #Flag to separate the images
para['davis_root']=train_parameters.data_path
###########################################################################################################
if para['benchmark']:
    torch.backends.cudnn.benchmark = True

local_rank = torch.distributed.get_rank()
print("local_rank",local_rank)
#local_rank = int(os.getenv('LOCAL_RANK', 1))
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

print('I am rank %d in this world of size %d!' % (local_rank, world_size))

"""
Model related
"""
if local_rank == 0:
    # Logging
    if para['id'].lower() != 'null':
        print('I will take the role of logging!')
        long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), para['id'])
    else:
        long_id = None
    logger = TensorboardLogger(para['id'], long_id)
    logger.log_string('hyperpara', str(para))

    # Construct the rank 0 model
    model = STCNModel(para, logger=logger, 
                    save_path=path.join(saved_models_Are_at, long_id, long_id) if long_id is not None else None, 
                    local_rank=local_rank, world_size=world_size).train()
else:
    # Construct model for other ranks
    model = STCNModel(para, local_rank=local_rank, world_size=world_size).train()

# Load pertrained model if needed
if para['load_model'] is not None:
    total_iter = model.load_model(para['load_model'])
    print("total_iter:",total_iter)
    print('Previously trained model loaded!')
else:
    total_iter = 0

if para['load_network'] is not None:
    model.load_network(para['load_network'])
    print('Previously trained network loaded!')

"""
Dataloader related
"""
# To re-seed the randomness everytime we start a worker
def worker_init_fn(worker_id): 
    return np.random.seed(torch.initial_seed()%(2**31) + worker_id + local_rank*100)

def construct_loader(dataset):

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)
    train_loader = DataLoader(dataset, para['batch_size'], sampler=train_sampler, num_workers=para['num_workers'],
                            worker_init_fn=worker_init_fn, drop_last=True, pin_memory=True)
    return train_sampler, train_loader

def renew_vos_loader(max_skip,separate_flag):
    # //5 because we only have annotation for every five frames
    # yv_dataset = VOSDataset(path.join(yv_root, 'JPEGImages'), 
    #                     path.join(yv_root, 'Annotations'), max_skip//5, is_bl=False, subset=load_sub_yv())
    
    if separate_flag==True:
        split_images(davis_root)
        davis_dataset = VOSDataset(path.join(davis_root, 'JPEGImages/sliced'), 
                        path.join(davis_root, 'Annotations/sliced'), max_skip, is_bl=False, subset=load_sub_davis())
    else:
        davis_dataset = VOSDataset(path.join(davis_root, 'JPEGImages'), 
                        path.join(davis_root, 'Annotations'), max_skip, is_bl=False, subset=load_sub_davis())
    #train_dataset = ConcatDataset([davis_dataset]*5 + [yv_dataset])
    train_dataset = ConcatDataset([davis_dataset]*10)
    #print('YouTube dataset size: ', len(yv_dataset))
    print('DAVIS dataset size: ', len(davis_dataset))
    print('Concat dataset size: ', len(train_dataset))
    print('Renewed with skip: ', max_skip)

    return construct_loader(train_dataset)

# def renew_bl_loader(max_skip):
#     train_dataset = VOSDataset(path.join(bl_root, 'JPEGImages'), 
#                         path.join(bl_root, 'Annotations'), max_skip, is_bl=True)

#     print('Blender dataset size: ', len(train_dataset))
#     print('Renewed with skip: ', max_skip)

#     return construct_loader(train_dataset)

"""
Dataset related
"""

"""
These define the training schedule of the distance between frames
We will switch to skip_values[i] once we pass the percentage specified by increase_skip_fraction[i]
Not effective for stage 0 training
"""
skip_values = [10, 15, 20, 25, 5]

if para['stage'] == 0:
    # static_root = path.expanduser(para['static_root'])
    static_root = "/root/data/videoanno/static"
    #fss_dataset = StaticTransformDataset(path.join(static_root, 'fss'), method=0)
    duts_tr_dataset = StaticTransformDataset(path.join(static_root, 'DUTS-TR'), method=1)
    duts_te_dataset = StaticTransformDataset(path.join(static_root, 'DUTS-TE'), method=1)
    ecssd_dataset = StaticTransformDataset(path.join(static_root, 'ecssd'), method=1)

    big_dataset = StaticTransformDataset(path.join(static_root, 'BIG_small'), method=1)
    hrsod_dataset = StaticTransformDataset(path.join(static_root, 'HRSOD_small'), method=1)

    # BIG and HRSOD have higher quality, use them more
    #train_dataset = ConcatDataset([fss_dataset, duts_tr_dataset, duts_te_dataset, ecssd_dataset]
    train_dataset = ConcatDataset([duts_tr_dataset, duts_te_dataset, ecssd_dataset]
            + [big_dataset, hrsod_dataset]*1)
    train_sampler, train_loader = construct_loader(train_dataset)

    print('Static dataset size: ', len(train_dataset))
#elif para['stage'] == 1:
    #increase_skip_fraction = [0.1, 0.2, 0.3, 0.4, 0.8, 1.0]
    #bl_root = path.join(path.expanduser(para['bl_root']))

    #train_sampler, train_loader = renew_bl_loader(5)
    #renew_loader = renew_bl_loader
else:
    # stage 2 or 3
    increase_skip_fraction = [0.1, 0.2, 0.3, 0.4, 0.9, 1.0]
    # VOS dataset, 480p is used for both datasets
    #yv_root = path.join(path.expanduser(para['yv_root']), 'train_480p')
    #davis_root = path.join(path.expanduser(para['davis_root']), '2017', 'trainval')
    davis_root = path.join(para['davis_root'],  'trainval')
    train_sampler, train_loader = renew_vos_loader(5,separate_flag)
    renew_loader = renew_vos_loader


"""
Determine current/max epoch
"""
total_epoch = math.ceil(para['iterations']/len(train_loader))
print("total_epoch:",total_epoch)
print("len(train_loader):",len(train_loader))
#total_epoch = 50
current_epoch = total_iter // len(train_loader)
print('Current epoch: ', current_epoch)
#current_epoch = total_epoch / len(train_loader)
print('Number of training epochs (the last epoch might not complete): ', total_epoch)
if para['stage'] != 0:
    increase_skip_epoch = [round(total_epoch*f) for f in increase_skip_fraction]
    # Skip will only change after an epoch, not in the middle
    print('The skip value will increase approximately at the following epochs: ', increase_skip_epoch[:-1])

"""
Starts training
"""
# Need this to select random bases in different workers

np.random.seed(np.random.randint(2**30-1) + local_rank*100)
try:
    # if separate_flag==True:
    #     split_images(davis_root)

    loss_values_gpu = []
    epoch_loss_gpu = []
    for cuda_ids in range(cudas):
        loss_values_gpu.append(f'loss_values_gpu{cuda_ids}')
        epoch_loss_gpu.append(f'epoch_loss_gpu{cuda_ids}')
        loss_values_gpu[cuda_ids]=[]
        epoch_loss_gpu[cuda_ids]=[]
        
    for e in range(current_epoch, total_epoch): 
        print('Epoch %d/%d' % (e, total_epoch))
        if para['stage']!=0 and e!=total_epoch and e>=increase_skip_epoch[0]:
            while e >= increase_skip_epoch[0]:
                cur_skip = skip_values[0]
                skip_values = skip_values[1:]
                increase_skip_epoch = increase_skip_epoch[1:]
            print('Increasing skip to: ', cur_skip)
            train_sampler, train_loader = renew_loader(cur_skip,separate_flag)

        # Crucial for randomness! 
        train_sampler.set_epoch(e)

        # Train loop
        model.train()

        for data in train_loader:
            loss = model.do_pass(data, total_iter)
            #print("loss:",loss.item(),loss.device)
            #分布式计算中，loss.device是一个数据类型，不是一个字符串
            #若要分别获取loss.device中cuda:1和cuda:2的值，需要将loss.device转换为字符串，然后再用if判断
            total_iter += 1
            if loss.item() is not None:
                for cuda_ids in range(cudas):
                    if str(loss.device)=='cuda:'+str(cuda_ids):
                        epoch_loss_gpu[cuda_ids].append(loss.item())
                        # print('Current loss: ', loss.device,epoch_loss_gpu[cuda_ids])
            # if total_iter % para['report_interval'] == 0:#这句话是说，每100次迭代，打印一次loss
            #     print('Iter %d, loss: %.4f' % (total_iter, loss.item()))
        for cuda_ids in range(cudas):
            if str(loss.device)=='cuda:'+str(cuda_ids):
                loss_values_gpu[cuda_ids].append(np.mean(epoch_loss_gpu[cuda_ids]))
                print('loss_values_gpu:',loss_values_gpu[cuda_ids])
        
        #print('0loss_values_gpu1:',loss_values_gpu1)
        if total_iter >= para['iterations']:
            break

    if not para['debug']:
        print("/////////////////////////////////////////////////////////////saving///////////////////////////////////////////////////////////////////////////////////////")
        model.save(total_iter)
        # Plotting the loss curve for GPU 0
        for cuda_ids in range(cudas):
            if str(loss.device)=='cuda:'+str(cuda_ids):
                loss_values = loss_values_gpu[cuda_ids]
                i=cuda_ids
                loss_txt_address=os.path.join('/root/data/videoanno/STCN-main/',f'epoch_loss_gpu{i}.txt')

                with open(loss_txt_address, 'w') as f: 
                    f.write('')#清空文件
                    for j in range (len(loss_values)):
                        f.write(f'Epoch {current_epoch+j}: GPU {i} Loss: {loss_values[j]}\n')
                plt.plot(range(current_epoch, current_epoch + len(loss_values)), loss_values)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title(f'Loss Curve (GPU {i})')
                plt.savefig(f'loss_curve_gpu{i}.png')
                plt.close()
        

finally:
    # Clean up
    if separate_flag==True:
        destroy_temporal_split_folder(path.join(davis_root, 'JPEGImages/sliced'))
        destroy_temporal_split_folder(path.join(davis_root, 'Annotations/sliced'))
    distributed.destroy_process_group()
