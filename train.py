import os  #操作系统交互
import os.path as op #操作路径简化
import torch  #框架核心
import numpy as np  #数值计算
import random  #随机数生成
import time   #时间管理模块

import torch.distributed as dist

from datasets import build_dataloader  #数据加载器构建
from processor.processor import do_train  #训练过程核心逻辑
from utils.checkpoint import Checkpointer  #模型保存/加载
from utils.iotools import save_train_configs  #配置保存
from utils.logger import setup_logger  #日志系统
from solver import build_optimizer, build_lr_scheduler  #优化器和学习率调度
from model import build_model  #模型构建
from utils.metrics import Evaluator  #评估器
from utils.options import get_args  #命令行参数解析
from utils.comm import get_rank, synchronize  #分布式训练通信



def set_seed(seed=0):  #随机种子设置
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  #多GPU设置
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  #保证可复现性
    torch.backends.cudnn.benchmark = True  #提高训练速度

def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.distributed = True
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    else:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

if __name__ == '__main__':
    args = get_args()##读取参数
    init_distributed_mode(args)

    set_seed(1+args.rank)##设置随机种子多GPU下
    name = args.name

  
    
    device = torch.device("cuda", args.local_rank)

    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())#获取本时区时间 格式4位年2位月2位日_2位时2位分2位秒（补零）
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{name}')#创建输出目录output_dir/数据集名/时间_实验名
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank()) #日志器初始化 实现「控制台输出 + 文件保存」双输出
    logger.info(f"Using {args.world_size} GPUs")
    logger.info(str(args).replace(",", "\n"))
    save_train_configs(args.output_dir, args)

    # get image-text pair datasets dataloader  核心组件构建
    # ===== dataloader (所有 rank 完全一致) =====
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)

# 其他 rank 再创建 loader（用 num_workers=0 更安全）
# if get_rank() != 0:
#     train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
model = build_model(args, num_classes) #模型构建
logger.info('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0)) #模型参数统计
model.to(device) #所有参数和缓冲区移动到GPU

if args.distributed: #分布式训练包装模型
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        # this should be removed if we update BatchNorm stats
        broadcast_buffers=False #关闭 BatchNorm 缓冲区（均值 / 方差）的跨 GPU 广播
    )
optimizer = build_optimizer(args, model)
scheduler = build_lr_scheduler(args, optimizer) 

is_master = get_rank() == 0
checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
evaluator = Evaluator(val_img_loader, val_txt_loader)

start_epoch = 1 #初始化起始轮数
if args.resume:
    checkpoint = checkpointer.resume(args.resume_ckpt_file)#加载检查点
    start_epoch = checkpoint['epoch']

do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)
