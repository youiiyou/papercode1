import logging
import os
import sys
import os.path as op


def setup_logger(name, save_dir, if_train, distributed_rank=0):
    logger = logging.getLogger(name)#创建日志器
    logger.setLevel(logging.DEBUG)#设置日志级别为DEBUG

    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if not op.exists(save_dir):
        print(f"{save_dir} is not exists, create given directory")
        os.makedirs(save_dir, exist_ok=True)
    if if_train:
        fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')
    else:
        fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger