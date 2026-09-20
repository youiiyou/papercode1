import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset


class CUHKPEDES(BaseDataset):
    """
    CUHK-PEDES

    Reference:
    Person Search With Natural Language Description (CVPR 2017)

    URL: https://openaccess.thecvf.com/content_cvpr_2017/html/Li_Person_Search_With_CVPR_2017_paper.html

    Dataset statistics:
    ### identities: 13003
    ### images: 40206,  (train)  (test)  (val)
    ### captions: 
    ### 9 images have more than 2 captions
    ### 4 identity have only one image

    annotation format: 
    [{'split', str,
      'captions', list,
      'file_path', str,
      'processed_tokens', list,
      'id', int}...]
    """
    dataset_dir = 'CUHK-PEDES'

    def __init__(self, root='', verbose=True):
        super(CUHKPEDES, self).__init__()#继承里面写子类名
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'imgs/')

        self.anno_path = op.join(self.dataset_dir, 'reid_raw.json')
        self._check_before_run()#检查是否存在

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)#按split分割标注

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:#控制日志输出的开关参数，是否打印额外详细信息
            self.logger.info("=> CUHK-PEDES Images and Captions are loaded")
            self.show_dataset_info()


    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

  
    def _process_anno(self, annos: List[dict], training=False):#List[dict]意思是一个List里面元素都是dict
        pid_container = set()
        if training:
            dataset = []#这里是列表
            image_id = 0
            for anno in annos:
                pid = int(anno['id']) - 1 # make pid begin from 0
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                captions = anno['captions'] # caption list
                for caption in captions:
                    dataset.append((pid, image_id, img_path, caption))
                image_id += 1
            for idx, pid in enumerate(pid_container):#确保pid从0开始且连续
                # check pid begin from 0 and no break
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container#dataset是每个元素为四元组的列表
        else:
            dataset = {}#这里是字典
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            for anno in annos:
                pid = int(anno['id'])#保持原id
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['file_path'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['captions'] # caption list
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
