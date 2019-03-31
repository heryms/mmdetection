import argparse

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader


if __name__ == '__main__':


    config = {
              "split_name": "train.list",
              "directory": "/home/heryms/kitti_dataset/object_det_training/training"}
    config["common"] = {}
    config["common"]["is_training"] = True
    info = {"type": "KittiDataset",
            "config": config,
            }

    dataset = obj_from_dict(info, datasets)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=2,
        workers_per_gpu=2,
        num_gpus=1,
        dist=True,
        shuffle=False
    )
    for i, data in enumerate(data_loader):
        # batch_size = data['img'][0].size(0)
        print("begin ", i, data['img'].data.size())
        data_img = data['img'].data.cpu().numpy()
        mmcv.imshow(data_img, "img", wait_time=0)