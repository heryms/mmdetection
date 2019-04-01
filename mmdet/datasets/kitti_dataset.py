import numpy as np
import cv2
import os
from functools import partial
from .kitti_utils import dataset_utils as ds
from torch.utils.data import Dataset
from mmcv.parallel import DataContainer as DC
from .utils import to_tensor


class KittiDataset(Dataset):
    def __init__(self, config, prep_func=None):
        print("init kitti dataset")
        directory = config["directory"]
        self.directory = directory
        split_name = config.get("split_name", None)
        self.split_name = split_name
        self.config = config
        self.update_dir()


        self.groundz = -1.7
        self._prep_func = prep_func

        self.is_training = self.config["common"]["is_training"]

        if self.is_training:
            self._set_group_flag()

    def update_dir(self):
        self.img_dir = self.directory + "/image_2"
        self.pc_dir = self.directory + "/velodyne"
        self.calib_dir = self.directory + "/calib"
        self.label_dir = self.directory + "/label_2"
        self.pred_dir = self.directory + "/pred_dir"
        self.split_dir = self.directory + "/list"
        self.name_list_ = sorted(
            [name.split('.')[0] for name in os.listdir(self.img_dir)])
        self.num_files_ = len(self.name_list_)
        if not os.path.exists(self.pred_dir):
            os.makedirs(self.pred_dir)
        if self.split_name is not None:
            self.update_name_list(self.split_name)

    def update_name_list(self, split_name):
        list_file_path = os.path.join(self.split_dir, split_name)
        with open(list_file_path, 'r') as f:
            lines = f.read().split('\n')
            new_name_list = [
                name.split('.')[0] for name in lines if len(name) > 4
            ]
        self.name_list_ = new_name_list
        self.num_files_ = len(self.name_list_)
        print("updated split {} with {} files".format(split_name,
                                                      self.num_files_))

    def get_img(self, i):
        imgpath = os.path.join(self.img_dir, self.name_list_[i] + '.png')
        return cv2.imread(imgpath)

    def get_pc(self, i):
        lidarpath = os.path.join(self.pc_dir, self.name_list_[i] + '.bin')
        points_v = np.fromfile(
            lidarpath, dtype=np.float32, count=-1).reshape([-1, 4])
        return points_v

    def get_calib(self, i):
        calibpath = os.path.join(self.calib_dir, self.name_list_[i] + '.txt')
        p2_extend, R0_rect_extend, tr_velo_to_cam_extend = \
          ds.read_kitti_project_mat(calibpath)
        velo_to_img = p2_extend.dot(R0_rect_extend.dot(tr_velo_to_cam_extend))
        return velo_to_img

    def get_detlabels(self, i):
        calibpath = os.path.join(self.calib_dir, self.name_list_[i] + '.txt')
        p2_extend, R0_rect_extend, tr_velo_to_cam_extend = \
          ds.read_kitti_project_mat(calibpath)
        img3d_to_velo = np.linalg.inv(
            R0_rect_extend.dot(tr_velo_to_cam_extend))
        labelpath = os.path.join(self.label_dir, self.name_list_[i] + '.txt')
        detlabel = ds.kitti_label_to_detlabel(labelpath, img3d_to_velo)
        return detlabel

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            self.flag[i] = 1

    def __len__(self):
        return self.num_files_

    def __getitem__(self, idx):

        if idx >= self.num_files_:
            np.random.shuffle(self.name_list_)
        pc = self.get_pc(idx)
        img = self.get_img(idx)
        img = img.transpose(2, 0, 1)
        # print(pc.shape)
        # print(img.shape)
        calib = self.get_calib(idx)
        if self.is_training:
            label = self.get_detlabels(idx)
        img_meta = dict(
            ori_shape=img.shape,
            img_shape=img.shape)
        pc_mask = np.ones_like(pc)
        point_num = len(pc)
        pc_meta = dict(
            point_num=point_num
        )
        if self.is_training:
            data = dict(
                img=DC(to_tensor(img), stack=True),
                img_meta=DC(img_meta, cpu_only=True),
                pc_meta=DC(pc_meta, cpu_only=True),
                gt_bboxes=label,
                pc=DC(to_tensor(pc), stack=True),
                calib=DC(to_tensor(calib), stack=True),
                pc_mask=DC(to_tensor(pc_mask), stack=True)
            )
        else:
            data = dict(
                img=[to_tensor(img)],
                img_meta=DC(img_meta, cpu_only=True),
                pc_meta=DC(pc_meta, cpu_only=True),
                pc=[to_tensor(pc)],
                calib=[to_tensor(calib)],
                pc_mask=[to_tensor(pc_mask)]
            )
        return data