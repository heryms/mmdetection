import torch
import torch.nn as nn
import numpy as np
import open3d
import time
from torch_scatter import scatter_max


def read_pointcloud(file):
  data = np.fromfile(file, np.float32)
  data = np.reshape(data, (-1, 4))
  return data


def mask_op(data, x_min, x_max):
  # print(x_min)
  # x_min = torch.FloatTensor([x_min]).cuda()
  # x_max = torch.FloatTensor([x_max]).cuda()
  mask = (data > x_min) & (data < x_max)
  return mask


def get_fov(data):
  mask = data[:, 0] > torch.abs(data[:, 1])
  return mask


def quantitize(data, lim_min, lim_max, size):
  idx = (data - lim_min) / (lim_max - lim_min) * size
  idx = idx.type(torch.IntTensor)
  return idx



class PFELayer(nn.Module):
  def __init__(self):
    super(PFELayer, self).__init__()

  def forward(self, pc, lim, size):
    x = pc[:, 0]
    y = pc[:, 1]

    mask_x = mask_op(x, lim[0][0], lim[0][1])
    mask_y = mask_op(y, lim[1][0], lim[1][1])
    fov_mask = get_fov(pc)
    # fastest one gather op
    mask = (mask_x) & (mask_y)
    pc = pc[mask]
    xidx = quantitize(pc[:, 0], lim[0][0], lim[0][1], size[0])
    yidx = quantitize(pc[:, 1], lim[1][0], lim[1][1], size[1])
    idx = xidx * size[0].type(torch.IntTensor) + yidx
    idx_l = idx.long()
    idx_3 = idx.view(-1, 1).repeat(1, 4).long().cuda()
    # pc_feature = pc.transpose(0, 1).cuda()
    # print(idx_3.size(), pc_feature.size())
    # out = torch.zeros((4, 640 * 640)).cuda()

    out = pc.new_zeros((640 * 640, 4))
    # s = time.time()
    out,  argmax = scatter_max(pc, idx_3,out=out, dim=0)
    print(out.size())
    v = out[idx_l, :]
    print(v.size())
    # a1, a2 = np.unique(idx, return_counts=True)
    # a2 = torch.from_numpy(a2).cuda()
    # # x_unique_count = torch.stack([(idx == x_u).sum() for x_u in x_unique])
    return pc, out


def test_scatter():
  from torch_scatter import scatter_max

  src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
  index = torch.tensor([4, 5, 4, 2, 3])
  out = src.new_zeros((2, 6))

  out, argmax = scatter_max(src, index, out=out)

  print(out)


def test_scatter_v2():
  from torch_scatter import scatter_max


  # src = torch.rand((110000, 128)).cuda()
  s1 = time.time()
  src = torch.cuda.FloatTensor(110000, 128).uniform_()
  # index = torch.cuda.FloatTensor(110000).uniform_().view((-1, 1)).long().expand_as(src)
  # print(index.size())
  index = torch.randint(0, 1200, (110000, )).view((-1, 1)).long().expand_as(src).cuda()

  s = time.time()
  out = src.new_zeros((640*640, 128)).cuda()
  out, argmax = scatter_max(src, index, out=out, dim=0)
  e = time.time()
  print((e - s) * 1000, " ms")
  print((e - s1) * 1000, " ms")


if __name__ == "__main__":
  lidar = read_pointcloud("/home/heryms/kitti"
                          "/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/0000000000.bin")

  lim = [[torch.FloatTensor([-32.]).cuda(),
          torch.FloatTensor([64.]).cuda()],
         [torch.FloatTensor([-48.]).cuda(),
          torch.FloatTensor([48.]).cuda()]]
  size = [torch.FloatTensor([640.]).cuda(), torch.FloatTensor([640.]).cuda()]
  for i in range(1000):
    test_scatter_v2()
  exit(0)
  # layer = PFELayer()
  # data_torch = torch.FloatTensor(lidar).cuda()
  #
  # torch.IntTensor
  # for i in range(2000):
  #   s = time.time()
  #   out = layer(data_torch, lim, size)
  #   e = time.time()
  #   print((e - s) * 1000, " ms")
  # print(out.shape)
  #
  # pcd = open3d.PointCloud()
  # pcd.points = open3d.Vector3dVector(out[0].cpu().numpy()[:, :3])
  # open3d.draw_geometries([pcd])
  # a = [-32.]
  # a_torch = torch.FloatTensor(a)
  # print(a_torch)
  #
  #
  #
  #
  #
