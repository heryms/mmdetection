import numpy as np
import math
from collections import namedtuple

DetLabel = namedtuple(
    'DetLabel',
    [
        'clses',
        'imgbboxes',
        'l',  # length
        'w',  # width
        'h',  # height
        'x',
        'y',
        'z',
        'yaw',  # from forward to left
        'alpha',  # observation angle
        'npoints',  # number of points in the objects
        'nobj',  # number of objects
        'id',
        'loc',
        'hminmax',
        'truncate',
        'occlusion',
        'scores'
    ])
# namedtuple support defualts only in pyhton 3
# This is a hacky way to put defualt values in DetLable
DetLabel.__new__.__defaults__ = (None, ) * len(DetLabel._fields)

KITTI_LABELS = {
    'none': (0, 'Background'),
    'Car': (1, 'Vehicle'),
    'Van': (1, 'Vehicle'),
    'Truck': (1, 'Vehicle'),
    'Cyclist': (3, 'Vehicle'),
    'Pedestrian': (2, 'Person'),
    'Person': (2, 'Person'),
    'Person_sitting': (0, 'Person'),
    'Tram': (1, 'Vehicle'),
    'Misc': (0, 'Misc'),
    'DontCare': (0, 'DontCare'),
}


def kitti_label_to_detlabel(path, img3d_to_velo):
    '''
  input:
      path of the txt file
  '''
    detlabel = DetLabel(
        clses=[],
        imgbboxes=[],
        h=[],
        w=[],
        l=[],
        x=[],
        y=[],
        z=[],
        yaw=[],
        alpha=[],
        npoints=[],
        nobj=[],
        truncate=[],
        occlusion=[],
        id=[],
        loc=[],
        hminmax=[],
        scores=[])

    with open(path, mode='r') as file:
        ll = file.read().strip().split("\n")

    for l in ll:
        items = l.strip().split(" ")
        # print(items, len(items))
        if len(items) < 14:
            continue

        # todo in python3 we need to remove the b

        # todo in python2 items[0].encode('ascii')
        label = KITTI_LABELS[items[0]][0]
        if label == 0:
            continue  # ignore those that are not defined in KITTI_LABELS or dc

        alpha = float(items[3])
        obh = float(items[8])
        obw = float(items[9])
        obl = float(items[10])

        x_c = float(items[11])
        y_c = float(items[12])
        z_c = float(items[13])
        ry = float(items[14])
        if len(items) < 16:
            score = 0
        else:
            score = float(items[15])
        # convert to the volendyne coordinate
        x, y, z, _ = img3d_to_velo.dot(np.array([x_c, y_c, z_c, 1.]))

        z = z + obh / 2
        rz = -ry

        # if(any([val < lim[0] or val > lim[1] for val, lim in [(x, xlim), (y, ylim)]])):
        #     continue

        detlabel.clses.append(label)
        detlabel.truncate.append(float(items[1]))
        detlabel.occlusion.append(float(items[2]))
        detlabel.imgbboxes.append((float(items[4]), float(items[5]),
                                   float(items[6]), float(items[7])))

        detlabel.alpha.append(alpha)
        detlabel.h.append(obh)
        detlabel.w.append(obw)
        detlabel.l.append(obl)
        detlabel.x.append(x)
        detlabel.y.append(y)
        detlabel.z.append(z)  # from camara to velo coordinate
        detlabel.yaw.append(rz)
        detlabel.scores.append(score)
        detlabel.loc.append(
            xywlYawToCorner([x, y, obl, obw, math.pi / 2. + rz]))
        detlabel.hminmax.append(convertToHminmax(z, obh))

    detlabel.nobj.append(len(detlabel.x))
    return detlabel


def cornerToXywlYaw(cor):
    xy = np.sum(cor, axis=0) / 4
    w = (np.linalg.norm((cor[0] - cor[1])) + np.linalg.norm(
        (cor[2] - cor[3]))) / 2
    l = (np.linalg.norm((cor[1] - cor[2])) + np.linalg.norm(
        (cor[3] - cor[0]))) / 2

    long_edge = (cor[0] - cor[1] + cor[3] - cor[2]) / 2
    short_edge = (cor[0] + cor[1] - cor[3] - cor[2]) / 2
    yaw = np.arctan2(long_edge[0], long_edge[1])
    return xy, l, w, yaw


def xywlYawToCorner(xywlYaw):
    x, y, l, w, yaw = xywlYaw
    center = [x, y]
    size = [l, w]
    rot = np.asmatrix([[math.cos(yaw), -math.sin(yaw)],
                       [math.sin(yaw), math.cos(yaw)]])
    plain_pts = np.asmatrix([[0.5 * size[0], 0.5 * size[1]],
                             [0.5 * size[0], -0.5 * size[1]],
                             [-0.5 * size[0], -0.5 * size[1]],
                             [-0.5 * size[0], 0.5 * size[1]]])
    tran_pts = np.asarray(rot * plain_pts.transpose())
    return tran_pts.transpose() + np.array([x, y])


def convertToHminmax(z, h):
    return np.array([z - h / 2, z + h / 2])


def read_kitti_project_mat(path):
    with open(path, mode='r') as file:
        ll = file.read().strip().split("\n")
    p2 = np.array([float(numstr) for numstr in ll[2].split()[1:]]).reshape(
        3, 4)
    p2_extend = np.eye(4)
    p2_extend[0:3, 0:4] = p2
    R0_rect = np.array(
        [float(numstr) for numstr in ll[4].split()[1:]]).reshape(3, 3)
    R0_rect_extend = np.eye(4)
    R0_rect_extend[0:3, 0:3] = R0_rect
    tr_velo_to_cam = np.array(
        [float(numstr) for numstr in ll[5].split()[1:]]).reshape(3, 4)
    tr_velo_to_cam_extend = np.eye(4)
    tr_velo_to_cam_extend[0:3, 0:4] = tr_velo_to_cam
    return p2_extend, R0_rect_extend, tr_velo_to_cam_extend


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_calib_matrix(path, extend_matrix=True):
    image_info = {}
    with open(path, mode='r') as f:
        lines = f.readlines()
    P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]]).reshape(
        [3, 4])
    P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]]).reshape(
        [3, 4])
    P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]]).reshape(
        [3, 4])
    P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]]).reshape(
        [3, 4])
    if extend_matrix:
        P0 = _extend_matrix(P0)
        P1 = _extend_matrix(P1)
        P2 = _extend_matrix(P2)
        P3 = _extend_matrix(P3)
    image_info['calib/P0'] = P0
    image_info['calib/P1'] = P1
    image_info['calib/P2'] = P2
    image_info['calib/P3'] = P3
    R0_rect = np.array(
        [float(info) for info in lines[4].split(' ')[1:10]]).reshape([3, 3])
    if extend_matrix:
        rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
        rect_4x4[3, 3] = 1.
        rect_4x4[:3, :3] = R0_rect
    else:
        rect_4x4 = R0_rect
    image_info['calib/R0_rect'] = rect_4x4
    Tr_velo_to_cam = np.array(
        [float(info) for info in lines[5].split(' ')[1:13]]).reshape([3, 4])
    Tr_imu_to_velo = np.array(
        [float(info) for info in lines[6].split(' ')[1:13]]).reshape([3, 4])
    if extend_matrix:
        Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
        Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
    image_info['calib/Tr_velo_to_cam'] = Tr_velo_to_cam
    image_info['calib/Tr_imu_to_velo'] = Tr_imu_to_velo
    return image_info


def image_mask(pc, image, proj_mat):
    pc_extended = np.append(
        pc[:, :3], np.expand_dims(np.ones_like(pc[:, 0]), axis=-1), axis=1)

    idx = np.matmul(pc_extended, np.transpose(proj_mat))
    depth = idx[:, 2]
    idx = idx[:, 0:2] / np.expand_dims(idx[:, 2], axis=-1)
    if isinstance(image, list):
        image_shape = image
    else:
        image_shape = image.shape

    valid_mask = np.logical_and(idx[:, 1] >= 0, idx[:, 1] < image_shape[0])
    valid_mask = np.logical_and(idx[:, 0] >= 0, valid_mask)
    valid_mask = np.logical_and(idx[:, 0] < image_shape[1], valid_mask)
    valid_mask = np.logical_and(valid_mask, depth >= 0)
    return pc[valid_mask]