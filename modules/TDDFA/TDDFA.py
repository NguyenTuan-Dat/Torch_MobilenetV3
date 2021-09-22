import os.path as osp
import numpy as np
import cv2
import yaml

from modules.TDDFA.utils.io import _load
from modules.TDDFA.utils.tddfa_util import _parse_param, similar_transform
from modules.TDDFA.bfm.bfm import BFMModel
from modules.TDDFA.utils.pose import calc_pose

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)

class TDDFA_Blob(object):
    """TDDFA_ONNX: the ONNX version of Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self):
        # torch.set_grad_enabled(False)
        kvs = yaml.load(open("modules/TDDFA/configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
        # load onnx version of BFM
        bfm_fp = kvs.get('bfm_fp', make_abs_path('/configs/bfm_noneck_v3.pkl'))

        # load for optimization
        bfm = BFMModel(bfm_fp, shape_dim=kvs.get('shape_dim', 40), exp_dim=kvs.get('exp_dim', 10))
        self.tri = bfm.tri
        self.u_base, self.w_shp_base, self.w_exp_base = bfm.u_base, bfm.w_shp_base, bfm.w_exp_base

        # config
        self.size = kvs.get('size', 120)

        param_mean_std_fp = f'modules/TDDFA/configs/param_mean_std_62d_{self.size}x{self.size}.pkl'

        # params normalization config
        r = _load(param_mean_std_fp)
        self.param_mean = r.get('mean')
        self.param_std = r.get('std')

    def preprocess_image(self, face_frame):

        img = face_frame
        # cv2.imshow("face", img)
        img = cv2.resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)[np.newaxis, ...]

        inp_dct = {'input': img}
        return inp_dct

    def postprocess(self, param):
        param = param.flatten().astype(np.float32)
        param = param * self.param_std + self.param_mean  # re-scale
        return param

    def recon_vers(self, param, roi_box):
        size = self.size
        R, offset, alpha_shp, alpha_exp = _parse_param(param)
            
        pts3d = R @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp). \
            reshape(3, -1, order='F') + offset
        pts3d = similar_transform(pts3d, roi_box, size)
        return pts3d

    def cal_pose(self, param):
        _, pose = calc_pose(param)
        # print(f'yaw: {pose[0]:.1f}, pitch: {pose[1]:.1f}, roll: {pose[2]:.1f}')
        return pose[0], pose[1], -pose[2]
