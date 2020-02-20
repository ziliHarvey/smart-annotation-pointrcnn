import os
import numpy as np
import torch.utils.data as torch_data
import lib.utils.calibration as calibration
import lib.utils.kitti_utils as kitti_utils
from PIL import Image
import argoverse
import lib.datasets.ground_segmentation as gs
from pyntcloud import PyntCloud
import random
import copy


class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='train'):
        self.split = split
        is_test = self.split == 'test'
        self.lidar_dir = os.path.join(root_dir,"sample/argoverse/lidar")
        lidarfile_list = os.listdir(self.lidar_dir)
        
        self.lidar_idx_list = ['%06d'%l for l in range(len(lidarfile_list))]
        self.lidar_names = [x.split('.')[0] for x in lidarfile_list]

        self.lidar_file_extension = [x.split('.')[1] for x in lidarfile_list]
        
        self.lidar_name_table = dict(zip(self.lidar_idx_list, self.lidar_names))
        self.lidar_ext__table = dict(zip(self.lidar_idx_list, self.lidar_file_extension))

        self.num_sample = self.lidar_idx_list.__len__()
        
        self.argo_to_kitti = np.array([[6.927964e-03, -9.999722e-01, -2.757829e-03],
                                       [-1.162982e-03, 2.749836e-03, -9.999955e-01],
                                       [9.999753e-01, 6.931141e-03, -1.143899e-03]])

        self.ground_removal = True
        
    def get_lidar(self,idx):
        ext = self.lidar_ext__table["%06d"%idx]
        lidar_file = os.path.join(self.lidar_dir,self.lidar_name_table["%06d"%idx] + '.'+ ext )
        
        assert os.path.exists(lidar_file)
        
        if(ext == 'ply'):
            data = PyntCloud.from_file(lidar_file)
            x = np.array(data.points.x)[:, np.newaxis]
            y = np.array(data.points.y)[:, np.newaxis]
            z = np.array(data.points.z)[:, np.newaxis]
            pts_lidar = np.concatenate([x,y,z], axis = 1)   

        elif(ext == 'bin'):
            pts_lidar = np.fromfile(lidar_file).reshape(-1,3)[:,:3]

        else:
            pass
        
        if self.ground_removal: 
            pts_lidar = gs.ground_segmentation(pts_lidar)
        
        pts_lidar = np.dot(self.argo_to_kitti,pts_lidar.T).T
        
        return pts_lidar
