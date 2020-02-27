#!/usr/bin/python
# -*- coding: utf-8 -*-
from os import listdir, makedirs, system, chdir, getcwd
from os.path import isfile, join, dirname, realpath, isdir
import json
import numpy as np
from models import Frame, fixed_annotation_error
from pyntcloud import PyntCloud
from preprocess import preprocess



# temporary script, will be replaced later
pythonApp = "python "
script_seg = "eval_rcnn.py --cfg_file cfgs/argo_config_sampling_trainfull.yaml --rcnn_ckpt checkpoint_epoch_40.pth --rpn_ckpt checkpoint_epoch_50.pth --batch_size 1 --eval_mode rcnn --test"

class FrameHandler:

    CUR_DIR = dirname(realpath(__file__))
    DATASET_DIR = join(CUR_DIR, 'test_dataset')
    INPUT_BIN_DIR = 'sample/argoverse/lidar'
    GROUND_REMOVED_DIR = 'ground_removed'
    OUTPUT_ANN_DIR = join(CUR_DIR, 'output')

    def __init__(self):
        self.drives = dict()
        for drive in listdir(self.DATASET_DIR):
            if 'sync' in drive:
                bin_dir = join(self.DATASET_DIR, drive,
                               self.INPUT_BIN_DIR)
                self.drives[drive] = []
                for f in listdir(bin_dir):
                    #print(join(bin_dir, f),f)
                    if isfile(join(bin_dir, f)) and '.ply' in f:
                        self.drives[drive].append(f.split('.ply')[0])
                self.drives[drive] = sorted(self.drives[drive])

    def get_frame_names(self):
        """
        Get all the frame names
        """

        # return ",".join(self.frame_names)

        return str(self.drives)

    
    def load_pc(self,pc_filepath):
        full_data = PyntCloud.from_file(pc_filepath)
        x = np.array(full_data.points.x)[:, np.newaxis]
        y = np.array(full_data.points.y)[:, np.newaxis]
        z = np.array(full_data.points.z)[:, np.newaxis]
        full_data = np.concatenate([x,y,z], axis = 1)

        return full_data

    def get_pointcloud(
        self,
        drivename,
        fname,
        dtype=str,
        ground_removed=False,
        ):

        seg_dir = "PointCNN/output/rcnn/argo_config_sampling_trainfull/eval/epoch_no_number/sample/test_mode/rpn_result/data"
        data_dir = "test_dataset/0_drive_0064_sync/sample/argoverse/lidar"
        orig_lidar = data_dir + "/" + fname + ".ply"
        seg_file = seg_dir + "/" + fname + ".npy"
    
        if not isfile(seg_file):
            # execute pointrcnn
            # currently have to run on the whole files to generate corresponding out
            # will be replaced by only inferencing on this specific fil
            preprocess()

        full_data = None
        if isfile(seg_file):
            #full_data = np.fromfile(orig_lidar).astype('float32').reshape(-1,3) 
            
            #data = np.load(seg_file).reshape(-1, 5)
            #data[np.isnan(data)] = .0
            pts_lidar = self.load_pc(orig_lidar)#np.asarray(o3d.io.read_point_cloud(orig_lidar).points).astype('float32')
            #x = pts_lidar[:,0].reshape(-1,1)
            #y = pts_lidar[:,1].reshape(-1,1)
            #z = pts_lidar[:,2].reshape(-1,1)
            #full_data = np.concatenate([x,y,z], axis = 1)
            full_data = np.hstack((pts_lidar,np.arange(pts_lidar.shape[0]).reshape(-1,1)))
            #full_data = load_pc(orig_lidar)

            
        #bin_dir = join(self.DATASET_DIR, drivename, self.INPUT_BIN_DIR)
        #filename = join(bin_dir, fname.split('.')[0] + '.ply')
        #pc = PyntCloud.from_file(filename)
        #data = pc.points.to_numpy()[:, :4]
        #print(data)
        #data[np.isnan(data)] = .0
        #print(data.dtype)
        
        if dtype == str:
            data = full_data.flatten(order='C').tolist()
            data_str = ','.join([str(x) for x in data])
            return data_str
        return full_data

    def load_annotation(
        self,
        drivename,
        fname,
        settingsControls,
        dtype='object',
        ):
        fname = settingsControls['AnnotatorId'] + '.' + fname.split('.'
                )[0] + '.json'
        try:
            with open(join(self.OUTPUT_ANN_DIR, drivename, fname), 'r'
                      ) as read_file:
                #print ('file: ', read_file)
                try:
                    frame = json.load(read_file)
                    if dtype == 'object':
                        return Frame.parse_json(frame)
                    else:
                        return frame
                except json.JSONDecodeError:
                    return ''
        except:
            return ''

    def save_annotation(
        self,
        drivename,
        fname,
        json_str,
        settingsControls,
        ):
        """
........Saves json string to output directory. 

........Inputs:
........- fname: Frame name. Can have file extension. 
........- json_str: String in json to be saved

........Returns 1 if successful, 0 otherwise
........"""

        assert type(json_str) == str, 'json must be a string'
        if not isdir(self.OUTPUT_ANN_DIR):
            try:
                makedirs(self.OUTPUT_ANN_DIR)
            except OSError:
                print('Creation of the directory {} failed'.format(self.OUTPUT_ANN_DIR))

        output_drive_dir = join(self.OUTPUT_ANN_DIR, drivename)
        if not isdir(output_drive_dir):
            try:
                makedirs(output_drive_dir)
            except OSError:
                print('Creation of the directory {} failed'.format(output_drive_dir))

        try:
            json_object = json.loads(json_str)
            print("FrameTracking", settingsControls["FrameTracking"])
            if(settingsControls["FrameTracking"]): # Update kalman state
                json_object = fixed_annotation_error(json_object)
            json_str = json.dumps(json_object)
        except ValueError:
            print('Annotation not a valid json')
            return 0
        fname = settingsControls['AnnotatorId'] + '.' + fname.split('.'
                )[0] + '.json'
        save_filename = join(self.OUTPUT_ANN_DIR, drivename, fname)
        print('save_filename', save_filename)
        with open(save_filename, 'w') as f:
            f.write(json_str)
            return 1
        return 0

