#!/usr/bin/python
# -*- coding: utf-8 -*-
from os import listdir, makedirs
from os.path import isfile, join, dirname, realpath, isdir
import json
import numpy as np
from models import Frame, fixed_annotation_error
from pyntcloud import PyntCloud


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
                    if isfile(join(bin_dir, f)) and '.ply' in f:
                        self.drives[drive].append(f.split('.ply')[0])
                self.drives[drive] = sorted(self.drives[drive])

    def get_frame_names(self):
        """
........Get all the frame names
........"""

        # return ",".join(self.frame_names)

        return str(self.drives)

    def get_pointcloud(
        self,
        drivename,
        fname,
        dtype=str,
        ground_removed=False,
        ):
        """
........Gets point cloud as list of floats

........Input:
........- fname: Frame name. Can have file extension. 

........Returns a string of comma-separated floats. The number of floats 
........is 4N, where N is the number of points in the point cloud. 
........Each point is represented by 4 numbers - the x, y, z coordinates 
........as well as the intensity.
........"""

        bin_dir = join(self.DATASET_DIR, drivename, self.INPUT_BIN_DIR)
        filename = join(bin_dir, fname.split('.')[0] + '.ply')
        pc = PyntCloud.from_file(filename)
        data = pc.points.to_numpy()[:, :4]
        data[np.isnan(data)] = .0
#         if ground_removed:
#             filename = join(self.DATASET_DIR, drivename,
#                             self.GROUND_REMOVED_DIR, fname.split('.'
#                             )[0] + '.bin')
#             data = np.fromfile(filename, dtype=np.float32)
        if dtype == str:
            data = data.flatten(order='C').tolist()
            data_str = ','.join([str(x) for x in data])
            return data_str
        else:
            if ground_removed:
                return data.reshape((-1, 4))
            else:
                return data.reshape((-1, 4))[:, :3]

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


