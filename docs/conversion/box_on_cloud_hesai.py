import numpy as np
import pandas as pd
import json
import argparse
import os
from pyntcloud import PyntCloud
import mayavi.mlab as mlab

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--json', type=str, default="Guest.000.json", help="specify the annotation file")
parser.add_argument('--data', type=str, default="000.bin", help="specify point cloud file")
args = parser.parse_args()

class kitti_box:
    def __init__(self, params):
        self.id    = "Box ID: " + str(params[0])
        self.x     = params[2]
        self.y     = params[1]
        self.z     = params[3]
        self.theta = np.pi/2 - params[4]
        self.w     = params[5]
        self.l     = params[6]
        self.h     = params[7]

    def convert_to_8corners(self):

        rot = np.array([[ np.cos(self.theta),  -np.sin(self.theta),                  0],
                        [ np.sin(self.theta),   np.cos(self.theta),                  0],
                        [                  0,                    0,                  1]])
        x_corners = [ -self.l/2,  -self.l/2,  self.l/2,  self.l/2,
                      -self.l/2,  -self.l/2,  self.l/2,  self.l/2]
        y_corners = [  self.w/2,  -self.w/2, -self.w/2,  self.w/2,
                       self.w/2,  -self.w/2, -self.w/2,  self.w/2]
        z_corners = [  self.h/2,   self.h/2,  self.h/2,  self.h/2,
                      -self.h/2,  -self.h/2, -self.h/2, -self.h/2]
        
        corners_3d = np.dot(rot, np.vstack([x_corners, y_corners, z_corners]))
        
        corners_3d[0, :] = corners_3d[0, :] + self.x
        corners_3d[1, :] = corners_3d[1, :] + self.y
        corners_3d[2, :] = corners_3d[2, :] + self.z
        return np.transpose(corners_3d)

def read_annotation(bboxes, annotation_path):
    json_data = None
    with open(annotation_path) as annotation:
        json_data = json.load(annotation)
    box_list = json_data["frame"]["bounding_boxes"]
    idx = 0
    for box in box_list:
        box_obj = kitti_box([idx, box["kitti_x"], box["kitti_y"], box["kitti_z"],
                            box["kitti_theta"], box["kitti_l"], box["kitti_w"], box["kitti_h"]])
        bboxes.append(box_obj)
        idx += 1

def draw_points(fig):
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], pc[:, 3], color=None, mode='point', 
                  colormap = 'autumn', scale_factor=1, figure=fig)

def draw_boxes(bboxes, fig):
    for box_obj in bboxes:
        corners = box_obj.convert_to_8corners()
        color = (0, 1, 0)
        line_width = 1
        for i in range(4):
            start, end = i, (i+1)%4
            for (start, end) in [(i, (i+1)%4), (i+4, (i+1)%4+4), (i, i+4)]:
                mlab.plot3d([corners[start, 0], corners[end, 0]],
                            [corners[start, 1], corners[end, 1]],
                            [corners[start, 2], corners[end, 2]],
                            color=color, tube_radius=None, line_width=line_width, figure=fig)
        mlab.text3d(box_obj.x, box_obj.y, box_obj.z, box_obj.id, color=color, line_width=line_width, scale=0.5, figure=fig)


if __name__ == "__main__":

    pc_path = args.data
    annotation_path = args.json
    #pc = PyntCloud.from_file(pc_path).points.to_numpy()
    pc = np.fromfile(pc_path, np.float32).reshape(-1, 4)
    bboxes = []
    read_annotation(bboxes, annotation_path)

    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1600, 1000))
    draw_points(fig)
    draw_boxes(bboxes, fig)

    mlab.show()
    
