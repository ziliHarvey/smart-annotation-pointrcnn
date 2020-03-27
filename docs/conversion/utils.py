"""
@Author: Zi Li
@Email: zili@andrew.cmu.edu

Useful functions/class for format conversion and visualization
"""
import numpy as np
import json
import mayavi.mlab as mlab

class kitti_box:
    def __init__(self, params):
        self.id    = params[0]
        self.x     = params[1]
        self.y     = params[2]
        self.z     = params[3]
        self.theta = params[4]
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

def draw_points(fig, pc):
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], pc[:, 3], color=None, mode='point', 
                  colormap = 'autumn', scale_factor=1, figure=fig)

def draw_boxes(fig, bboxes):
    for box_obj in bboxes:
        corners = box_obj.convert_to_8corners()
        for i in range(4):
            start, end = i, (i+1)%4
            for (start, end) in [(i, (i+1)%4), (i+4, (i+1)%4+4), (i, i+4)]:
                mlab.plot3d([corners[start, 0], corners[end, 0]],
                            [corners[start, 1], corners[end, 1]],
                            [corners[start, 2], corners[end, 2]],
                            color=(0, 1, 0), tube_radius=None, line_width=1, figure=fig)
        mlab.text3d(box_obj.x, box_obj.y, box_obj.z, str(int(box_obj.id)), color=(1, 1, 0), line_width=2, scale=0.8, figure=fig)

def read_annotation_json(bboxes, annotation_path):
    json_data = None
    with open(annotation_path) as annotation:
        json_data = json.load(annotation)
    box_list = json_data["frame"]["bounding_boxes"]
    for box in box_list:
        box_obj = kitti_box([box["kitti_id"], box["kitti_x"], box["kitti_y"], box["kitti_z"],
                            box["kitti_theta"], box["kitti_w"], box["kitti_l"], box["kitti_h"]])
        bboxes.append(box_obj)

def read_annotation_kitti(bboxes, fname):
    lines = np.loadtxt(fname, delimiter=',')
    for line in lines:
        params = []
        box_obj = kitti_box(line)
        bboxes += [box_obj]
        