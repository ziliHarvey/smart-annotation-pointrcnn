"""
@author: Zi Li
@Email: zili@andrew.cmu.edu

This file shows how to render KITTI-formated annotation bounding boxes on point cloud.
It works on Hesai Point cloud data cllected for Traffic21 Dataset.
"""
import numpy as np
import argparse
import mayavi.mlab as mlab
from utils import kitti_box, read_annotation_kitti, draw_boxes, draw_points

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--fname', type=str, default="000.txt", help="specify name of the KITTI-formatted file")
parser.add_argument('--data', type=str, default="000.bin", help="specify point cloud file")
args = parser.parse_args()

def draw_boxes_on_points(fname, data):
    pc = np.fromfile(data, np.float32).reshape(-1, 4)
    bboxes = []
    read_annotation_kitti(bboxes, fname)
    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1600, 1000))
    draw_points(fig, pc)
    draw_boxes(fig, bboxes)
    mlab.show()

if __name__ == "__main__":
    draw_boxes_on_points(args.fname, args.data)
