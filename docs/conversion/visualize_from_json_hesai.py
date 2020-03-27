"""
@author: Zi Li
@Email: zili@andrew.cmu.edu

This file renders KITTI-formated annotation bounding boxes on point cloud.
It works on Hesai Point cloud data cllected for Traffic21 Dataset.
"""
import numpy as np
import argparse
import mayavi.mlab as mlab
from utils import kitti_box, read_annotation_json, draw_points, draw_boxes

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--json', type=str, default="Guest.000.json", help="specify the annotation file")
parser.add_argument('--data', type=str, default="000.bin", help="specify point cloud file")
args = parser.parse_args()

def draw_boxes_on_points(json, data):
    pc = np.fromfile(data, np.float32).reshape(-1, 4)
    bboxes = []
    read_annotation_json(bboxes, json)
    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1600, 1000))
    draw_points(fig, pc)
    draw_boxes(fig, bboxes)
    mlab.show()

if __name__ == "__main__":
    draw_boxes_on_points(args.json, args.data)
