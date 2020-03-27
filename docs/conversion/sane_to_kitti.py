"""
@author: Zi Li
@Email: zili@andrew.cmu.edu

This file converts SANE-PointRCNN JSON annotations to KITTI-formatted annotations.
"""
import numpy as np
import argparse
from utils import kitti_box, read_annotation_json

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--json', type=str, default="Guest.000.json", help="specify the annotation file")
parser.add_argument('--fname', type=str, default="000.txt", help="specify name of the KITTI-formatted file")
args = parser.parse_args()

def json_to_kitti(json, fname):
    bboxes = []
    read_annotation_json(bboxes, json)
    with open(fname, 'w') as f:
        for bbox in bboxes:
            print('%d, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' %
                 (bbox.id, bbox.x, bbox.y, bbox.z, bbox.theta, bbox.w, bbox.l, bbox.h), file=f)       

if __name__ == "__main__":
    json_to_kitti(args.json, args.fname)