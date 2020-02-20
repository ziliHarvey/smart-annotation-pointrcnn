import numpy as np
import os
import glob
'''
file_dir = "test_dataset/0_drive_0064_sync/sample/argoverse/lidar"
bbox_dir = "PointCNN/output/rcnn/argo_config_sampling_trainfull/eval/epoch_no_number/sample/test_mode/final_result/data"
seg_dir = "PointCNN/output/rcnn/argo_config_sampling_trainfull/eval/epoch_no_number/sample/test_mode/rpn_result/data"

# a script for converting PointRCNN output files into raw files naming convention
# file_dir: PC_315978406019574000.ply
# bbox_dir: 000000.txt => PC_315978406019574000.txt
# seg_dir: 000000.npy => PC_315978406019574000.npy
# this will be replaced later by PointRCNN automatically generating output with raw file name

files = [os.path.splitext(os.path.basename(path))[0] for path in sorted(glob.glob(file_dir + "/*.ply"))]
bboxes = [os.path.splitext(os.path.basename(path))[0] for path in sorted(glob.glob(bbox_dir + "/*.txt"))]
segs = [os.path.splitext(os.path.basename(path))[0] for path in sorted(glob.glob(seg_dir + "/*.npy"))]
for i in range(len(files)):
    bbox_src = bbox_dir + "/" + bboxes[i] + ".txt"
    bbox_det = bbox_dir + "/" + files[i] + ".txt"
    cmd_bbox = "mv " + bbox_src + " " + bbox_det
    os.system(cmd_bbox)
    seg_src = seg_dir + "/" + segs[i] + ".npy"
    seg_det = seg_dir + "/" + files[i] + ".npy"
    cmd_seg = "mv " + seg_src + " " + seg_det
    os.system(cmd_seg)
'''