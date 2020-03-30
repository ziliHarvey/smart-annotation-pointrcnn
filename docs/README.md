# Docs

## Overview
<p align="center"><img src="https://github.com/ziliHarvey/smart-annotation-pointrcnn/blob/master/docs/imgs/framework.png" width=70% height=70% align="center"></p>  
SANE-PointRCNN is using Flask framework, with plain js(most importantly Three.js), HTML and CSS, and backend in python (most importantly PointRCNN model for predicting bounding boxes automatically). The communication is throught AJAX back and forth using JQuery.

## Usage

### Data Preparation
Current tool supports **.bin**, **.ply** and **.pcd** LiDAR point clouds. To annotate on your own data, place all the files under [test_loader](https://github.com/ziliHarvey/smart-annotation-pointrcnn/tree/master/app/test_dataset/0_drive_0064_sync/sample/argoverse/lidar).

### Supported Operations
(1) Check **Fully Automated Bbox** checkbox to enable fully automatic annotation on the data.  
(2) Hold **A** key and click anywhere, a bbox will be generated using DBSCAN.  
(3) Hold **Ctrl** key and drag to manually annotate the bounding box.
(4) **Backspace** to delete box.  
(5) Press **R** key to visualize all inbox points in green and outside in red in the panel.  
(6) To save annotation, simply go to the next frame, or click **Save** button.

### Annotation Conversion
<p align="center"><img src="https://github.com/ziliHarvey/smart-annotation-pointrcnn/blob/master/docs/imgs/visualize_kitti_format.png" width=70% height=70% align="center"></p>

After annotation, extract json files from app/output. The following script is an example of converting it to KITTI format. For KITTI format, each line represent a vehicle's **tracking ID**, **x** (x coordinate of box center), **y**(y coordinate of box center), **z**(z coordinate of box center), **theta**(heading angle), **w**(width), **l**(length) and **h**(height).   

```
python sane_to_kitti.py --json Guest.000.json --fname 000.txt
```

To visualize 3d bounding boxes with point cloud, here are 2 quick offline scripts.  

```
python visualize_from_json_hesai.py --json Guest.000.json --data 000.bin
python visualize_from_kitti_hesai.py --fname 000.txt --data 000.bin
```
[utils.py](https://github.com/ziliHarvey/smart-annotation-pointrcnn/blob/master/docs/conversion/utils.py) provides an example of computing 8 corners coordinate from KITTI-formatted txt files.
