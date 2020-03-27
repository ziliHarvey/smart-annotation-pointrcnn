# Docs

## Overview
<p align="center"><img src="https://github.com/ziliHarvey/smart-annotation-pointrcnn/blob/master/docs/imgs/framework.png" width=50% height=50% align="center"></p>

## Usage

### Data Preparation
Current tool supports .bin, .ply and .pcd LiDAR point clouds. To annotate on your own data, place all the files under [test_loader](https://github.com/ziliHarvey/smart-annotation-pointrcnn/tree/master/app/test_dataset/0_drive_0064_sync/sample/argoverse/lidar).

### Supported Operations
(1) Check "Fully Automated Bbox" checkbox to enable fully automatic annotation on the data.  
(2) Hold "A" key and click anywhere, a bbox will be generated using DBSCAN.  
(3) Hold "Ctrl" key and drag to manually annotate the bounding box.
(4) Backspace to delete box.  
(5) Press "R" key to visualize all inbox points in green and outside in red in the panel.  
(6) To save annotation, simply go to the next frame, or click "Save" button.

### Annotation Conversion
