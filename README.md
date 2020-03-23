# smart-annotation-pointrcnn (IN PROGRESS...)
## Environment
Tested on Debian 9.9, Cuda: 10.0, Python: 3.6, Pytorch: 1.2.0 with Anaconda

## Installation
```
git clone --recursive https://github.com/ziliHarvey/smart-annotation-pointrcnn.git
cd app/PointCNN/
sh build_and_install.sh
```
Also install all necessary libraries using conda, such as flask, easydict,tqdm, tensorboardX, etc.

## Usage
**Notice that currently it only works on [the given data](https://github.com/ziliHarvey/smart-annotation-pointrcnn/tree/master/app/test_dataset/0_drive_0064_sync/sample/argoverse/lidar)**
```
cd app
python app.py
```
And open your browser and go to http://0.0.0.0:7772. The first time you load will take relatively longer (around 1 min)
and the rest will be very fast.

### Progress
- [x] Reorganized the code base by running PointRCNN as backend
- [x] Fully-Automated-Bbox click
- [x] Segmented object points display
- [x] One-click annotation by holding A key and click on the point
- [x] Fix heading angle in boxes display
- [x] Display all LiDAR points with corresponded point labels
- [x] Modify dataLoader to run on the specied file
- [ ] remove legacy code and files and clean
- [ ] Modify dataLoader to run for inference without ground truth
- [ ] Tracking, etc.
