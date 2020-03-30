# smart-annotation-pointrcnn
SANE-PointRCNN, a browser-based 3D bounding boxes annotation tool assisted by PointRCNN.
<p align="center"><img src="https://github.com/ziliHarvey/smart-annotation-pointrcnn/blob/zili/docs/imgs/demo.png"></p>

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
```
cd app
python app.py
```
Open your browser and then go to http://0.0.0.0:7772. The first time loading will be relatively slow and the rest will be very fast.  
For detailed instructions on annotating your own data, please refer to [Docs](https://github.com/ziliHarvey/smart-annotation-pointrcnn/blob/zili/docs/README.md).

## Progress
- [x] Reorganized the code base by running PointRCNN as backend
- [x] Fully-Automated-Bbox click
- [x] Segmented object points display
- [x] One-click annotation by holding A key and click on the point
- [x] Fix heading angle in boxes display
- [x] Display all LiDAR points with corresponded point labels
- [x] Modify dataLoader to run on the specied file
- [x] Modify dataLoader to run for inference without ground truth
- [x] JSON and KITTI-format conversion and offline visualization
- [ ] remove legacy code and files and clean
- [ ] Tracking, etc.

## Contact
[Zi Li](https://github.com/ziliHarvey)  
[Kartik Sah](https://github.com/Kartik17)
