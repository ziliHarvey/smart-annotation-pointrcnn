from os import system, chdir, getcwd

pythonApp = "python "
script_seg = "eval_rcnn.py --cfg_file cfgs/argo_config_sampling_trainfull.yaml --rcnn_ckpt checkpoint_epoch_40.pth --rpn_ckpt checkpoint_epoch_50.pth --batch_size 1 --eval_mode rcnn --test"

def preprocess():
    print("=============================================")
    print("=============================================")
    print("Processing data begines......................")
    cur_dir = getcwd()
    script_dir = cur_dir + "/PointCNN/tools"
    chdir(script_dir)
    system(pythonApp + script_seg)
    chdir(cur_dir)
    system(pythonApp + "rename.py")
    print("Processing data finishes................ enjoy!")
    print("==============================================")
    print("==============================================")
