import os
import sys
import numpy as np
import os.path
import time
import random


pythonApp = "python "
script_seg = "  PointCNN/predicting_point_segmentation.py "

def check_succes_sys_call(_command, file_check):
    i = 0
    while os.path.isfile(file_check) == False :
        print(i, os.path.isfile(file_check), file_check)
        os.system(" killall python3.6 & ")
        os.system(_command)  
        if(i>0):
            time.sleep(random.randint(5,30))
        i = i + 1
    return True

def get_pointcnn_labels_axcrf(filename):
    return None

def get_pointcnn_labels(filename, settingsControls, ground_removed=False, foreground_only=True):
    
    
    print("please wait....", filename)

    seg_dir = "PointCNN/output/rcnn/argo_config_sampling_trainfull/eval/epoch_no_number/sample/test_mode/rpn_result/data"

    drivename, fname = filename.split("/")
    seg_file = seg_dir + "/" + fname + ".npy"
    # will add check isfile later 
    bounded_indices = np.load(seg_file).reshape(-1, 5)[:, -1].flatten()
    print(bounded_indices) 
    
    
    # if(settingsControls["WithDenoising"] == False):
    #     postfix = "normal-weights"

    #     if(ground_removed):
            
    #         if os.path.isfile( os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+"ground_removed.bin") ) == False :
    #             check_succes_sys_call(pythonApp+ script_seg + " --ground_removed=1 --retrieve_whole_files=0 --filename={}".format(filename),  os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+"ground_removed.bin") )

    #             os.system(pythonApp+ script_seg + " --ground_removed=1 --retrieve_whole_files=1 --filename={}".format(filename)+ " &")
            
            
    #         bounded_indices = np.fromfile(
    #                         os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+"ground_removed.bin"),
    #                         dtype=np.int)
        
        
    #     else: #Non-Ground Removed
            
    #         if os.path.isfile( os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+postfix+".bin") ) == False :

          
    #             check_succes_sys_call(pythonApp+ script_seg + " --ground_removed=0 --retrieve_whole_files=0 --postfix="+postfix+" --filename={}".format(filename),   os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+postfix+".bin"))
                    

    #             os.system(pythonApp+ script_seg + " --ground_removed=0  --retrieve_whole_files=1 --postfix="+postfix+" --filename={}".format(filename)+ " & ")
                
                
    #         bounded_indices = np.fromfile(
    #             os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+postfix+".bin"),
    #             dtype=np.int)
            
            

    # else: # Denoise
        
    #     postfix = "denoise-weights"
        
    #     print("exist", os.path.isfile( os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+postfix+".bin") ), os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+postfix+".bin"))
        
    #     if os.path.isfile( os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+postfix+".bin") ) == False :

            
    #         check_succes_sys_call(pythonApp+ script_seg + " --ground_removed=0 --retrieve_whole_files=0 --postfix="+postfix+" --filename={}".format(filename), os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+postfix+".bin"))
            
    #         os.system(pythonApp+ script_seg + " --ground_removed=0  --retrieve_whole_files=1 --postfix="+postfix+" --filename={}".format(filename)+ " & ")
            
            
    #     bounded_indices = np.fromfile(
    #                         os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+postfix+".bin"),
    #                         dtype=np.int)
        
    if(foreground_only):
        bounded_indices =   (bounded_indices == 1.0 ).nonzero()[0]
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(bounded_indices.shape)
        return bounded_indices.tolist()
    else:
        return bounded_indices


