import os
import sys
import numpy as np
import os.path
import time
import random


pythonApp = " CUDA_VISIBLE_DEVICES=0 python3.6 "
script_axcrf = " PointCNN/predicting_point_segmentation.py " #"predicting_point_segmentation_with_axcrf.py  "
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
    # ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

    # drivename, fname = filename.split("/")    
    # if os.path.isfile( os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+"axcrf.bin") ) == False :

    #     os.system(" killall python3.6 & ")
            
    #     os.system(pythonApp+ script_axcrf + "--retrieve_whole_files=0 --filename={}".format(filename))
        
    #     os.system(pythonApp+ script_axcrf + "--retrieve_whole_files=1 --filename={}".format(filename)+ " &")

    # bounded_indices = np.fromfile(
    #                     os.path.join(ROOT_DIR, "PointCNN/output/"+drivename+"_"+fname+"axcrf.bin"),
    #                     dtype=np.int)
    # #os.system("rm {}/PointCNN/output/"+drivename+"_"+fname+".bin".format(ROOT_DIR))

    # return bounded_indices.tolist()
    # # os.system("rm classify/bounding_boxes/*.json")
    return None

def get_pointcnn_labels(filename, settingsControls, ground_removed=False, foreground_only=True):
    
    
    # print("please wait....", filename)

    # ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

    # drivename, fname = filename.split("/")    

    
    
    
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
        
   

    # if(foreground_only):
    #     bounded_indices =   (bounded_indices > 0 ).nonzero()[0]

    #     return bounded_indices.tolist()
    # else:
    #     return bounded_indices
    return None

