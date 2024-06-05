import os
import torch
import glob
#python /root/data/videoanno/STCN-main/TRAIN_SETTINGS.py
class train_parameters():
    def __init__(self):
        # self.model='Mar19_13.33.26_retrain_s03_1500.pth'
        self.project_path = os.path.abspath(__file__)#'/root/data/videoanno/STCN-main/TRAIN_SETTINGS.py'
        # print("self.project_path",self.project_path)
        self.STCN_path = os.path.dirname(self.project_path)#'/root/data/videoanno/STCN-main'
        self.train_path=os.path.join(self.STCN_path,"train.py")
        # print("self.STCN_path",self.STCN_path)
        self.local_path = os.path.dirname(self.STCN_path)#'/root/data/videoanno'
        # print("self.local_path",self.local_path)

        # self.data_path = os.path.join(self.local_path,'datasets')

        # print("self.data_path",self.data_path)
        #self.output=os.path.expanduser('~/data/videoanno/self_dataset_EDGE_copy/results')
        


        
        self.top=5
        self.amp_off=False
        self.mem_every=5
        # self.split_images=True
        self.separate_flag=True
        self.save_model_at=os.path.join(self.STCN_path,'Model_saves')
        # self.load_network=os.path.join(self.STCN_path,'Model_saves_lvlanjiejing/May21_22.27.57_retrain_s03/May21_22.27.57_retrain_s03_1500.pth')
        # Find all .pth files in the directory
        # Find all directories in Model_saves that contain "retrain" in their names
        retrain_dirs = [dir for dir in os.listdir(os.path.join(self.STCN_path, 'Model_saves')) if 'retrain' in dir]
        
        if retrain_dirs:
            # Sort the directories by modification time in descending order
            retrain_dirs.sort(key=lambda dir: os.path.getmtime(os.path.join(self.STCN_path, 'Model_saves', dir)), reverse=True)
            # Get the path of the latest retrain directory
            latest_retrain_dir = os.path.join(self.STCN_path, 'Model_saves', retrain_dirs[0])
            # Find all .pth files in the latest retrain directory
            pth_files = glob.glob(os.path.join(latest_retrain_dir, '*_1500.pth'))
            if pth_files:
            # Sort the files by modification time in descending order
                pth_files.sort(key=os.path.getmtime, reverse=True)
            # Get the path of the latest .pth file
                latest_pth_file = pth_files[0]
                self.load_network = latest_pth_file
            else:
            # Handle the case when no .pth files are found in the latest retrain directory
                print("No .pth files found in the latest retrain directory.")
                self.load_network = os.path.join(self.STCN_path,'\saves\stcn_s0.pth')
        else:
        # Handle the case when no retrain directories are found
            print("No retrain directories found.")
            self.load_network = os.path.join(self.STCN_path,'\saves\stcn_s0.pth')
            
        self.train_stage="retrain_s03"



    
            
