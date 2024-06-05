
import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/TEST_METRIC')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eval_generic import eval_generic

import datetime
from TEST_METRIC.TEST_METRICS import test_metric_logger
import glob
from TRAIN_PARAS import train_parameters
import sys
from COLOR_SHOW_CONFRONTATION import color_confrontation


#python /root/data/videoanno/STCN-main/TRAIN_SETTINGS.py
class train_test(train_parameters):
    def __init__(self,dataset_address):
        super().__init__()
        self.data_path=dataset_address
        self.output=os.path.join(self.data_path,'results')
        self.videos_number_generate()
        self.classes=os.path.join(self.STCN_path,'util/your_video_list.txt')
        self.test_set=os.path.join(self.data_path,"test-dev")
    def run_train(self):
        # START AUTO TRAIN
        os.system(f"""CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 9842 --nproc_per_node=2  {self.train_path} --id {self.train_stage} --load_network {self.load_network} --stage 3 """)
        

    def run_test(self):
        # eval_generic(original_path=os.path.join(self.data_path,'test-dev/Annotations/original'),
        #         sliced_path=os.path.join(self.data_path,'/results/sliced'), 
        #         dest_path=os.path.join(self.data_path,'results'),
        #         para=self)
        eval_generic(original_path=os.path.join(self.data_path,'test-dev/Annotations/original'),
                sliced_path=os.path.join(self.data_path,'/results/non_sliced'), 
                dest_path=os.path.join(self.data_path,'results'),
                para=self)
    
    def confrontation(self):
        current_time=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name=  "STCN_results"+str(current_time)  
        logger_name='log_video1'
        # color_confrontation('/root/data/videoanno/STCN-main/self_dataset_LVLAN/LVLAN_FOR_COLOR')

        test_metric_logger(label_path=os.path.join(self.test_set,"test/original_labels/video1"), 
                           test_folder=os.path.join(self.test_set,'test/images/video1'), 
                           result_path=os.path.join(self.data_path,folder_name,"label_results/video1"),
                           save_path=os.path.join(self.data_path,folder_name,'METRIC_PICS/video1'),
                           logger_path=os.path.join(self.data_path,folder_name,logger_name))
        

    def videos_number_generate(self):     
        folder_path = os.path.join(self.data_path, 'test-dev/JPEGImages')
        output_file = os.path.join(self.STCN_path,'util/your_video_list.txt')
        # Clear the contents of the output file
        with open(output_file, 'w') as file:
            file.truncate(0)
        # Get all file names in the folder
        file_names = os.listdir(folder_path)
        # Append file names to the output file
        with open(output_file, 'a') as file:
            for file_name in file_names:
                for ender in ['','00','01','10','11']:
                    file.write(file_name + ender+'\n')
if __name__=="__main__":
    torch.cuda.empty_cache()
    # train_test().run_train()
    # train_test().run_test()
    train_test().confrontation()