'''
Project     : Do not edit
Author       : Zxy
Date         : Do not edit
LastEditTime: 2023-05-29 19:40:26
Descripttion : 
'''
# gt test water xcx
import os
import numpy as np
import cv2
from skimage import io,img_as_ubyte
from PIL import Image

#PATH SHOULD BE LIKE:/root/data/videoanno/self_dataset
def color_confrontation(dataset_folder):
    for folder in os.listdir(os.path.join(dataset_folder, 'results')): 
        if folder.startswith('video') and folder[5:].isdigit():
            #folders should be at somewhere like:dataset/result/, and folders should be:video1,video2......
            img_path = os.path.join(dataset_folder,'trainval/JPEGImages',folder)
            label_path =os.path.join( dataset_folder,'trainval/Annotations',folder)
            pred_path = os.path.join( dataset_folder,'results',folder)
            # pred_path = "./dataset_for_test/label_boundary_uncorrected"
            # water_path = "./dataset_for_test/watered"
            # xcx_path = '/root/data/zhangxinyi/data/gyy-water/dataset_for_test/xcx_boundary'
            
            tar_path = os.path.join( dataset_folder,"results/confrontation_result",folder)
            os.makedirs(tar_path, mode=0o777, exist_ok=True)
            files = os.listdir(label_path)

            for file in files:
                if 'png' not in file:
                    continue
                print(file)
                label = cv2.imread(label_path+"/"+file, 0)
                label[label>0] = 255
                #test_img = cv2.imread(pred_path+"/label_"+file, 0)
                test_img = cv2.imread(pred_path+"/"+file, 0)
                #下面这句话是说
                test_img[test_img>0] = 255
                
                # water_img = cv2.imread(water_path+"/"+file, 0)
                # water_img[water_img>0] = 255
                # xcx = cv2.imread(xcx_path+"/"+file, 0)
                # xcx[xcx>0] = 255

                # label = cv2.cvtColor(label, cv2.COLOR_GRAY2RGB)
                # test_img=cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB)
                # a, b ,c = label.shape
                # unit_pic = np.zeros((a, b, c))
                a, b = label.shape
                unit_pic = np.zeros((a, b, 3))
                # print(unit_pic.shape)
            #     # 两张图片叠图：输出图像效果
                unit_pic[(label == 255) & (test_img == 255)] = [255, 255, 255]  # 白色为双方重合的，正确的
                unit_pic[(label == 255) & (test_img != 255)] = [255, 0, 0]  # 红色为没预测对的label
                unit_pic[(label != 255) & (test_img == 255)] = [0, 0, 255]  # 蓝色为pred多预测的
                
                # xcx和晶界重叠效果
                # unit_pic[(label == 255) & (xcx == 255)] = [255, 255, 255]  # 白色为双方重合的，正确的
                # unit_pic[(label == 255) & (xcx != 255)] = [255, 0, 0]  # 红色为label
                # unit_pic[(label != 255) & (xcx == 255)] = [0, 144, 255]  # 蓝色为xcx
                
                # 将label蒙版到image上

                file = file.replace('.png', '.jpg')
                image = (cv2.imread(img_path+"/"+file, 1)).astype('float')
                # image = (image - np.min(image)) / (np.max(image)-np.min(image))*255.0
                # unit_pic = (unit_pic - 127.5) / 127.5
                
                # image_uint8 = img_as_ubyte(image)
                # io.imsave('image.jpg', image_uint8)
                # unit_pic_uint8 = img_as_ubyte(unit_pic)  # Scale the values to the range of 0 to 255
                # io.imsave(tar_anno_path+"/"+file, unit_pic_uint8)
                
                # print(image.shape)
                # print(image.dtype)
                # print(np.min(image), np.max(image))
                # print(np.min(unit_pic), np.max(unit_pic))
                
                #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            #     print(image.shape)
                # print(unit_pic.shape)
                    
                    # alpha 为第一张图片的透明度
                alpha = 0.9
                # beta 为第二张图片的透明度
                beta = 0.9
                gamma = 0
                # cv2.addWeighted 将原始图片与 mask 融合
                #save_img = cv2.addWeighted(image, alpha, unit_pic, beta, gamma, dtype = cv2.CV_32F)
                save_img = img_as_ubyte(((alpha/2.0 * image + beta/2.0 * unit_pic + gamma)) / 255.0)
                # print("RANGE",np.min(save_img), np.max(save_img))
                # cv2.imwrite(os.path.join(output_fold,'mask_img.jpg'), mask_img)
                # print(np.min(save_img), np.max(save_img))
                save_img = (save_img - np.min(save_img)) / (np.max(save_img)-np.min(save_img))*255.0
                # print("RECTIFIED_RANGE",np.min(save_img), np.max(save_img))
                # 显示
                # # unit_pic[(water_img != 255) & (test_img == 255)] = [255,128,0] # 矫正的区域
                # unit_pic[(water_img != 255) & (label != 255)& (test_img == 255)] = [0, 255, 0]  # 绿色，water进行矫正正确的区域
                
                # unit_pic[(water_img != 255) & (label == 255) & (test_img == 255)] = [218,112,214] # 紫色，water矫正错误区域
                
                # 这是存储普通两图叠加的图像的
                save_img = ((save_img)).astype('uint8')
                
                # show_pic = unit_pic[0:1024,0:1024]
            #     cv2.imwrite(tar_path+"/"+file, unit_pic)
                io.imsave(tar_path+"/"+file, save_img)
                # print(file)