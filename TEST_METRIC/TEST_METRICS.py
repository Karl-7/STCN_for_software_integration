import os
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.getcwd())
from skimage import io
import TEST_METRIC.METRICS as METRICS
import sys



def test_metric_logger(label_path,test_folder,result_path,logger_path,save_path): 
    #(要有标答和学生作答才能比较哪里不同)  
    #label_path= original label folder; e.g. original files/labels/video1(标答地址)  
    #result_path=predicted label folder; e.g. test_results/labels/video1(学生作答地址)
    #save_path=save path; e.g. metric_result/video1(保存地址，保存对比结果，通过对比结能看出来学生的答案有多标准)
    
    # test_folder = '/root/data/videoanno/self_dataset_LVLAN copy/trainval/JPEGImages/video2/'######## pred原图
    # test_path = '/root/data/videoanno/self_dataset_LVLAN copy/results/video2/'

    label_path = Path(label_path)
    npy_temporal_path='/root/data/videoanno/npy_temporal_path'
    os.makedirs(npy_temporal_path, exist_ok=True)#true就是如果文件夹存在就不创建，如果不存在就创建

######### water

    test_path = Path(result_path)
 
    # save_path = '/root/data/videoanno/self_dataset_LVLAN copy/results/metric_result/video2/'
    
    result_save_path = Path(save_path)
#         result_save_path = Path(data_onlytest_path,"split_test_little-samewidth",'result-cut',test_folder)
    result_save_path.mkdir(parents=True, exist_ok=True)
    
    # logger = Logger(True, Path(result_save_path, "log.txt"))
    # # 记录日志
    # logger.log_print("label: {}, test:{}".format(label_path, test_path))
    
    # 推理vi值
#         m_test_files = ['arrayscan0-SE_19_11_1_1.png', 'arrayscan0-SE_9_2_1_2.png', 'arrayscan0-SE_14_17_1_2.png', 'arrayscan0-SE_9_2_2_2.png', 'arrayscan0-SE_14_17_1_1.png', 'arrayscan0-SE_9_2_1_1.png', 'arrayscan0-SE_19_11_1_2.png', 'arrayscan0-SE_9_2_2_1.png', 'arrayscan0-SE_4_14_1_2.png', 'arrayscan0-SE_4_14_1_1.png']
    m_test_files = sorted(os.listdir(test_path))
    # print(m_test_files)
    # 删减之后的test
#         m_test_files = ['arrayscan0-SE_19_11_1_1.png', 'arrayscan0-SE_9_2_1_2.png', 'arrayscan0-SE_9_2_2_2.png', 'arrayscan0-SE_9_2_1_1.png', 'arrayscan0-SE_19_11_1_2.png', 'arrayscan0-SE_9_2_2_1.png']
    
    os.makedirs(os.path.dirname(logger_path), exist_ok=True)
    # # Redirect print statements to a file
    # sys.stdout = open(logger_path, 'w')
    logger=open(logger_path,'w')
    print("测试文件夹",test_folder,"\n目录",m_test_files)
    logger.write("测试文件夹"+test_folder+"\n目录"+str(m_test_files)+"\n")
    total_vi_metric = 0.0
    total_ARI_metric = 0.0
    total_MAP_metric = 0.0
    abs_GRAIN_metric = 0.0
    rel_GRAIN_metric = 0.0
    merge_error = 0.0
    split_error = 0.0
    for filename in m_test_files:
        if 'checkpoints' in filename:
            continue
        # logger.log_print("img_name: {}----------------------------------".format(filename))
        # 加载label
        label = cv2.imread(os.path.join(label_path, filename))
        # label[label > 0] = 1
        # a = filename[0:3]+".npy"
        # print("a = ", a)


        label_replace = os.path.join(npy_temporal_path, filename.replace("png", "npy"))
        # 保存.npy文件
        np.save(label_replace, np.array(label))
        # 读取.npy文件
        label = np.load(label_replace, allow_pickle=True)
        print("filename",filename)
        # logger.write("filename"+filename+"\n")
        print("label.shape",label.shape)
        # logger.write("label.shape"+str(label.shape)+"\n")
        label = label[:,:,0]
        # label = np.load(str(os.path.join(label_path, a)))
        label[label > 0] = 1
        # label = cv2.dilate(label, None, iterations=3)
        # 加载pred
        test_img = cv2.imread(os.path.join(test_path, filename), 0)
        test_img[test_img > 0] = 1
        # test_img = cv2.dilate(test_img, None, iterations=3)


        # 因为边界标注效果不好，所以将边界向内cut10再进行测试
        a, b = label.shape
        step = 10
        test_img = test_img[step:a-step, step:b-step]
        label = label[step:a-step, step:b-step]
        print("裁剪后大小", test_img.shape)
        print("裁剪后label大小", label.shape)
        # logger.write("裁剪后大小"+str(test_img.shape)+"\n")
        # logger.write("裁剪后label大小"+str(label.shape)+"\n")

        # test_img = np.load(str(os.path.join(test_path, filename.replace("png", "npy"))))
        # test_img[test_img > 0] = 1

        # output = cv2.imread(os.path.join(data_test_path, img), 0)
    ################## 测试图片参数 #########################
        # 分别测试图像的vi，me，se
        temp_merge_error, temp_split_error, temp_vi_metric = METRICS.get_vi(test_img, label)
        # temp_vi_metric = metrics.get_metric('vi', test_img, label)
        total_vi_metric += temp_vi_metric
        merge_error += temp_merge_error
        split_error += temp_split_error

        # 把merge error存在名字里面，方便之后排序
        temp_merge_error = format(temp_merge_error, '.6f')
        # filename = str(temp_merge_error)+"-"+filename
        print("temp_merge_error",temp_merge_error)
        # logger.write("temp_merge_error"+str(temp_merge_error)+"\n")

        print("测试晶粒的ARI和MAP 值")
        # logger.write("测试晶粒的ARI和MAP 值"+"\n")
        temp_ARI_metric = METRICS.get_metric('ARI', ~test_img, ~label)
        # temp_ARI_metric = metrics.get_metric('ARI', test_img, label)
        total_ARI_metric += temp_ARI_metric

        temp_MAP_metric = METRICS.get_metric('MAP', ~test_img, ~label)
        # temp_MAP_metric = metrics.get_metric('MAP', test_img, label)
        total_MAP_metric += temp_MAP_metric
        # 晶粒度-计算值
        temp_GRAIN_metric = METRICS.get_metric('GRAIN', test_img, test_img)
        label_GRAIN_metric = METRICS.get_metric('GRAIN', label, label)
        print("temp_GRAIN_metric:", temp_GRAIN_metric)
        print("label_GRAIN_metric:", label_GRAIN_metric)
        # logger.write("temp_GRAIN_metric:"+str(temp_GRAIN_metric)+"\n")
        # logger.write("label_GRAIN_metric:"+str(label_GRAIN_metric)+"\n")
        # 晶粒度-绝对误差
        abs_GRAIN_metric += abs(temp_GRAIN_metric-label_GRAIN_metric)
        # 晶粒度-相对误差(更改相对误差计算为，|gt-pred|\gt
        rel_GRAIN_metric += abs((label_GRAIN_metric-temp_GRAIN_metric))/label_GRAIN_metric

        a, b = test_img.shape
        # print(img)
        # 展示图像
        unit_pic = np.zeros((a, b, 3))

        # 输出图像效果
        unit_pic[(label == 1) & (test_img == 1)] = [255, 255, 255] # 白色为双方重合的，正确的
        unit_pic[(label == 1) & (test_img != 1)] = [255, 0, 0] # 红色为pred多预测的
        unit_pic[(label != 1) & (test_img == 1)] = [0, 144, 255] # 蓝色为没预测对的label
        print(unit_pic.shape)
        # logger.write("unit_pic.shape="+str(unit_pic.shape)+"\n")
        plt.figure(figsize=(20, 20))
        plt.subplot(1, 1, 1), plt.imshow(unit_pic), plt.title(test_folder+filename), plt.axis("off")
        plt.savefig(str(Path(result_save_path, filename)))
        # plt.show()

        # # print(result_save_path)
        # # cv2.imwrite(str(result_save_path)+"/"+filename, unit_pic)
        # unit_pic = unit_pic.astype(np.uint8)
        # io.imsave(str(result_save_path)+"/"+filename, unit_pic)
    #
    ######### 晶粒参数
        # logger.log_print("vi:{}".format(temp_vi_metric))
        # logger.log_print('ARI:{}'.format(temp_ARI_metric))
        # logger.log_print('MAP:{}'.format(temp_MAP_metric))
        # logger.log_print('abs-GRAIN:{}'.format(abs_GRAIN_metric))
        # logger.log_print('rel-GRAIN:{}'.format(rel_GRAIN_metric))

    # final_vi = total_vi_metric/len(m_test_files)
    # final_ari = total_ARI_metric / len(m_test_files)
    # final_map = total_MAP_metric / len(m_test_files)
    # final_abs_GRAIN = abs_GRAIN_metric / len(m_test_files)
    # final_rel_GRAIN = rel_GRAIN_metric / len(m_test_files)
    # final_merge_err = merge_error / len(m_test_files)
    # final_split_err = split_error / len(m_test_files)

    # logger.log_print("label: {}, test:{}".format(label_path, test_path))
    # logger.log_print("最终平均结果为：")
    # logger.log_print('vi:{}'.format(total_vi_metric/len(m_test_files)))
    # logger.log_print('ARI:{}'.format(final_ari))
    # logger.log_print('MAP:{}'.format(final_map))
    # logger.log_print('abs-GRAIN:{}'.format(final_abs_GRAIN))
    # logger.log_print('rel-GRAIN:{}'.format(final_rel_GRAIN))
    # logger.log_print('merge-err:{}'.format(final_merge_err))
    # logger.log_print('split-err:{}'.format(final_split_err))
    

    print("最终平均结果为：")
    print('vi', total_vi_metric/len(m_test_files))
    print('ARI', total_ARI_metric / len(m_test_files))
    print('MAP', total_MAP_metric / len(m_test_files))
    print('abs-GRAIN', abs_GRAIN_metric / len(m_test_files))
    print('rel-GRAIN', rel_GRAIN_metric / len(m_test_files))
    print('merge-err', merge_error / len(m_test_files))
    print('split-err', split_error / len(m_test_files))
    # logger.write("最终平均结果为："+"\n")
    # logger.write('vi '+str(total_vi_metric/len(m_test_files))+"\n")
    # logger.write('ARI '+str(total_ARI_metric / len(m_test_files))+"\n")
    # logger.write('MAP '+ str(total_MAP_metric / len(m_test_files))+"\n")
    # logger.write('abs-GRAIN '+str(abs_GRAIN_metric / len(m_test_files))+"\n")
    # logger.write('rel-GRAIN '+str(rel_GRAIN_metric / len(m_test_files))+"\n")
    # logger.write('merge-err '+str(merge_error / len(m_test_files))+"\n")
    # logger.write('split-err '+str(split_error / len(m_test_files))+"\n")
    logger.write("最终平均结果为："+"\n")
    logger.write('vi '+'ARI '+'MAP '+ 'abs-GRAIN '+'rel-GRAIN '+'merge-err '+'split-err '+"\n")
    logger.write(str(total_vi_metric/len(m_test_files))+"\n"+str(total_ARI_metric / len(m_test_files))+"\n")
    logger.write(str(total_MAP_metric / len(m_test_files))+"\n")
    logger.write(str(abs_GRAIN_metric / len(m_test_files))+"\n")
    logger.write(str(rel_GRAIN_metric / len(m_test_files))+"\n")
    logger.write(str(merge_error / len(m_test_files))+"\n")
    logger.write(str(split_error / len(m_test_files))+"\n")
    logger.close()

    # Close the file
    sys.stdout.close()