import os
from PIL import Image
import shutil
from TRAIN_PARAS import train_parameters
#PATH SHOULD BE LIKE:/root/data/videoanno/self_dataset/trainval 
def split_images(folder_path):
    # Get the list of subfolders in the given folder
    # print(str(folder_path))
    train_settings_para=train_parameters()
    if folder_path.endswith("trainval") :
        sub_folders=[subfolder for subfolder in os.listdir(folder_path) if (os.path.isdir(os.path.join(folder_path, subfolder)))]
    elif folder_path.endswith("test-dev"):
        sub_folders = [subfolder for subfolder in os.listdir(folder_path) if (os.path.isdir(os.path.join(folder_path, subfolder)) and subfolder.startswith("video") and not subfolder.endswith("sliced"))]
    # print(sub_folders)#annotations,jpegimages
    for folders in sub_folders:
        current_folder_path=os.path.join(folder_path,folders)#video1,2
        # print("current_folder_path",str(current_folder_path))
        #subfolders=annotations,jpegimages, 
        #sub_folders=video1,2
        sub_folders=[subfolder for subfolder in os.listdir(current_folder_path) if (os.path.isdir(os.path.join(current_folder_path, subfolder)) and not subfolder.endswith("sliced"))]
        for video in sub_folders:
            video_path = os.path.join(current_folder_path, video)
            # print("video_path",str(video_path))
            sliced_path= os.path.join(current_folder_path, f"sliced")
            output_folder = {}
    
            sliced_numbers=['00','01','10','11']
            for i in range(4):
                output_folder[i] = os.path.join(sliced_path, f"{video}-{sliced_numbers[i]}")  # .../test-dev/sliced/video1
                os.makedirs(output_folder[i], exist_ok=True)
                with open(train_settings_para.classes, 'a') as file:
                    file.write(output_folder[i] + '\n')

                # print(f"output_folder[{i}]={output_folder[i]}")

            # Get the list of image files in the subfolder
            image_files = [file for file in os.listdir(video_path) if file.endswith(".png") or file.endswith(".jpg")]

            for image_file in image_files:
                image_path = os.path.join(video_path, image_file)
                image = Image.open(image_path)

                # Get the dimensions of the image
                width, height = image.size

                # Calculate the width and height of each split image
                split_width = width // 2
                split_height = height // 2

                for i in range(2):
                    for j in range(2):
                        # Calculate the coordinates for cropping the image
                        left = j * split_width
                        upper = i * split_height
                        right = left + split_width
                        lower = upper + split_height

                        # Crop the image
                        split_image = image.crop((left, upper, right, lower))
                        # partial_image_path = os.path.join(output_folder, f"{i}{j}")
                        # if not os.path.exists(partial_image_path):
                        #     os.makedirs(partial_image_path, exist_ok=True)
                        # Convert i and j to binary and concatenate them
                        binary_value = f"{i}{j}"
                        # Convert binary_value to decimal
                        decimal_value = int(binary_value, 2)
                        # Use decimal_value in your code
                        # Save the split image
                        if image_file.endswith(".png"):  
                            split_image_path = os.path.join(output_folder[decimal_value], f"{i}{j}_{image_file.split('.')[0]}.png")
                        elif image_file.endswith(".jpg"):
                            split_image_path = os.path.join(output_folder[decimal_value], f"{i}{j}_{image_file.split('.')[0]}.jpg")
                        split_image.save(split_image_path)


# Provide the path to the folder containing the subfolders
            
# #folder_path = "/root/data/videoanno/self_dataset/trainval/Annotations"
# folder_path = "/root/data/videoanno/self_dataset/trainval/JPEGImages"
# split_images(folder_path)
def destroy_temporal_split_folder(folder_path):
    shutil.rmtree(folder_path, ignore_errors=True)
    print(f"Folder {folder_path} deleted")


def re_concate_the_splitted_images(original_path,sliced_path,dest_path):
    #例：original_path="/root/data/videoanno/self_dataset/test-dev/Annotations/original"
    #sliced_path="/root/data/videoanno/self_dataset/results/sliced"
    #dest_path="/root/data/videoanno/self_dataset/results"
    # sliced_path=os.path.join(dest_path, "sliced")
    os.makedirs(sliced_path, exist_ok=True)
    sliced_videos=[subfolder for subfolder in os.listdir(sliced_path) if (os.path.isdir(os.path.join(sliced_path, subfolder)) )]
    print('sliced_videos:',sliced_videos)
    #即sliced_videos=/result/sliced/videt1-00,video1-01,video1-10,video1-11,video2-00,video2-01,video2-10,video2-11.......
    original_videos=[subfolder for subfolder in os.listdir(original_path) if (os.path.isdir(os.path.join(original_path, subfolder))  and subfolder.startswith("video") and not subfolder.endswith("sliced"))]
   
    original_videos.sort(key=lambda x: int(x[5:]))
    print('original_videos:',original_videos)
    #即original_videos=/Annotations/video1,video2
    
    for video in original_videos:
        #在result/创建video1,video2文件夹
        reconcate_folders=os.path.join(dest_path, f"{video}")
        os.makedirs(reconcate_folders, exist_ok=True)

        #current_sliced_videos=['video1-00','video1-01','video1-10','video1-11']
        current_sliced_videos=[subfolder for subfolder in sliced_videos if subfolder.startswith(video)]
        print('current_sliced_videos:',current_sliced_videos)
        #original_images_in_videoXX=['000.jpg','001.jpg','002.jpg'...]
        original_images_in_videoXX=[file for file in os.listdir(os.path.join(original_path,f"{video}")) if file.endswith(".png") or file.endswith(".jpg")]
        # print('original_images_in_videoXX:',original_images_in_videoXX)
        for image in original_images_in_videoXX:#image=000.jpg
            print('image:',image)
            
            current_image_name = image.split(".")[0]
            frame_parts = []
            for sliced_folder in current_sliced_videos:
                #sliced_folder_path=results/sliced/video1-00
                # print('sliced_folder',sliced_folder)
                sliced_folder_path = os.path.join(sliced_path, sliced_folder)
                # print('sliced_folder_path',sliced_folder_path)
                for file in os.listdir(sliced_folder_path):
                    
                    if (file.endswith(".png") or file.endswith(".jpg")) and (file.split(".")[0].split("_")[1] == current_image_name):
                        frame_parts.append(os.path.join(sliced_folder_path, file))
                        # print('frame_parts:',frame_parts)
                        break
            #开始拼接：
            # Combine the four images into one
                
            split_width= Image.open(frame_parts[0]).size[0]
            split_height= Image.open(frame_parts[0]).size[1]
            combined_image = Image.new("RGB", (split_width * 2, split_height * 2))
            for i in range(2):
                for j in range(2):
                    frame_part = frame_parts[i * 2 + j]
                    part_image = Image.open(frame_part)
                    combined_image.paste(part_image, (j * split_width, i * split_height))

            # Save the combined image
            combined_image.save(os.path.join(reconcate_folders, image))