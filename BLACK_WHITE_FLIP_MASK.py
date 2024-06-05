import os
from PIL import Image,ImageFilter
# from TRAIN_SETTINGS import train_parameters as train_parameters
def flip_black_white(folder_path):
    subfolders = [subfolder for subfolder in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subfolder))]
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join(subfolder_path, filename)
                image = Image.open(image_path)
                image = image.convert("L")  # Convert to grayscale
                inverted_image = Image.eval(image, lambda x: 255 - x)  # Invert black and white pixels
                # Apply erosion to the inverted image
                eroded_image = inverted_image.filter(ImageFilter.MinFilter(size=5))
                # Save the eroded image
                eroded_image.save(image_path)
                print(f"Eroded image saved: {filename}")
                print(f"Flipped black and white pixels in {filename}")

# Usage example\
# para=train_parameters()
# folder_path = os.join(para.data_path,"trainval/Annotations")

# flip_black_white(folder_path)