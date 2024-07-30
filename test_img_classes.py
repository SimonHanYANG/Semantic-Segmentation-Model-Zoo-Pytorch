from PIL import Image
import numpy as np
import os

def print_unique_pixel_values(image_path):
    # 打开图像
    image = Image.open(image_path)
    
    # 将图像转换为numpy数组
    image_array = np.array(image)
    
    # 获取图像中所有唯一像素值
    unique_pixel_values = np.unique(image_array)
    
    # 打印唯一像素值
    print(f"Only pixel values are: {unique_pixel_values}")
    
    
def is_single_channel(image_path):
    try:
        # 打开图像
        image = Image.open(image_path)
        # 获取图像模式
        mode = image.mode
        # 判断是否为单通道
        return mode == 'L'
    except Exception as e:
        print(f"无法打开图像 {image_path}: {e}")
        return False
    
    
def check_images_in_directory(directory_path):
    # 获取目录下所有文件
    files = os.listdir(directory_path)
    for file in files:
        file_path = os.path.join(directory_path, file)
        if os.path.isfile(file_path):
            is_single = is_single_channel(file_path)
            print(f"{file}: {'YES' if is_single else 'NO'} is single channel image.")
            
            print("\n")
            
            print_unique_pixel_values(file_path)
            

if __name__ == "__main__":
    # directory_path = '/223010087/SimonWorkspace/SemanticSegmentation/semantic-segmentation-pytorch/segmentation/test/dataset/cityspaces/labeled/train/'  # 替换为你的目录路径
    # directory_path = "/223010087/SimonWorkspace/SemanticSegmentation/semantic-segmentation-pytorch/data/cityscapes/gtFine_labels_trainvaltest/test/berlin"
    directory_path = "/223010087/SimonWorkspace/SemanticSegmentation/semantic-segmentation-pytorch/data/sperm/spermselectionv2_no114/combined_all_masks2one_5classes/"
    check_images_in_directory(directory_path)
    