import os
import numpy as np
from skimage import io

# 设置包含 npz 文件的文件夹路径
npz_folder = '/media/dell/disk2/zhoujie/MICCAI23-deeplab_TCT/Abdome_data/CT_fold4/test'
output_folder = '/media/dell/disk2/zhoujie/MICCAI23-deeplab_TCT/Abdome_data/save_img'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍历 npz 文件
for npz_file in os.listdir(npz_folder):
    if npz_file.endswith('.npz'):
        # 加载 npz 文件
        file_path = os.path.join(npz_folder, npz_file)
        data = np.load(file_path)

        # 提取 image 和 label
        image = data['image']
        label = data['label']  # 如果需要 label，可以保存或处理；否则可以忽略



        image_min = image.min()
        image_max = image.max()
        normalized_image = (image - image_min) / (image_max - image_min)  # 归一化


        # 将归一化后的 image 转换为 uint8 格式
        image_uint8 = (normalized_image * 255).astype(np.uint8)
        # 如果是3通道图像，取其中一个通道作为灰度图
        if image_uint8.shape[2] == 3:  # 检查是否为3通道
            image_uint8 = np.mean(image_uint8, axis=2).astype(np.uint8)  # 取平均值作为灰度图

        # 构建保存的 PNG 文件名
        image_name = os.path.splitext(npz_file)[0] + '.png'
        image_path = os.path.join(output_folder, image_name)

        # 将 image 保存为 PNG 文件
        io.imsave(image_path, image_uint8)

        print(f'Saved: {image_path}')