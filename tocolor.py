import os
import numpy as np
from PIL import Image

# 设置包含 PNG 图片的文件夹路径
png_folder = '/media/dell/disk2/zhoujie/MICCAI23-deeplab_TCT/prediction_MSCMR_deeplab_test'
output_folder = '/media/dell/disk2/zhoujie/MICCAI23-deeplab_TCT/prediction_MSCMR_deeplab_testtocorlor'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 定义颜色映射
color_mapping = {
    # 50: (163, 178, 181),    # 红色
    # 100: (169, 160, 179),   # 绿色
    # 150: (219, 157, 171),   # 蓝色
    # 200: (244, 127, 116)  # 黄色
    0: (68, 1, 84),
    50: (53, 183, 120),
    100: (48, 103, 141),
    150: (253, 231, 26)

}

# 遍历 PNG 文件
for png_file in os.listdir(png_folder):
    if png_file.endswith('.png'):
        # 加载 PNG 文件
        file_path = os.path.join(png_folder, png_file)
        img = Image.open(file_path)

        # 将图像转换为 numpy 数组
        img_array = np.array(img)

        # 创建一个彩色图像数组
        color_image = np.zeros((*img_array.shape, 3), dtype=np.uint8)

        # 根据像素值映射颜色
        for pixel_value, color in color_mapping.items():
            color_image[img_array == pixel_value] = color

        # 将彩色图像转换为 PIL Image
        color_image_pil = Image.fromarray(color_image)

        # 保存彩色图像
        output_path = os.path.join(output_folder, png_file)
        color_image_pil.save(output_path)

        print(f'Saved: {output_path}')