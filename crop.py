import os
import cv2


def crop_image(image_path, output_dir, crop_size=(255, 255)):
    """
    将图像裁剪成指定大小的小块并保存。

    参数:
    - image_path: 输入图像路径
    - output_dir: 输出裁剪图像保存的目录
    - crop_size: 裁剪的尺寸 (height, width)，默认为 (255, 255)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    h, w, _ = image.shape
    crop_h, crop_w = crop_size

    # 遍历裁剪
    count = 0
    for y in range(0, h, crop_h):
        for x in range(0, w, crop_w):
            # 计算裁剪区域
            crop_img = image[y:y + crop_h, x:x + crop_w]

            # 如果裁剪区域小于目标尺寸，则跳过
            if crop_img.shape[0] != crop_h or crop_img.shape[1] != crop_w:
                continue

            # 保存裁剪图像
            crop_filename = os.path.join(output_dir, f"crop_{count}.png")
            cv2.imwrite(crop_filename, crop_img)
            count += 1
            print(f"保存裁剪图像: {crop_filename}")


# 示例用法
input_image = "/media/dell/disk2/zhoujie/MICCAI23-deeplab_TCT/target_adapt/DeepLab_Abdomen_MR2CT_Adapt_FixMatch_fold4_pseudo/exp_1_time_2024-11-29 20-54-28-nodenosy/visuals/img_pred_pseudo_gt80.png"  # 替换为输入图像的路径
output_directory = "/media/dell/disk2/zhoujie/MICCAI23-deeplab_TCT/target_adapt/DeepLab_Abdomen_MR2CT_Adapt_FixMatch_fold4_pseudo/exp_1_time_2024-11-29 20-54-28-nodenosy/visuals/80"  # 替换为输出目录
crop_image(input_image, output_directory, crop_size=(255, 255))
