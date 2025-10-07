from PIL import Image

# 打开图片
image = Image.open("/media/dell/disk2/zhoujie/MICCAI23-deeplab_TCT/定量评估彩图/pre-adapt/0024_8.png")  # 替换为你的图片路径

# 转换为像素访问对象
pixels = image.load()

# 打印图片的尺寸
print("Image Size:", image.size)  # 输出图片的宽、高

# 访问特定像素值 (例如第 10 行第 20 列的像素)
# pixel = pixels[20, 10]  # (R, G, B) 或 (R, G, B, A) 值
# print("Pixel value at (10, 20):", pixel)

# 遍历所有像素值（可能会非常慢）
for i in range(image.height):  # 高度
    for j in range(image.width):  # 宽度
        print(f"Pixel at ({i}, {j}):", pixels[j, i])
