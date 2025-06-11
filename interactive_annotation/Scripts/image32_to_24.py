import os
from PIL import Image


def convert_images(input_path, output_path, background_color=(0, 0, 0)):
    """
    Convert 32-bit images to 24-bit images within a directory.

    Parameters:
    - input_path: Path to the directory containing the input images.
    - output_path: Path to the directory where the converted images will be saved.
    """
    # 确保输出路径存在
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 遍历输入路径下的所有文件
    for filename in os.listdir(input_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            # 构建完整的文件路径
            file_path = os.path.join(input_path, filename)

            try:
                # 加载图像
                with Image.open(file_path) as img:
                    # Check if the image has an alpha channel
                    if img.mode == 'RGBA':
                        # Create a new image with a solid color background
                        background = Image.new('RGB', img.size, background_color)
                        # Paste the image using alpha as a mask
                        background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                        rgb_img = background
                    else:
                        # Convert image to RGB if not having an alpha channel
                        rgb_img = img.convert('RGB')

                    # 构建输出文件路径
                    output_file_path = os.path.join(output_path, filename)
                    # 保存转换后的图像
                    rgb_img.save(output_file_path)
                    print(f"Converted and saved: {output_file_path}")
            except Exception as e:
                print(f"Error converting {file_path}: {e}")


# 使用示例
input_path = 'C:/data/PhDthesis/data/image_seg/test' #'C:/data/PhDthesis/data/L2_predict/sumv2_data_for_compare/2D/methods_exp/manual/images/rgba'
output_path = 'C:/data/PhDthesis/data/image_seg/test' #'C:/data/PhDthesis/data/L2_predict/sumv2_data_for_compare/2D/methods_exp/manual/images/rgb'
convert_images(input_path, output_path)
