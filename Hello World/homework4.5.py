import face_recognition
from PIL import Image, ImageDraw

def detect_faces(image_path, model="cnn", upscale=2):
    """
    高精度人脸检测（仅绘制人脸框）
    :param image_path: 图片路径
    :param model: 检测模型，可选 "hog" 或 "cnn"（默认）
    :param upscale: 上采样次数，默认2（检测更小人脸）
    """
    try:
        # 加载图片
        image = face_recognition.load_image_file(image_path)
        
        # 使用指定模型检测人脸
        face_locations = face_recognition.face_locations(
            image, 
            number_of_times_to_upsample=upscale, 
            model=model
        )

        # 转换图片格式并准备绘图
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        # 绘制所有人脸框
        for top, right, bottom, left in face_locations:
            draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)

        # 保存并显示结果
        output_path = "4.5.jpg"
        pil_image.save(output_path)
        pil_image.show()

        print(f"✅ 检测到 {len(face_locations)} 张人脸")
        if not face_locations:
            print("提示：未检测到人脸，可能由于遮挡、侧脸或图像模糊")
        else:
            print(f"结果已保存至：{output_path}")

    except FileNotFoundError:
        print("错误：图片文件不存在！")
    except Exception as e:
        print(f"发生错误：{str(e)}")

# 使用示例
detect_faces(
    image_path="image.png",  # 图片路径
    model="cnn",             # 使用CNN模型
    upscale=2                # 上采样次数
)