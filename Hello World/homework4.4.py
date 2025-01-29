import face_recognition
from PIL import Image, ImageDraw

def simple_face_detect(image_path):
    try:
        # 加载图片并识别人脸
        image = face_recognition.load_image_file(image_path)
        faces = face_recognition.face_locations(image, model="cnn")
        
        # 转换图片格式并准备绘图
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        # 绘制所有人脸框
        for top, right, bottom, left in faces:
            draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)

        # 保存并显示结果
        pil_image.save("4.4.jpg")
        pil_image.show()
        
        print(f"成功检测到 {len(faces)} 张人脸")
        if not faces:
            print("提示：未检测到人脸可能由于遮挡、侧脸或图像模糊")

    except FileNotFoundError:
        print("错误：图片文件不存在！")

# 使用示例
simple_face_detect("image.png")