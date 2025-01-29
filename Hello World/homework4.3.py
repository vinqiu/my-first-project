import face_recognition
import cv2

# 读取图像
image = cv2.imread('image.png')

# 使用face_recognition检测人脸
face_locations = face_recognition.face_locations(image)

# 在检测到的人脸周围绘制矩形框
for (top, right, bottom, left) in face_locations:
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)

# 显示结果图像
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果图像
cv2.imwrite('4.3.jpg', image)