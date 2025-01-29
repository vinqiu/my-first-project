import cv2

# 加载预训练的Haar级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 读取图像
image = cv2.imread('image.png')

# 将图像转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 在检测到的人脸周围绘制矩形框
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 显示结果图像
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果图像
cv2.imwrite('4.1faces.jpg', image)