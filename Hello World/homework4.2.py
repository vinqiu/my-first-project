import cv2

# 加载预训练的Haar级联分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 读取图像
image = cv2.imread('image.png')

# 将图像转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 改进参数设置
# scaleFactor: 每次图像缩小的比例（默认1.1，可以尝试更小的值，如1.05）
# minNeighbors: 每个候选矩形应该保留的邻居数量（默认5，可以尝试更大的值，如10）
# minSize: 人脸的最小尺寸（默认30x30，可以根据图像中的人脸大小调整）
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,  # 更小的值可以检测到更小的人脸
    minNeighbors=10,   # 更高的值可以减少误检，但可能漏检
    minSize=(30, 30),  # 根据图像中的人脸大小调整
    flags=cv2.CASCADE_SCALE_IMAGE
)

# 在检测到的人脸周围绘制矩形框
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 显示结果图像
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果图像
cv2.imwrite('4.2.jpg', image)