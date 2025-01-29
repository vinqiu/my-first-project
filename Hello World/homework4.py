import face_recognition
import cv2 as cv
import matplotlib.pyplot as plt
image = face_recognition.load_image_file('image.png')
face_locations =face_recognition.face_locations(image,model='cnn')
for face_location in face_locations:
    top,right,bottom,left = face_location
    cv.rectangle(image,(left,top),(right,bottom),(0,0,255),2)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    cv.imwrite('detected_faces_face_recognition.jpg', image)