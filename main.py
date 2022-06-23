import cv2
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords,img
def detect(img,faceCascade):
    color={"blue":(255,0,0),"red":(0,0,255),"green":(0,255,0)}
    coords,img=draw_boundary(img,faceCascade,1.1,10,color['green'],"face")
    return img
faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture=cv2.VideoCapture(0)
while True:
    _,img=video_capture.read()
    img=detect(img,faceCascade)
    cv2.imshow('face detection',img)
    cv2.waitKey(1)

