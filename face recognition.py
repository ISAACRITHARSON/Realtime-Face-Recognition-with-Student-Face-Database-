import cv2

def generate_dataset(img,id,img_id):
    cv2.imwrite("data/user."+str(id)+"."+str(img_id)+".jpg",img)

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text,clf):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        id,_=clf.predict(gray_img[y:y+h,x:x+w])
        if id==1:
              cv2.putText(img, "Person", (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords,img


def recognize(img,clf,faceCascade):
    color={"blue":(255,0,0),"red":(0,0,255),"green":(0,255,0),"white":(255,255,255)}
    coords, img = draw_boundary(img, faceCascade, 1.1, 10, color['green'], "Student",clf)
    return img




def detect(img,faceCascade,img_id):
    color={"blue":(255,0,0),"red":(0,0,255),"green":(0,255,0),"white":(255,255,255)}
    coords,img=draw_boundary(img,faceCascade,1.1,10,color['red'],"Unknown")

    if len(coords)==4:
        roi_img=img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
        user_id=1
        generate_dataset(roi_img,user_id,img_id)
    return img
faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
clf=cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")
video_capture=cv2.VideoCapture(0)
img_id=0

while True:
    _,img=video_capture.read()
    #img=detect(img,faceCascade,img_id)
    img = recognize(img,clf,faceCascade)
    cv2.imshow('face detection',img)
    img_id+=1
    cv2.waitKey(1)

