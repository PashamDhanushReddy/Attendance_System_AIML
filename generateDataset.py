import cv2
import os
import  numpy as np
def generate_dataset(name):
    path = 'Dataset/' + name
    isExist = os.path.exists(path)
    if isExist:
        print("Name Already Taken")
        # In a GUI context, you might want to show a messagebox here
        # For now, we'll just return and not generate dataset
        return
    else:
        os.makedirs(path)

    face_classifier=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    def face_cropped(img):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_classifier.detectMultiScale(gray,1.3,5)

        if len(faces) == 0:
            return None
        for (x,y,w,h) in faces:
            cropped_face=img[y:y+h,x:x+w]
            return cropped_face

    v_cap=cv2.VideoCapture(0)
    img_id=0

    while True:
        ret,frame=v_cap.read()
        if face_cropped(frame) is not None:
            img_id += 1
            face_r=cv2.resize(face_cropped(frame),(400,400))
            face_r=cv2.cvtColor(face_r,cv2.COLOR_BGR2GRAY)

            fileNamePath='Dataset/'+name+'/'+name+"_"+str(img_id)+".jpg"
            cv2.imwrite(fileNamePath,face_r)
            cv2.putText(face_r,str(img_id),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

            cv2.imshow("Cropped_Face",face_r)
            if cv2.waitKey(1)=='q' or int(img_id)==10:
                break

    v_cap.release()
    cv2.destroyAllWindows()
    print("Collection image sample is completed!!")


