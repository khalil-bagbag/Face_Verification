from django.shortcuts import render
from django.http import HttpResponse
import os  
import cv2  
import numpy as np  
from keras.models import model_from_json    
from django.core.files.storage import FileSystemStorage

from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import json ,base64 
from PIL import Image
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet


from django.views.decorators.csrf import csrf_exempt


import cv2 

#load model  
model = model_from_json(open("/home/khalil/Bureau/PI_Face_Recognition/Models/modelCNN_json.json", "r").read())  
model2 = load_model("/home/khalil/Bureau/PI_Face_Recognition/Models/mask_detector.model",compile=False)
model_atrribut= load_model("/home/khalil/Bureau/PI_Face_Recognition/Models/weights-FC37-MobileNetV2-0.92.hdf5",compile=False)
embedder = FaceNet()
#load weights  
model.load_weights('/home/khalil/Bureau/PI_Face_Recognition/Models/CNN.48-0.66.hdf5') 


  
detector = MTCNN()




def index(request):
    context={'a':1}
    return render(request,'index.html',context)
def emotion_page(request):
    context={'a':1}
    return render(request,'emotion.html',context)
def attributes_page(request):
    context={'a':1}
    return render(request,'attributes.html',context)
def mask_page(request):
    context={'a':1}
    return render(request,'mask.html',context)
def verification_page(request):
    context={'a':1}
    return render(request,'verification.html',context)

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
    
def camera(request):
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            print(check) #prints true as long as the webcam is running
            print(frame) #prints matrix values of each framecd 
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'): 
                cv2.imwrite(filename='saved_img.jpg', img=frame)
                webcam.release()
                img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
                img_new = cv2.imshow("Captured Image", img_new)
                cv2.waitKey(1650)
                cv2.destroyAllWindows()
                print("Processing image...")
                img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
                print("Converting RGB image to grayscale...")
                gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
                print("Converted RGB image to grayscale...")
                print("Resizing image to 28x28 scale...")
                img_ = cv2.resize(gray,(28,28))
                print("Resized...")
                img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
                print("Image saved!")
            
                break
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
        
        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
    return render(request,'index.html')       

    




def predictImage(request):
    predictions =[]
    thisdict= []

    fileObj=request.FILES['Path_file']
    img = cv2.imdecode(np.fromstring(fileObj.read(), np.uint8), cv2.IMREAD_UNCHANGED) 
    faces_detected = detector.detect_faces(img)  

   
    for face in faces_detected:
        x, y, w, h = face['box']
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi_gray=gray_img[y:y+w,x:x+h] #cropping region of interest i.e. face area from  image  
        roi_gray=cv2.resize(roi_gray,(48,48))  
        img_pixels = img_to_array(roi_gray)  
        img_pixels = np.expand_dims(img_pixels, axis =0)  
        img_pixels /= 255  
        pred = model.predict(img_pixels) 
        print(pred) 

        #find max indexed array  
        max_index = np.argmax(pred[0]) 
        print(max_index) 

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral') 
        dictionnaire = {"angry": pred[0][0],"disgust": pred[0][1],"fear": pred[0][2],"happy": pred[0][3],"sad": pred[0][4],"surprise": pred[0][5],"neutral": pred[0][6]}
        thisdict.append(dictionnaire) 
        print(thisdict)
        predicted_emotion = emotions[max_index]
        predictions.append(predicted_emotion)

    

    image = base64.b64encode(img)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # fontScale
    fontScale = 1
    
    # Blue color in BGR
    color = (0, 0, 255)
    
    # Line thickness of 2 px
    thickness = 2
    
    str_word=' '.join(map(str, predictions))
    for face,p in zip(faces_detected,predictions)   :
        x, y, w, h = face['box']
        keypoints =face['keypoints']
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.circle(img,(keypoints['left_eye']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['right_eye']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['nose']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['mouth_left']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['mouth_right']), 2, (0,155,255), 2)
        cv2.putText(img,p,(x, y + h + 30 ),font, fontScale, color, thickness, cv2.LINE_AA)  

    
    img = image_resize(img, width=500, height=400, inter=cv2.INTER_AREA)
    _, jpeg = cv2.imencode('.jpeg', img)
    b64 = base64.encodebytes(jpeg.tobytes())
    
   
    return render(request,'emotion.html',{'predictedLabel':predictions,'image':b64.decode("utf-8"),
    'data': thisdict}) 

def mask(request):

    predictions=[]
    
    fileObj=request.FILES['Path_file']
    img = cv2.imdecode(np.fromstring(fileObj.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    RGB_img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    faces_detected = detector.detect_faces(img)

    for face in faces_detected:
        x, y, w, h = face['box']
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray=RGB_img[y:y+w,x:x+h] #cropping region of interest i.e. face area from  image  
        roi_gray=cv2.resize(roi_gray,(224,224))  
        img_pixels = img_to_array(roi_gray) 
        img_pixels /= 255.0 
        img_pixels = np.expand_dims(img_pixels, axis = 0)      
        pred = model2.predict(img_pixels) 
        

        #find max indexed array  
        max_index = np.argmax(pred[0])  

        mask = ('mask', 'no_mask')  
        predicted_mask = mask[max_index]
        predictions.append(predicted_mask)

    

    image = base64.b64encode(img)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # fontScale
    fontScale = 1
    
    # Blue color in BGR
    color = (0, 0, 255)
    
    # Line thickness of 2 px
    thickness = 2
    
    str_word=' '.join(map(str, predictions))
    for face,p in zip(faces_detected,predictions)   :
        x, y, w, h = face['box']
        keypoints =face['keypoints']
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.circle(img,(keypoints['left_eye']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['right_eye']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['nose']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['mouth_left']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['mouth_right']), 2, (0,155,255), 2)
        cv2.putText(img,p,(x, y + h + 30 ),font, fontScale, color, thickness, cv2.LINE_AA)  

    
    img = image_resize(img, width=500, height=400, inter=cv2.INTER_AREA)
    _, jpeg = cv2.imencode('.jpeg', img)
    b64 = base64.encodebytes(jpeg.tobytes())
    
   
    return render(request,'mask.html',{'predictedLabel':predictions,'image':b64.decode("utf-8")})


def attribut(request):

    predictions=[]
    
    fileObj=request.FILES['Path_file']
    img = cv2.imdecode(np.fromstring(fileObj.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    RGB_img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    faces_detected = detector.detect_faces(img)

    for face in faces_detected:
        x, y, w, h = face['box']
        cv2.rectangle(img, (x-30, y-30), (x+w+30, y+h+30), (0, 0, 255), 2)
        roi_gray=RGB_img[y:y+w,x:x+h] #cropping region of interest i.e. face area from  image  
        roi_gray=cv2.resize(roi_gray,(224,224))  
        img_pixels = img_to_array(roi_gray) 
        img_pixels /= 255.0 
        img_pixels = np.expand_dims(img_pixels, axis = 0)      
        pred = model_atrribut.predict(img_pixels)[0]
        

         

        attributes=['5_o_Clock_Shadow', 'Arched_Eyebrows','Bags_Under_Eyes', 'Bald' ,'Bangs' ,'Big_Lips','Big_Nose' ,'Black_Hair' ,'Blond_Hair','Brown_Hair', 'Bushy_Eyebrows',
                        'Chubby' ,'Double_Chin' ,'Eyeglasses', 'Goatee' ,'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones' ,'Male', 'Mouth_Slightly_Open' ,'Mustache' ,'Narrow_Eyes', 'No_Beard' ,'Oval_Face',
                        'Pointy_Nose' ,'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair' ,'Wavy_Hair' ,'Wearing_Earrings', 'Wearing_Hat' ,'Wearing_Lipstick',
                        'Wearing_Necklace' ,'Wearing_Necktie' ,'Young'] 
        

        predicted_attributs = { k:str(v)[:4]   for k, v in zip(attributes,pred) if v>0.5}
        predictions.append(predicted_attributs)
        
    print(predictions)
    length=len(predictions)
       
        
       

    

    image = base64.b64encode(img)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # fontScale
    fontScale = 0.5
    
    # Blue color in BGR
    color = (0, 0, 255)
    
    # Line thickness of 2 px
    thickness = 1
    k=0
    
    str_word=' '.join(map(str, predictions))
    for face,p in zip(faces_detected,predictions)   :
        x, y, w, h = face['box']
        keypoints =face['keypoints']
        cv2.rectangle(img, (x-30, y-30), (x+w+30, y+h+30), (0, 0, 255), 2)
        cv2.circle(img,(keypoints['left_eye']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['right_eye']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['nose']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['mouth_left']), 2, (0,155,255), 2)
        cv2.circle(img,(keypoints['mouth_right']), 2, (0,155,255), 2)
        for key, value in predicted_attributs.items() :
            text= key+" "+value
            k=k+1
            print(text)
            cv2.putText(img,str(text),(x+100, y+30*k),font, fontScale, color, thickness, cv2.LINE_AA)  

        

    
    img = image_resize(img, width=500, height=400, inter=cv2.INTER_AREA)
    _, jpeg = cv2.imencode('.jpeg', img)
    b64 = base64.encodebytes(jpeg.tobytes())
    
   
    return render(request,'attributes.html',{'predictedLabel':predictions,'image':b64.decode("utf-8")})



def verification(request):

    predictions=[]

    required_size=(160, 160)

    
    
    fileObj1=request.FILES['Path_file']
    img1 = cv2.imdecode(np.fromstring(fileObj1.read(), np.uint8), cv2.IMREAD_UNCHANGED) 
    face1 = embedder.extract(img1, threshold=0.95)
    print(face1)
    yhat1=face1[0]['embedding']
    print(yhat1)
    

    fileObj2=request.FILES['Path_file2']
    img2 = cv2.imdecode(np.fromstring(fileObj2.read(), np.uint8), cv2.IMREAD_UNCHANGED) 
    face2 = embedder.extract(img2, threshold=0.95)
    print(face2)
    yhat2=face2[0]['embedding']
    print(yhat2)
    """
    cette fonction permet d'obtenir les embedding d'un visage en procédant d'abord à la normalisation réquise par le facenet
    """
  
    alpha = np.sum((yhat1*yhat2))/(np.linalg.norm(yhat1)*np.linalg.norm(yhat2))
    alpha = (np.arccos(alpha)*180)/np.pi
    print(len(face1))
    print(len(face2))
    
    if  len(face1) ==1 and len(face2) ==1 :
        if alpha >50:
            a="Faces dont belong to the same person"
        else :
            a="Faces belong to the same person"
    else :
        a="Each picture must contain one face"


    

    for f in face1:
        x, y, w, h = f['box']
        keypoints =f['keypoints']
        cv2.rectangle(img1, (x-30, y-30), (x+w+30, y+h+30), (0, 0, 255), 2)
        cv2.circle(img1,(keypoints['left_eye']), 2, (0,155,255), 2)
        cv2.circle(img1,(keypoints['right_eye']), 2, (0,155,255), 2)
        cv2.circle(img1,(keypoints['nose']), 2, (0,155,255), 2)
        cv2.circle(img1,(keypoints['mouth_left']), 2, (0,155,255), 2)
        cv2.circle(img1,(keypoints['mouth_right']), 2, (0,155,255), 2)

    for f in face2 :
        x, y, w, h = f['box']
        keypoints =f['keypoints']
        cv2.rectangle(img2, (x-30, y-30), (x+w+30, y+h+30), (0, 0, 255), 2)
        cv2.circle(img2,(keypoints['left_eye']), 2, (0,155,255), 2)
        cv2.circle(img2,(keypoints['right_eye']), 2, (0,155,255), 2)
        cv2.circle(img2,(keypoints['nose']), 2, (0,155,255), 2)
        cv2.circle(img2,(keypoints['mouth_left']), 2, (0,155,255), 2)
        cv2.circle(img2,(keypoints['mouth_right']), 2, (0,155,255), 2)

        

        


    img1 = image_resize(img1, width=500, height=400, inter=cv2.INTER_AREA)
    _, jpeg = cv2.imencode('.jpeg', img1)
    b64_img1 = base64.encodebytes(jpeg.tobytes())

    img2 = image_resize(img2, width=500, height=400, inter=cv2.INTER_AREA)
    _, jpeg = cv2.imencode('.jpeg', img2)
    b64_img2 = base64.encodebytes(jpeg.tobytes())


    
   
    return render(request,'verification.html',{'a':a,'image1':b64_img1.decode("utf-8"),'image2':b64_img2.decode("utf-8")})











