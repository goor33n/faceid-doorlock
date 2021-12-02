# -*- coding: utf-8 -*-
# 페이스아이디 관련 코드는 https://github.com/codeingschool/Facial-Recognition 에서 가져왔습니다.
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import drivers
import time
import datetime
import RPi.GPIO as GPIO

SWITCH_PIN = 4
BUZZER_PIN = 20
SERVO_PIN = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(BUZZER_PIN, 430)
serPwm = GPIO.PWM(SERVO_PIN, 50) # 서보 모터 pwm
serPwm.start(6) # 서보 모터 실행 (6: 잠금, 2.5: 열림)
time.sleep(0.02)

display = drivers.Lcd()

switchCount = 0

faceRegisted = 1
faceNotMatch = 0

pwm.start(100)
time.sleep(0.1)
pwm.start(0)

#  얼굴등록
def faceRegist():
    display.lcd_clear()
    display.lcd_display_string("Put your face", 1)
    display.lcd_display_string("on the camera", 2)
    face_classifier = cv2.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')



    def face_extractor(img):

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        if faces is():
            return None

        for(x,y,w,h) in faces:
            cropped_face = img[y:y+h, x:x+w]

        return cropped_face


    cap = cv2.VideoCapture(0)
    count = 0

    display.lcd_clear()
    
    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count+=1
            face = cv2.resize(face_extractor(frame),(200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = 'faces/user'+str(count)+'.jpg'
            cv2.imwrite(file_name_path,face)

            
            display.lcd_display_string(str(count)+"%"+" complete", 1) # 진행률 표시

            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
        else:
            # print("Face not Found")
            pass

        if cv2.waitKey(1)==13 or count==100:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Colleting Samples Complete!!!')

    display.lcd_clear()
    display.lcd_display_string("Face Regist", 1)
    display.lcd_display_string("Complete!", 2)
    faceRegisted = 1
    print(faceRegisted)
    time.sleep(2)
    display.lcd_clear()

    return 0
### 얼굴등록 함수 끝 ###

### 얼굴 확인 ###
def faceId():

    data_path = 'faces/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

    Training_Data, Labels = [], []

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()

    model.train(np.asarray(Training_Data), np.asarray(Labels))

    print("Model Training Complete!!!!!")

    face_classifier = cv2.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')

    def face_detector(img, size = 0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        if faces is():
            return img,[]

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))

        return img,roi

    cap = cv2.VideoCapture(0)

    unlockCount = 0
    lockCount = 0
    notCount = 0

    while True:
        # serPwm.ChangeDutyCycle(6)
        # serPwm.start(6)

        ret, frame = cap.read()

        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100*(1-(result[1])/300))
                display_string = str(confidence)+'% Confidence it is user'
            cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)


            if confidence > 75:
                cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)
                unlockCount += 1
                lockCount = 0
                if unlockCount > 5: # 일정 시간동안 얼굴 매치가 성공하면
                    return 1
                

            else:
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)
                lockCount += 1
                if lockCount == 10: # 일정 시간동안 얼굴 매치가 안되면
                    return 0


        except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            notCount += 1
            if notCount > 20: # 일정 시간 이상 얼굴을 찾지 못했으면
                return 2
            pass

        if cv2.waitKey(1)==13:
            break

        time.sleep(0.1)
        


    cap.release()
    cv2.destroyAllWindows()
### 얼굴 확인 함수 끝 ###

try:
    while True:
        switchVal = GPIO.input(SWITCH_PIN)



        now = datetime.datetime.now()
        display.lcd_display_string(now.strftime("%x%X"), 1)
        # display.lcd_display_string(str(now.time()), 1)
        # print(faceRegisted)

        time.sleep(0.1)

        if faceNotMatch == 3:
            pwm.start(20)
            time.sleep(5)
            pwm.start(0)
            faceNotMatch = 0

        # 버튼을 2초간 누르면 screenVal 값을 바꾼다.
        if switchVal == 1:
            while True:
                switchVal = GPIO.input(SWITCH_PIN)
                switchCount += 0.1
                if switchVal == 0:
                    if switchCount > 2:
                        faceRegist()
                        faceRegisted = 1
                    else:
                        if faceRegisted == 0:
                            display.lcd_clear()
                            display.lcd_display_string("Please regist", 1)
                            display.lcd_display_string("your Face", 2)
                            time.sleep(2)
                        elif faceRegisted == 1:
                            display.lcd_clear()
                            display.lcd_display_string("Detecting..", 1)
                            time.sleep(0.1)
                            if faceId() == 1: # 1: 인식 성공, 0: 매치 안됨, 2: 인식 안됨
                                display.lcd_clear()
                                display.lcd_display_string("Unlock", 1)
                                faceNotMatch = 0
                                serPwm.ChangeDutyCycle(2.5) # 열림
                                time.sleep(5)
                                serPwm.ChangeDutyCycle(6) # 닫힘
                                time.sleep(1)
                    
                            elif faceId() == 0:
                                display.lcd_clear()
                                display.lcd_display_string("Not match", 1)
                                faceNotMatch += 1
                                pwm.start(20)
                                time.sleep(0.2)
                                pwm.start(0)
                                time.sleep(0.2)
                                pwm.start(20)
                                time.sleep(0.2)
                                pwm.start(0)
                                time.sleep(2)
                            elif faceId() == 2:
                                display.lcd_clear()
                                display.lcd_display_string("Face not found", 1)
                                display.lcd_display_string("Try again", 2)
                                time.sleep(2)
                            

                    switchCount = 0
                    display.lcd_clear()
                    break
                time.sleep(0.1)
            


finally:
    display.lcd_clear()
    GPIO.cleanup()
    pwm.stop()
    serPwm.stop()
    print("cleaning up!")
  