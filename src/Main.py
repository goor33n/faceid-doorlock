# -*- coding: utf-8 -*-
# 페이스아이디 관련 코드는 https://github.com/codeingschool/Facial-Recognition 에서 참고하였습니다.
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import drivers
import time
import datetime
import RPi.GPIO as GPIO

SWITCH_PIN = 4   # 스위치 핀
BUZZER_PIN = 20  # 부저 핀
SERVO_PIN = 18   # 서보모터 핀

GPIO.setmode(GPIO.BCM)
GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # 내부 pull down 저항
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(BUZZER_PIN, 430)  # 부저 pwm
serPwm = GPIO.PWM(SERVO_PIN, 50) # 서보 모터 pwm

display = drivers.Lcd() # 디스플레이

switchCount = 0  # 스위치가 눌린 시간을 기록
faceRegisted = 0 # 얼굴이 등록되었는지 여부 (1: 등록됨, 0: 등록되지 않음)
faceNotMatch = 0 # 인식된 얼굴이 등록된 얼굴과 맞지 않은 횟수 기록

pwm.start(20)  # 작동이 시작될 때 소리내기
time.sleep(0.1)
pwm.start(0)

serPwm.start(6) # 서보 모터 실행 (6: 잠금, 2.5: 열림)
time.sleep(0.02)

### 얼굴등록 함수 ###
def faceRegist():
    display.lcd_clear()
    display.lcd_display_string("Put your face", 1)  # 디스플레이 표시
    display.lcd_display_string("on the camera", 2)

    # 얼굴인식 xml 파일
    face_classifier = cv2.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')

    def face_extractor(img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # 흑백 처리
        faces = face_classifier.detectMultiScale(gray,1.3,5) # 얼굴 찾기

        if faces is():  # 얼굴을 찾지 못하면 None 리턴
            return None

        for(x,y,w,h) in faces: # 얼굴을 찾으면
            cropped_face = img[y:y+h, x:x+w] # 얼굴 크기만큼 cropped_face에 넣음

        return cropped_face # cropped_face 리턴

    cap = cv2.VideoCapture(0) # 카메라 실행
    count = 0 # 저장할 이미지 카운트

    display.lcd_clear()
    
    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None: # 얼굴을 감지하면
            count+=1
            face = cv2.resize(face_extractor(frame),(200,200)) # 이미지 크기 200x200으로
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) # 흑백으로

            file_name_path = 'faces/user'+str(count)+'.jpg'  # 얼굴 사진 파일 이름 설정
            cv2.imwrite(file_name_path,face)   # faces 폴더에 얼굴 사진 저장

            display.lcd_display_string(str(count)+"%"+" complete", 1) # 진행률 표시

            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face Cropper',face)
            
        else:  # 얼굴을 감지하지 못하면
            # print("Face not Found")
            pass

        if cv2.waitKey(1)==13 or count==100: # 얼굴 사진 100장 저장하면 break
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Colleting Samples Complete!!!')

    display.lcd_clear()
    display.lcd_display_string("Face Regist", 1) # 얼굴 등록 완료 메시지 표시
    display.lcd_display_string("Complete!", 2)
    time.sleep(2)
    display.lcd_clear()

    return 0
### 얼굴등록 함수 끝 ###

### 얼굴 확인 함수 ###
def faceId():

    data_path = 'faces/' 
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))] # 얼굴 이미지 파일 얻기

    Training_Data, Labels = [], []  # 데이터와 매칭될 라벨 변수

    for i, files in enumerate(onlyfiles):  # 파일 개수 만큼 반복
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # 이미지 불러오기
        Training_Data.append(np.asarray(images, dtype=np.uint8)) # Training_Data 리스트에 이미지를 바이트 배열로 추가
        Labels.append(i) # Labels 리스트에 카운트 번호 추가

    Labels = np.asarray(Labels, dtype=np.int32) # Labels를 32비트 정수로 변환

    model = cv2.face.LBPHFaceRecognizer_create() # 모델 생성

    model.train(np.asarray(Training_Data), np.asarray(Labels)) # 모델 학습

    print("Model Training Complete!!!!!")

    face_classifier = cv2.CascadeClassifier('./xml/haarcascade_frontalface_default.xml')

    def face_detector(img, size = 0.5): # 얼굴 감지 후 리턴
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

    unlockCount = 0  # 얼굴이 매치된 경우 카운트
    lockCount = 0 # 얼굴이 매치되지 않은 경우 카운트
    notCount = 0 # 얼굴을 찾지 못한 경우 카운트

    while True:
        ret, frame = cap.read()

        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face) # 학습한 모델로 예측

            if result[1] < 500: # result[1]: 신뢰도, 0에 가까울수록 자신과 같다는 뜻
                confidence = int(100*(1-(result[1])/300))  # 신뢰도를 백분위로 변환
                display_string = str(confidence)+'% Confidence it is user'
            cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)


            if confidence > 75: # 정확도가 75%보다 높으면 unlock
                cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)
                unlockCount += 1
                lockCount = 0
                if unlockCount > 5: # 일정 시간동안 얼굴 매치가 성공하면
                    return 1
                

            else: # 정확도가 75% 이하면 lock
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)
                lockCount += 1
                if lockCount == 10: # 일정 시간동안 얼굴 매치가 안되면
                    return 0


        except: # 얼굴을 찾지 못했을 때
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
        switchVal = GPIO.input(SWITCH_PIN) # 스위치의 값을 switchVal에 받음

        now = datetime.datetime.now() # 현재 시간 설정
        display.lcd_display_string(now.strftime("%x%X"), 1) # 현재 날짜와 시간을 디스플레이에 표시
        time.sleep(0.1)

        if faceNotMatch == 3:  # 만약 얼굴인식에 세 번 실패하면 5초간 소리를 낸다.
            pwm.start(20)
            time.sleep(5)
            pwm.start(0)
            faceNotMatch = 0


        if switchVal == 1: # 스위치를 눌렀을 때
            while True:
                switchVal = GPIO.input(SWITCH_PIN) # 스위치 값을 계속 받음
                switchCount += 0.1     # 스위치가 눌린 시간 기록
                if switchVal == 0:    # 스위치에서 손을 뗐을 때
                    if switchCount > 2: # 스위치를 2초간 눌렀을 때: 얼굴등록
                        faceRegist()   # 얼굴 등록 함수 호출
                        faceRegisted = 1 # (1: 얼굴이 등록됨, 2: 얼굴이 등록되지 않음)
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

                            # faceId() : 인식된 얼굴을 등록된 얼굴과 대조하는 함수 (1: 인식 성공, 0: 매치 안됨, 2: 인식 안됨)
                            if faceId() == 1: ## 인식에 성공했을 때
                                display.lcd_clear()
                                display.lcd_display_string("Unlock", 1) # 디스플레이에 Unlcok 표시
                                faceNotMatch = 0  # 얼굴 일치하지 않은 횟수 초기화
                                serPwm.ChangeDutyCycle(2.5) # 열림
                                time.sleep(5)
                                serPwm.ChangeDutyCycle(6) # 닫힘
                                time.sleep(1)
                    
                            elif faceId() == 0: ## 인식에 실패했을 때
                                display.lcd_clear()
                                display.lcd_display_string("Not match", 1) # 디스플레이에 Not match 표시
                                faceNotMatch += 1  # 얼굴 일치하지 않은 횟수 카운트

                                pwm.start(20)    # 부저음 울림
                                time.sleep(0.2)
                                pwm.start(0)
                                time.sleep(0.2)
                                pwm.start(20)
                                time.sleep(0.2)
                                pwm.start(0)
                                time.sleep(2)

                            elif faceId() == 2: ## 얼굴을 찾지 못했을 때
                                display.lcd_clear()
                                display.lcd_display_string("Face not found", 1) # 디스플레이에 표시
                                display.lcd_display_string("Try again", 2)
                                time.sleep(2)
                            
                    switchCount = 0  # 얼굴인식 작업이 끝나고 스위치가 눌린 시간 초기화
                    display.lcd_clear() # 디스플레이 초기
                    break
                time.sleep(0.1)
            


finally:
    display.lcd_clear()
    GPIO.cleanup()
    pwm.stop()
    serPwm.stop()
    print("cleaning up!")
  