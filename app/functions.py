import mediapipe as mp
import cv2 as cv
import numpy as np
from math import acos, degrees
import threading as th
import pyttsx3 as tts

def text_to_speech(text):
    engine = tts.init()
    engine.setProperty('rate', 250)
    engine.say(text)
    engine.runAndWait()

"""
### Declaración de coordenadas de articulaciones:
#Declaración del joint 0(nariz)
narizX = int(results.pose_landmarks.landmark[0].x * width)
narizY = int(results.pose_landmarks.landmark[0].y * height)

#Declaración del joint 11(hombro derecho)
hombroDerechoX = int(results.pose_landmarks.landmark[11].x * width)
hombroDerechoY = int(results.pose_landmarks.landmark[11].y * height)

#Declaración del joint 12(hombro izquierdo)
hombroIzquierdoX = int(results.pose_landmarks.landmark[12].x * width)
hombroIzquierdoY = int(results.pose_landmarks.landmark[12].y * height)

#Declaración del joint 13(codo derecho)
codoDerechoX = int(results.pose_landmarks.landmark[13].x * width)
codoDerechoY = int(results.pose_landmarks.landmark[13].y * height)

#Declaración del joint 14(codo izquierdo)
codoIzquierdoX = int(results.pose_landmarks.landmark[14].x * width)
codoIzquierdoY = int(results.pose_landmarks.landmark[14].y * height)

#Declaración del joint 15(muñeca derecha)
munecaDerechaX = int(results.pose_landmarks.landmark[15].x * width)
munecaDerechaY = int(results.pose_landmarks.landmark[15].y * height)

#Declaración del joint 16(muñeca izquierda)
munecaIzquierdaX = int(results.pose_landmarks.landmark[16].x * width)
munecaIzquierdaY = int(results.pose_landmarks.landmark[16].y * height)

#Declaración del joint 23 (cadera derecha)
caderaDerechaX = int(results.pose_landmarks.landmark[23].x * width)
caderaDerechaY = int(results.pose_landmarks.landmark[23].y * height)

#Declaración del joint 24 (cadera izquierda)
caderaIzquierdaX = int(results.pose_landmarks.landmark[24].x * width)
caderaIzquierdaY = int(results.pose_landmarks.landmark[24].y * height)

#Declaración del joint 25(rodilla derecha)
rodillaDerechaX = int(results.pose_landmarks.landmark[25].x * width)
rodillaDerechaY = int(results.pose_landmarks.landmark[25].y * height)

#Declaración del joint 26(rodilla izquierda)
rodillaIzquierdaX = int(results.pose_landmarks.landmark[26].x * width)
rodillaIzquierdaY = int(results.pose_landmarks.landmark[26].y * height)

#Declaración del joint 27(tobillo derecha)
tobilloDerechoX = int(results.pose_landmarks.landmark[27].x * width)
tobilloDerechoY = int(results.pose_landmarks.landmark[27].y * height)

#Declaración del joint 28(tobillo izquierda)
tobilloIzquierdoX = int(results.pose_landmarks.landmark[28].x * width)
tobilloIzquierdoY = int(results.pose_landmarks.landmark[28].y * height)

#Declaración del joint 31(punta del pie derecha)
pieDerechoX = int(results.pose_landmarks.landmark[31].x * width)
pieDerechoY = int(results.pose_landmarks.landmark[31].y * height)

#Declaración del joint 32(punta del pie izquierda)
pieIzquierdoX = int(results.pose_landmarks.landmark[32].x * width)
pieIzquierdoY = int(results.pose_landmarks.landmark[32].y * height)

### Referencias geometricas
nariz = np.array([narizX, narizY])
hombroIZquierdo = np.array([hombroIzquierdoX, hombroIzquierdoY])
hombroDerecho = np.array([hombroDerechoX, hombroDerechoY])
codoDerecho = np.array([codoDerechoX, codoDerechoY])
codoIzquierdo = np.array([codoIzquierdoX, codoIzquierdoY])
munecaDerecha = np.array([munecaDerechaX, munecaDerechaY])
munecaIzquierda = np.array([munecaIzquierdaX, munecaIzquierdaY])
caderaDerecha = np.array([caderaDerechaX, caderaDerechaY])
caderaIzquierda = np.array([caderaIzquierdaX, caderaIzquierdaY])
rodillaDerecha = np.array([rodillaDerechaX, rodillaDerechaY])
rodillaIzquierda = np.array([rodillaIzquierdaX, rodillaIzquierdaY])
tobilloDerecho = np.array([tobilloDerechoX, tobilloDerechoY])
tobilloIzquierdo = np.array([tobilloIzquierdoX, tobilloIzquierdoY])
pieDerecho = np.array([pieDerechoX, pieDerechoY])
pieIzquierdo = np.array([pieIzquierdoX, pieIzquierdoY])

### Lineas entre articulaciones del ángulo a evaluar(Ángulo del codo)
lineaHombroCodoDerecho = np.linalg.norm(hombroDerecho-codoDerecho) -> l1 = cateto
lineaCodoMunecaDerecho = np.linalg.norm(codoDerecho-munecaDerecha) -> l2 = cateto
lineaMunecaHombroDerecho = np.linalg.norm(munecaDerecha-hombroDerecho) -> l3 = hipotenusa

### Cálculo de ángulo (EJ: codo)
angulo = degrees(acos((l1**2 + l3**2 - l2**2) / (2*l1*l3)))
OR
angulo = degrees(acos( (lineaHombroCodoDerecho**2 + lineaMunecaHombroDerecho**2 - lineaCodoMunecaDerecho**2)/ (2*lineaHombroCodoDerecho*lineaMunecaHombroDerecho) ))

"""

### Módulo isómetrico (PROXIMAMENTE)
def plancha(cap):
    return 'plancha'

### Ejercicios RTR
def remo_parado(cap):
    return 'remo parado'

def lateral_raise(cap):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    up = False
    down = True
    count = 0
    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:            
            ret, frame = cap.read()
            height, width, _ = frame.shape
            if ret:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)                
                results = pose.process(frame)        
                if results.pose_landmarks is not None:
                    #Declaración del joint 11(hombro derecho)
                    hombroDerechoX = int(results.pose_landmarks.landmark[11].x * width)
                    hombroDerechoY = int(results.pose_landmarks.landmark[11].y * height)

                    #Declaración del joint 12(hombro izquierdo)
                    hombroIzquierdoX = int(results.pose_landmarks.landmark[12].x * width)
                    hombroIzquierdoY = int(results.pose_landmarks.landmark[12].y * height)

                    #Declaración del joint 13(codo derecho)
                    codoDerechoX = int(results.pose_landmarks.landmark[13].x * width)
                    codoDerechoY = int(results.pose_landmarks.landmark[13].y * height)

                    #Declaración del joint 14(codo izquierdo)
                    codoIzquierdoX = int(results.pose_landmarks.landmark[14].x * width)
                    codoIzquierdoY = int(results.pose_landmarks.landmark[14].y * height)

                    #Declaración del joint 15(muñeca derecha)
                    munecaDerechaX = int(results.pose_landmarks.landmark[15].x * width)
                    munecaDerechaY = int(results.pose_landmarks.landmark[15].y * height)

                    #Declaración del joint 16(muñeca izquierda)
                    munecaIzquierdaX = int(results.pose_landmarks.landmark[16].x * width)
                    munecaIzquierdaY = int(results.pose_landmarks.landmark[16].y * height)

                    #Declaración del joint 23 (cadera derecha)
                    caderaDerechaX = int(results.pose_landmarks.landmark[23].x * width)
                    caderaDerechaY = int(results.pose_landmarks.landmark[23].y * height)

                    #Declaración del joint 24 (cadera izquierda)
                    caderaIzquierdaX = int(results.pose_landmarks.landmark[24].x * width)
                    caderaIzquierdaY = int(results.pose_landmarks.landmark[24].y * height)

                    #Declaración de puntos de referencia
                    hombroDerecho = np.array([hombroDerechoX, hombroDerechoY])
                    hombroIzquierdo = np.array([hombroIzquierdoX, hombroIzquierdoY])
                    codoDerecho = np.array([codoDerechoX, codoDerechoY])
                    codoIzquierdo = np.array([codoIzquierdoX, codoIzquierdoY])
                    caderaDerecha = np.array([caderaDerechaX, caderaDerechaY])
                    caderaIzquierda = np.array([caderaIzquierdaX, caderaIzquierdaY])
                    munecaDerecha = np.array([munecaDerechaX, munecaDerechaY])
                    munecaIzquierda = np.array([munecaIzquierdaX, munecaIzquierdaY])

                    #Declaración de lineas en base a los puntos de referencia
                    l1 = np.linalg.norm(hombroDerecho - codoDerecho) # Catero
                    l2 = np.linalg.norm(caderaDerecha - hombroDerecho) # Cateto 
                    l3 = np.linalg.norm(codoDerecho - caderaDerecha) # Hipotenusa

                    l4 = np.linalg.norm(hombroIzquierdo - codoIzquierdo) # Cateto
                    l5 = np.linalg.norm(caderaIzquierda - hombroIzquierdo) # Cateto 
                    l6 = np.linalg.norm(codoIzquierdo - caderaIzquierda) # Hipotenusa                   
                    
                    #Calculo de angulo entre hombro codo y muñeca
                    angle1 = degrees(acos((l1**2 + l3**2 - l2**2) / (2*l1*l3)))
                    angle2 = degrees(acos((l4**2 + l6**2 - l5**2) / (2*l4*l6)))

                    angleShoulder = (angle1 + angle2)/2

                    # Dibujar puntos y líneas
                    cv.circle(frame, (hombroDerechoX, hombroDerechoY), 6, (0, 0, 255), 4)
                    cv.circle(frame, (caderaDerechaX, caderaDerechaY), 6, (0, 0, 255), 4)
                    cv.circle(frame, (codoDerechoX, codoDerechoY), 6, (0, 0, 255), 4)

                    cv.circle(frame, (hombroIzquierdoX, hombroIzquierdoY ), 6, (0, 0, 255), 4)
                    cv.circle(frame, (caderaIzquierdaX, caderaIzquierdaY), 6, (0, 0, 255), 4)
                    cv.circle(frame, (codoIzquierdoX, codoIzquierdoY), 6, (0, 0, 255), 4)

                    cv.line(frame, (hombroDerechoX, hombroDerechoY), (caderaDerechaX, caderaDerechaY), (255, 0, 0), 20)
                    cv.line(frame, (caderaDerechaX, caderaDerechaY), (codoDerechoX, codoDerechoY), (255, 0, 0), 5)
                    cv.line(frame, (codoDerechoX, codoDerechoY), (hombroDerechoX, hombroDerechoY), (255, 0, 0), 20)              

                    cv.line(frame, (hombroIzquierdoX, hombroIzquierdoY), (caderaIzquierdaX, caderaIzquierdaY), (255, 0, 0), 20)
                    cv.line(frame, (caderaIzquierdaX, caderaIzquierdaY), (codoIzquierdoX, codoIzquierdoY), (255, 0, 0), 5)
                    cv.line(frame, (codoIzquierdoX, codoIzquierdoY), (hombroIzquierdoX, hombroIzquierdoY), (255, 0, 0), 20)
                    
                    #Impresión de la imagen final
                    cv.putText(frame, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angleShoulder)), (100, 80), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angle1)), (hombroDerechoX+30, hombroDerechoY), 1, 1.5, (0, 255, 0), 2)
                    cv.putText(frame, str(int(angle2)), (hombroIzquierdoX+30, hombroIzquierdoY), 1, 1.5, (0, 255, 0), 2)

                    # Condiciones para contar una repetición
                    if angleShoulder >= 140 and down == True:
                        th.Thread(target=text_to_speech, args=('OHP',)).start()
                    if angleShoulder <= 140 and 100 <= angleShoulder:
                        arrow_length = 100
                        arrow_thickness = 3
                        arrow_color = (0, 0, 255)  # Verde
                        x_start = frame.shape[1] - arrow_length
                        y_start = arrow_length * 2
                        x_end = frame.shape[1] - arrow_length
                        y_end = arrow_length
                        cv.arrowedLine(frame, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)
                        up = True
                        down = False
                    if up == True and down == False and angleShoulder <= 99:
                        arrow_length = 100
                        arrow_thickness = 3
                        arrow_color = (0, 255, 0) #Verde
                        cv.arrowedLine(frame, (width - arrow_length, arrow_length), (width - arrow_length, arrow_length * 2),arrow_color, thickness=arrow_thickness, tipLength=0.3)
                        # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                        th.Thread(target=text_to_speech, args=('Puedes subir',)).start()                        
                        down=True                   
                    if up == True and down == True and angleShoulder >= 80:
                        count += 1
                        up = False
                        down = False
                        th.Thread(target=text_to_speech, args=('Repetición número ' + str(count),)).start()                    
                    if down == True and angleShoulder <= 90:
                        th.Thread(target=text_to_speech, args=('¡Está subiendo demasiado!',)).start()

                    # Dibujar el arco del área del angulo
                    center1 = (hombroDerechoX, hombroDerechoY)  # Puedes ajustar el centro del arco según tus necesidades
                    center2 = (hombroIzquierdoX, hombroIzquierdoY)

                    # Dibuja los círculos en la imagen superpuesta
                    cv.circle(frame, center1, 20, (255,255,255), 1)  # -1 rellena el arco
                    cv.circle(frame, center2, 20, (255,255,255), 0)  # -1 rellena el arco                    
                    if angle1 >= 160:
                        cv.circle(frame, center1, 10, (0,0,255), -1)  # -1 rellena el arco
                    if angle2 >= 160:
                        cv.circle(frame, center2, 10, (0,0,255), -1)  # -1 rellena el arco
                    if 160>=angle1 and angle1>= 150:
                        cv.circle(frame, center1, 11, (0,65,255), -1)  # -1 rellena el arco                    
                    if 160>=angle2 and angle2>= 150:
                        cv.circle(frame, center2, 11, (0,65,255), -1)  # -1 rellena el arco
                    if 150>=angle1 and angle1>= 140:
                        cv.circle(frame, center1, 12, (0,119,255), -1)  # -1 rellena el arco                    
                    if 150>=angle2 and angle2>= 140:
                        cv.circle(frame, center2, 12, (0,119,255), -1)  # -1 rellena el arco
                    if 140>=angle1 and angle1>= 130:
                        cv.circle(frame, center1, 13, (0,154,230), -1)  # -1 rellena el arco
                    if 140>=angle2 and angle2>= 130:
                        cv.circle(frame, center2, 13, (0,154,230), -1)  # -1 rellena el arco
                    if 130>=angle1 and angle1>= 120:
                        cv.circle(frame, center1, 14, (0,205,255), -1)  # -1 rellena el arco                    
                    if 130>=angle2 and angle2>= 120:
                        cv.circle(frame, center2, 14, (0,205,255), -1)  # -1 rellena el arco
                    if 120>=angle1 and angle1>= 110:
                        cv.circle(frame, center1, 15, (0,230,255), -1)  # -1 rellena el arco
                    if 120>=angle2 and angle2>= 110:
                        cv.circle(frame, center2, 15, (0,230,255), -1)  # -1 rellena el arco
                    if 110>=angle1 and angle1>= 100:
                        cv.circle(frame, center1, 16, (0,255,239), -1)  # -1 rellena el arco
                    if 110>=angle2 and angle2>= 100:
                        cv.circle(frame, center2, 16, (0,255,239), -1)  # -1 rellena el arco
                    if 100>=angle1 and angle1>= 90:
                        cv.circle(frame, center1, 17, (0,255,188), -1)  # -1 rellena el arco
                    if 100>=angle2 and angle2>= 90:
                        cv.circle(frame, center2, 17, (0,255,188), -1)  # -1 rellena el arco
                    if 90>=angle1 and angle1>= 80:
                        cv.circle(frame, center1, 18, (0,255,154), -1)  # -1 rellena el arco
                    if 90>=angle2 and angle2>= 80:
                        cv.circle(frame, center2, 18, (0,255,154), -1)  # -1 rellena el arco
                    if angle1<=80:
                        cv.circle(frame, center1, 19, (0,255,60), -1)  # -1 rellena el arco
                    if angle2<=80:
                        cv.circle(frame, center2, 19, (0,255,60), -1)  # -1 rellena el arco            

                (flag, encodedImage) = cv.imencode(".jpg", frame)  
                
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                    bytearray(encodedImage) + b'\r\n')

def squat(cap):
    #Llamado de objetos mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    up = False
    down = False
    count = 0
    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:            
            ret, frame = cap.read()
            height, width, _ = frame.shape
            if ret:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)                
                results = pose.process(frame)        
                if results.pose_landmarks is not None:
                    #Declaración del joint 11(hombro derecho)
                    hombroDerechoX = int(results.pose_landmarks.landmark[11].x * width)
                    hombroDerechoY = int(results.pose_landmarks.landmark[11].y * height)

                    #Declaración del joint 12(hombro izquierdo)
                    hombroIzquierdoX = int(results.pose_landmarks.landmark[12].x * width)
                    hombroIzquierdoY = int(results.pose_landmarks.landmark[12].y * height)
                    
                    #Declaración del joint 23(cadera derecha)
                    caderaDerechaX = int(results.pose_landmarks.landmark[23].x * width)
                    caderaDerechaY = int(results.pose_landmarks.landmark[23].y * height)
                    
                    #Declaración del joint 24(cadera izquierda)
                    caderaIzquierdaX = int(results.pose_landmarks.landmark[24].x * width)
                    caderaIzquierdaY = int(results.pose_landmarks.landmark[24].y * height)

                    #Declaración del joint 25(rodilla derecha)
                    rodillaDerechaX = int(results.pose_landmarks.landmark[25].x * width)
                    rodillaDerechaY = int(results.pose_landmarks.landmark[25].y * height)

                    #Declaración del joint 26(rodilla izquierda)
                    rodillaIzquierdaX = int(results.pose_landmarks.landmark[26].x * width)
                    rodillaIzquierdaY = int(results.pose_landmarks.landmark[26].y * height)

                    #Declaración del joint 27(tobillo derecha)
                    tobilloDerechoX = int(results.pose_landmarks.landmark[27].x * width)
                    tobilloDerechoY = int(results.pose_landmarks.landmark[27].y * height)

                    #Declaración del joint 28(tobillo izquierda)
                    tobilloIzquierdoX = int(results.pose_landmarks.landmark[28].x * width)
                    tobilloIzquierdoY = int(results.pose_landmarks.landmark[28].y * height)

                    #Declaración de puntos de referencia
                    hombroIZquierdo = np.array([hombroIzquierdoX, hombroIzquierdoY])
                    hombroDerecho = np.array([hombroDerechoX, hombroDerechoY])
                    caderaDerecha = np.array([caderaDerechaX, caderaDerechaY])
                    caderaIzquierda = np.array([caderaIzquierdaX, caderaIzquierdaY])
                    rodillaDerecha = np.array([rodillaDerechaX, rodillaDerechaY])
                    rodillaIzquierda = np.array([rodillaIzquierdaX, rodillaIzquierdaY])
                    tobilloDerecho = np.array([tobilloDerechoX, tobilloDerechoY])
                    tobilloIzquierdo = np.array([tobilloIzquierdoX, tobilloIzquierdoY])
                    
                    #Declaración de lineas en base a los puntos de referencia
                    l1 = np.linalg.norm(rodillaIzquierda-tobilloIzquierdo)
                    l2 = np.linalg.norm(caderaIzquierda-tobilloIzquierdo)
                    l3 = np.linalg.norm(caderaIzquierda-rodillaIzquierda)
                    l4 = np.linalg.norm(rodillaDerecha-tobilloDerecho)
                    l5 = np.linalg.norm(caderaDerecha-tobilloDerecho)
                    l6 = np.linalg.norm(caderaDerecha-rodillaDerecha)
                    l7 = np.linalg.norm(hombroIZquierdo-caderaIzquierda)
                    l8 = np.linalg.norm(hombroDerecho-caderaDerecha)
                    l9 = np.linalg.norm(hombroIZquierdo-rodillaIzquierda)
                    l10 = np.linalg.norm(hombroDerecho-rodillaDerecha)

                    #Cálculo de ángulos en base al tciángulo formado por los joints
                    angle1 = degrees(acos((l1**2 + l3**2 - l2**2) / (2*l1*l3)))
                    angle2 = degrees(acos((l4**2 + l6**2 - l5**2) / (2*l4*l6)))

                    angle3 = degrees(acos((l7**2 + l3**2 - l9**2) / (2*l7*l3)))
                    angle4 = degrees(acos((l8**2 + l6**2 - l10**2) / (2*l8*l6)))

                    angleKnee = (angle1 + angle2) / 2
                    angleHip = (angle3 + angle4) / 2

                    #Dibujado de joints
                    cv.circle(frame, (caderaIzquierdaX, caderaIzquierdaY), 6, (0,0,255), 4)
                    cv.circle(frame, (rodillaIzquierdaX, rodillaIzquierdaY), 6, (255,0,0), 6)
                    cv.circle(frame, (caderaDerechaX, caderaDerechaY), 6, (0,0,255), 4)
                    cv.circle(frame, (rodillaDerechaX, rodillaDerechaY), 6, (255,0,0), 6)
                    cv.circle(frame, (hombroDerechoX, hombroDerechoY), 6, (0,0,255), 4)
                    cv.circle(frame, (hombroIzquierdoX, hombroIzquierdoY), 6, (0,0,255), 4)

                    #Dibujado de lineas entre los joints
                    cv.line(frame, (caderaIzquierdaX, caderaIzquierdaY), (rodillaIzquierdaX, rodillaIzquierdaY), (255,0,0), 20)
                    cv.line(frame, (caderaDerechaX, caderaDerechaY), (rodillaDerechaX, rodillaDerechaY), (255,0,0), 20)
                    cv.line(frame, (hombroIzquierdoX, hombroIzquierdoY), (caderaIzquierdaX, caderaIzquierdaY), (255,0,0), 20)
                    cv.line(frame, (hombroDerechoX, hombroDerechoY), (caderaDerechaX, caderaDerechaY), (255,0,0), 20)
                    cv.line(frame, (rodillaDerechaX, rodillaDerechaY), (hombroDerechoX, hombroDerechoY), (0,0,255), 5)
                    cv.line(frame, (rodillaIzquierdaX, rodillaIzquierdaY), (hombroIzquierdoX, hombroIzquierdoY), (0,0,255), 5)
                
                    #Impresión de la imagen final
                    cv.rectangle(frame, (40,12), (90,50), (150,150,150), -1)
                    cv.putText(frame, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angleHip)), (100, 80), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angle2)), (caderaDerechaX-20, caderaDerechaY+50), 1, 1.5, (0, 255, 0), 2)
                    cv.putText(frame, str(int(angle1)), (caderaIzquierdaX-20, caderaIzquierdaY+50), 1, 1.5, (0, 255, 0), 2)

                    #Contar la repetición de una sentadilla válida
                    if angleHip >= 115: #Se encuentra acostado
                        th.Thread(target=text_to_speech, args=('Squat',)).start()
                        # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                        arrow_length = 100
                        arrow_thickness = 3
                        arrow_color = (0, 255, 0)  # Verde
                        cv.arrowedLine(frame, (width - arrow_length, arrow_length), (width - arrow_length, arrow_length * 2),arrow_color, thickness=arrow_thickness, tipLength=0.3)
                        down = True
                    if down == True and up == False and angleHip <= 110:
                        # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                        arrow_thickness = 3
                        arrow_color = (0, 0, 255)  # Verde
                        x_start = frame.shape[1] - arrow_length
                        y_start = arrow_length * 2
                        x_end = frame.shape[1] - arrow_length
                        y_end = arrow_length
                        cv.arrowedLine(frame, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)
                        up = True
                        # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                        arrow_length = 100
                        arrow_thickness = 3
                        arrow_color = (0, 0, 255)  # Verde
                        x_start = frame.shape[1] - arrow_length
                        y_start = arrow_length * 2
                        x_end = frame.shape[1] - arrow_length
                        y_end = arrow_length
                        cv.arrowedLine(frame, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)                                        
                    if up == True and down == True  and angleHip >=115:
                        count += 1
                        up = False
                        down = False
                        th.Thread(target=text_to_speech, args=('Repetición número ' + str(count),)).start()

                    # Dibujar el arco del área del ángulo
                    center1 = (caderaIzquierdaX, caderaIzquierdaY)  # Puedes ajustar el centro del arco según tus necesidades
                    radius = 50  # Puedes ajustar el radio del arco según tus necesidades
                    cv.ellipse(frame, center1, (radius, radius), 0, 360 - angle1 / 2, 360 + angle1 / 2, (0, 0, 255), -1)  # -1 rellena el arco
                    
                    # Dibujar el arco del área del ángulo
                    center2 = (caderaDerechaX, caderaDerechaY)  # Puedes ajustar el centro del arco según tus necesidades
                    radius = 50  # Puedes ajustar el radio del arco según tus necesidades
                    cv.ellipse(frame, center2, (radius, radius), 0, 180 - angle2 / 2, 180 + angle2 / 2, (0, 0, 255), -1)  # -1 rellena el arco
                    
                    # Dibujar el arco del área del ángulo
                    center3 = (rodillaIzquierdaX, rodillaIzquierdaY)  # Puedes ajustar el centro del arco según tus necesidades
                    radius = 50  # Puedes ajustar el radio del arco según tus necesidades
                    cv.ellipse(frame, center3, (radius, radius), 0, 360 - angle1 / 2, 360 + angle1 / 2, (0, 0, 255), -1)  # -1 rellena el arco
                    
                    # Dibujar el arco del área del ángulo
                    center4 = (rodillaDerechaX, rodillaDerechaY)  # Puedes ajustar el centro del arco según tus necesidades
                    radius = 50  # Puedes ajustar el radio del arco según tus necesidades
                    cv.ellipse(frame, center4, (radius, radius), 0, 180 - angle2 / 2, 180 + angle2 / 2, (0, 0, 255), -1)  # -1 rellena el arco                    

                (flag, encodedImage) = cv.imencode(".jpg", frame)  
                
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                    bytearray(encodedImage) + b'\r\n')

def pushup(cap):
    #Llamado de objetos mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    up = False
    down = False
    count = 0
    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:
            ret, frame = cap.read()
            height, width, _ = frame.shape
            if ret:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)                
                results = pose.process(frame)        
                if results.pose_landmarks is not None:                    
                    #Declaración del joint 11(hombro derecho)
                    hombroDerechoX = int(results.pose_landmarks.landmark[11]. x * width)
                    hombroDerechoY = int(results.pose_landmarks.landmark[11]. y * height)

                    #Declaración del joint 12(hombro izquierdo) 
                    hombroIzquierdoX = int(results.pose_landmarks.landmark[12].x * width)
                    hombroIzquierdoY = int(results.pose_landmarks.landmark[12].y * height)

                    #Declaración del joint 13(codo derecho)
                    codoDerechoX = int(results.pose_landmarks.landmark[13].x * width)
                    codoDerechoY = int(results.pose_landmarks.landmark[13].y * height)

                    #Declaración del joint 14(codo izquierdo)
                    codoIzquierdoX = int(results.pose_landmarks.landmark[14].x * width)
                    codoIzquierdoY = int(results.pose_landmarks.landmark[14].y * height)

                    #Declaración del joint 15(muñeca derecha)
                    munecaDerechaX = int(results.pose_landmarks.landmark[15].x * width)
                    munecaDerechaY = int(results.pose_landmarks.landmark[15].y * height)

                    #Declaración del joint 16(muñeca izquierda)
                    munecaIzquierdaX = int(results.pose_landmarks.landmark[16].x * width)
                    munecaIzquierdaY = int(results.pose_landmarks.landmark[16].y * height)

                    #Declaración de puntos de referencia
                    hi = np.array([hombroIzquierdoX, hombroIzquierdoY])
                    hr = np.array([hombroDerechoX, hombroDerechoY])

                    ci = np.array([codoIzquierdoX, codoIzquierdoY])
                    cr = np.array([codoDerechoX, codoDerechoY])

                    mi = np.array([munecaIzquierdaX, munecaIzquierdaY])
                    mr = np.array([munecaDerechaX, munecaDerechaY])

                    #Declaración de lineas en base a los puntos de referencia
                    l1 = np.linalg.norm(hi-ci)
                    l2 = np.linalg.norm(ci-mi)
                    l3 = np.linalg.norm(hi-mi)
                    l4 = np.linalg.norm(hr-cr)
                    l5 = np.linalg.norm(cr-mr)
                    l6 = np.linalg.norm(hr-mr)

                    #Cálculo de ángulos en base al triángulo formado por los joints
                    angle1 = degrees(acos((l1**2 + l2**2 - l3**2) / (2*l1*l2)))
                    angle2 = degrees(acos((l4**2 + l5**2 - l6**2) / (2*l4*l5)))

                    angleElbow = (angle1 + angle2)/2

                    #Dibujado de joints
                    cv.circle(frame, (hombroIzquierdoX, hombroIzquierdoY), 6, (0,0,255), 4)
                    cv.circle(frame, (codoIzquierdoX, codoIzquierdoY), 6, (255,0,0), 6)
                    cv.circle(frame, (munecaIzquierdaX, munecaIzquierdaY), 6, (0,0,255), 4)
                    cv.circle(frame, (hombroDerechoX, hombroDerechoY), 6, (0,0,255), 4)
                    cv.circle(frame, (codoDerechoX, codoDerechoY), 6, (255,0,0), 6)
                    cv.circle(frame, (munecaDerechaX, munecaDerechaY), 6, (0,0,255), 4)
                    
                    #Dibujado de lineas entre los joints
                    cv.line(frame, (hombroIzquierdoX, hombroIzquierdoY), (codoIzquierdoX, codoIzquierdoY), (255,0,0), 20)
                    cv.line(frame, (codoIzquierdoX, codoIzquierdoY), (munecaIzquierdaX, munecaIzquierdaY), (255,0,0), 20)
                    cv.line(frame, (hombroIzquierdoX, hombroIzquierdoY), (munecaIzquierdaX, munecaIzquierdaY), (0,0,255), 5)
                    cv.line(frame, (hombroDerechoX, hombroDerechoY), (codoDerechoX, codoDerechoY), (255,0,0), 20)
                    cv.line(frame, (codoDerechoX, codoDerechoY), (munecaDerechaX, munecaDerechaY), (255,0,0), 20)
                    cv.line(frame, (hombroDerechoX, hombroDerechoY), (munecaDerechaX, munecaDerechaY), (0,0,255), 5)

                    #Impresión de la imagen final
                    cv.putText(frame, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angleElbow)), (100, 50), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angle2)), (codoDerechoX+30, codoDerechoY), 1, 1.5, (0, 255, 0), 2)
                    cv.putText(frame, str(int(angle1)), (codoIzquierdoX+30, codoIzquierdoY), 1, 1.5, (0, 255, 0), 2)

                    #contar la repetición de una flexión válida
                    if angleElbow >= 170:                        
                        th.Thread(target=text_to_speech, args=('Listo para inciar',)).start()
                    if angleElbow <= 150 and 120 <= angleElbow:
                        arrow_length = 100
                        arrow_thickness = 3
                        arrow_color = (0, 255, 0)  # Verde
                        cv.arrowedLine(frame, (width - arrow_length, arrow_length), (width - arrow_length, arrow_length * 2),arrow_color, thickness=arrow_thickness, tipLength=0.3)                    
                        up = True                
                    if up == True and down == False and angleElbow <= 90:                                        
                        # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                        th.Thread(target=text_to_speech, args=('Puedes subir',)).start()
                        arrow_length = 100
                        arrow_thickness = 3
                        arrow_color = (0, 0, 255)  # Verde
                        x_start = frame.shape[1] - arrow_length
                        y_start = arrow_length * 2
                        x_end = frame.shape[1] - arrow_length
                        y_end = arrow_length
                        cv.arrowedLine(frame, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)
                        down=True
                    if up == True and down == True and angleElbow>=150:
                        count += 1
                        up = False
                        down = False
                        th.Thread(target=text_to_speech, args=('Repetición número ' + str(count),)).start()
                    if down == True and angleElbow<=65:
                        th.Thread(target=text_to_speech, args=('¡Está bajando demasiado!',)).start()
                    
                    # Dibujar el arco del área del ángulo
                    center1 = (codoIzquierdoX, codoIzquierdoY)  # Puedes ajustar el centro del arco según tus necesidades
                    center2 = (codoDerechoX, codoDerechoY)  # Puedes ajustar el centro del arco según tus necesidades
                    
                    # Dibuja los círculos en la imagen superpuesta
                    cv.circle(frame, center1, 20, (255,255,255), 1)  # -1 rellena el arco
                    cv.circle(frame, center2, 20, (255,255,255), 0)  # -1 rellena el arco                    
                    if angle1 >= 160:
                        cv.circle(frame, center1, 10, (0,0,255), -1)  # -1 rellena el arco
                    if angle2 >= 160:
                        cv.circle(frame, center2, 10, (0,0,255), -1)  # -1 rellena el arco
                    if 160>=angle1 and angle1>= 150:
                        cv.circle(frame, center1, 11, (0,65,255), -1)  # -1 rellena el arco                    
                    if 160>=angle2 and angle2>= 150:
                        cv.circle(frame, center2, 11, (0,65,255), -1)  # -1 rellena el arco
                    if 150>=angle1 and angle1>= 140:
                        cv.circle(frame, center1, 12, (0,119,255), -1)  # -1 rellena el arco                    
                    if 150>=angle2 and angle2>= 140:
                        cv.circle(frame, center2, 12, (0,119,255), -1)  # -1 rellena el arco
                    if 140>=angle1 and angle1>= 130:
                        cv.circle(frame, center1, 13, (0,154,230), -1)  # -1 rellena el arco
                    if 140>=angle2 and angle2>= 130:
                        cv.circle(frame, center2, 13, (0,154,230), -1)  # -1 rellena el arco
                    if 130>=angle1 and angle1>= 120:
                        cv.circle(frame, center1, 14, (0,205,255), -1)  # -1 rellena el arco                    
                    if 130>=angle2 and angle2>= 120:
                        cv.circle(frame, center2, 14, (0,205,255), -1)  # -1 rellena el arco
                    if 120>=angle1 and angle1>= 110:
                        cv.circle(frame, center1, 15, (0,230,255), -1)  # -1 rellena el arco
                    if 120>=angle2 and angle2>= 110:
                        cv.circle(frame, center2, 15, (0,230,255), -1)  # -1 rellena el arco
                    if 110>=angle1 and angle1>= 100:
                        cv.circle(frame, center1, 16, (0,255,239), -1)  # -1 rellena el arco
                    if 110>=angle2 and angle2>= 100:
                        cv.circle(frame, center2, 16, (0,255,239), -1)  # -1 rellena el arco
                    if 100>=angle1 and angle1>= 90:
                        cv.circle(frame, center1, 17, (0,255,188), -1)  # -1 rellena el arco
                    if 100>=angle2 and angle2>= 90:
                        cv.circle(frame, center2, 17, (0,255,188), -1)  # -1 rellena el arco
                    if 90>=angle1 and angle1>= 80:
                        cv.circle(frame, center1, 18, (0,255,154), -1)  # -1 rellena el arco
                    if 90>=angle2 and angle2>= 80:
                        cv.circle(frame, center2, 18, (0,255,154), -1)  # -1 rellena el arco
                    if angle1<=80:
                        cv.circle(frame, center1, 19, (0,255,60), -1)  # -1 rellena el arco
                    if angle2<=80:
                        cv.circle(frame, center2, 19, (0,255,60), -1)  # -1 rellena el arco            

                (flag, encodedImage) = cv.imencode(".jpg", frame)  
                
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                    bytearray(encodedImage) + b'\r\n')

def ohp(cap):
    #Llamado de objetos mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    up = False
    down = False
    count = 0
    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:            
            ret, frame = cap.read()
            height, width, _ = frame.shape
            if ret:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)                
                results = pose.process(frame)        
                if results.pose_landmarks is not None:
                    #Declaración del joint 11(hombro derecho)
                    hombroDerechoX = int(results.pose_landmarks.landmark[11].x * width)
                    hombroDerechoY = int(results.pose_landmarks.landmark[11].y * height)

                    #Declaración del joint 12(hombro izquierdo)
                    hombroIzquierdoX = int(results.pose_landmarks.landmark[12].x * width)
                    hombroIzquierdoY = int(results.pose_landmarks.landmark[12].y * height)

                    #Declaración del joint 13(codo derecho)
                    codoDerechoX = int(results.pose_landmarks.landmark[13].x * width)
                    codoDerechoY = int(results.pose_landmarks.landmark[13].y * height)

                    #Declaración del joint 14(codo izquierdo)
                    codoIzquierdoX = int(results.pose_landmarks.landmark[14].x * width)
                    codoIzquierdoY = int(results.pose_landmarks.landmark[14].y * height)

                    #Declaración del joint 15(muñeca derecha)
                    munecaDerechaX = int(results.pose_landmarks.landmark[15].x * width)
                    munecaDerechaY = int(results.pose_landmarks.landmark[15].y * height)

                    #Declaración del joint 16(muñeca izquierda)
                    munecaIzquierdaX = int(results.pose_landmarks.landmark[16].x * width)
                    munecaIzquierdaY = int(results.pose_landmarks.landmark[16].y * height)

                    #Declaración de puntos de referencia
                    hi = np.array([hombroIzquierdoX, hombroIzquierdoY])
                    hr = np.array([hombroDerechoX, hombroDerechoY])

                    ci = np.array([codoIzquierdoX, codoIzquierdoY])
                    cr = np.array([codoDerechoX, codoDerechoY])

                    mi = np.array([munecaIzquierdaX, munecaIzquierdaY])
                    mr = np.array([munecaDerechaX, munecaDerechaY])

                    #Declaración de lineas en base a los puntos de referencia
                    l1 = np.linalg.norm(hi-ci)
                    l2 = np.linalg.norm(ci-mi)
                    l3 = np.linalg.norm(hi-mi)
                    l4 = np.linalg.norm(hr-cr)
                    l5 = np.linalg.norm(cr-mr)
                    l6 = np.linalg.norm(hr-mr)

                    #Cálculo de ángulos en base al triángulo formado por los joints
                    angle1 = degrees(acos((l1**2 + l2**2 - l3**2) / (2*l1*l2)))
                    angle2 = degrees(acos((l4**2 + l5**2 - l6**2) / (2*l4*l5)))

                    angleElbow = (angle1 + angle2)/2     

                    #Dibujado de joints
                    cv.circle(frame, (hombroIzquierdoX, hombroIzquierdoY), 6, (0,0,255), 4)
                    cv.circle(frame, (codoIzquierdoX, codoIzquierdoY), 6, (255,0,0), 6)
                    cv.circle(frame, (munecaIzquierdaX, munecaIzquierdaY), 6, (0,0,255), 4)
                    cv.circle(frame, (hombroDerechoX, hombroDerechoY), 6, (0,0,255), 4)
                    cv.circle(frame, (codoDerechoX, codoDerechoY), 6, (255,0,0), 6)
                    cv.circle(frame, (munecaDerechaX, munecaDerechaY), 6, (0,0,255), 4)

                    #Dibujado de lineas entre los joints
                    cv.line(frame, (hombroIzquierdoX, hombroIzquierdoY), (codoIzquierdoX, codoIzquierdoY), (255,0,0), 20)
                    cv.line(frame, (codoIzquierdoX, codoIzquierdoY), (munecaIzquierdaX, munecaIzquierdaY), (255,0,0), 20)
                    cv.line(frame, (hombroIzquierdoX, hombroIzquierdoY), (munecaIzquierdaX, munecaIzquierdaY), (0,0,255), 5)
                    cv.line(frame, (hombroDerechoX, hombroDerechoY), (codoDerechoX, codoDerechoY), (255,0,0), 20)
                    cv.line(frame, (codoDerechoX, codoDerechoY), (munecaDerechaX, munecaDerechaY), (255,0,0), 20)
                    cv.line(frame, (hombroDerechoX, hombroDerechoY), (munecaDerechaX, munecaDerechaY), (0,0,255), 5)

                    #Impresión de la imagen final
                    cv.putText(frame, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angleElbow)), (100, 80), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angle2)), (codoDerechoX+30, codoDerechoY), 1, 1.5, (0, 255, 0), 2)
                    cv.putText(frame, str(int(angle1)), (codoIzquierdoX+30, codoIzquierdoY), 1, 1.5, (0, 255, 0), 2)

                    #contar la repetición de una flexión válida
                    if angleElbow <= 150 and 120 <= angleElbow:
                        th.Thread(target=text_to_speech, args=('Puedes subir',)).start()
                        arrow_length = 100
                        arrow_thickness = 3
                        arrow_color = (0, 255, 0)  # Verde
                        cv.arrowedLine(frame, (width - arrow_length, arrow_length), (width - arrow_length, arrow_length * 2),arrow_color, thickness=arrow_thickness, tipLength=0.3)                    
                        up = True                
                    if up == True and down == False and angleElbow <= 90:                                        
                        # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                        arrow_length = 100
                        arrow_thickness = 3
                        arrow_color = (0, 0, 255)  # Verde
                        x_start = frame.shape[1] - arrow_length
                        y_start = arrow_length * 2
                        x_end = frame.shape[1] - arrow_length
                        y_end = arrow_length
                        cv.arrowedLine(frame, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)
                        down=True
                    if up == True and down == True and angleElbow>=150:
                        count += 1
                        up = False
                        down = False
                        th.Thread(target=text_to_speech, args=('Repetición número ' + str(count),)).start()
                    if down == True and angleElbow<=65:
                        th.Thread(target=text_to_speech, args=('¡Está bajando demasiado!',)).start()
                    
                    # Dibujar el arco del área del ángulo
                    center1 = (codoIzquierdoX, codoIzquierdoY)  # Puedes ajustar el centro del arco según tus necesidades
                    center2 = (codoDerechoX, codoDerechoY)  # Puedes ajustar el centro del arco según tus necesidades
                    
                    # Dibuja los círculos en la imagen superpuesta

                    cv.circle(frame, center1, 20, (255,255,255), 1)  # -1 rellena el arco
                    cv.circle(frame, center2, 20, (255,255,255), 0)  # -1 rellena el arco                    
                    if angle1 >= 160:
                        cv.circle(frame, center1, 10, (0,0,255), -1)  # -1 rellena el arco
                    if angle2 >= 160:
                        cv.circle(frame, center2, 10, (0,0,255), -1)  # -1 rellena el arco
                    if 160>=angle1 and angle1>= 150:
                        cv.circle(frame, center1, 11, (0,65,255), -1)  # -1 rellena el arco                    
                    if 160>=angle2 and angle2>= 150:
                        cv.circle(frame, center2, 11, (0,65,255), -1)  # -1 rellena el arco
                    if 150>=angle1 and angle1>= 140:
                        cv.circle(frame, center1, 12, (0,119,255), -1)  # -1 rellena el arco                    
                    if 150>=angle2 and angle2>= 140:
                        cv.circle(frame, center2, 12, (0,119,255), -1)  # -1 rellena el arco
                    if 140>=angle1 and angle1>= 130:
                        cv.circle(frame, center1, 13, (0,154,230), -1)  # -1 rellena el arco
                    if 140>=angle2 and angle2>= 130:
                        cv.circle(frame, center2, 13, (0,154,230), -1)  # -1 rellena el arco
                    if 130>=angle1 and angle1>= 120:
                        cv.circle(frame, center1, 14, (0,205,255), -1)  # -1 rellena el arco                    
                    if 130>=angle2 and angle2>= 120:
                        cv.circle(frame, center2, 14, (0,205,255), -1)  # -1 rellena el arco
                    if 120>=angle1 and angle1>= 110:
                        cv.circle(frame, center1, 15, (0,230,255), -1)  # -1 rellena el arco
                    if 120>=angle2 and angle2>= 110:
                        cv.circle(frame, center2, 15, (0,230,255), -1)  # -1 rellena el arco
                    if 110>=angle1 and angle1>= 100:
                        cv.circle(frame, center1, 16, (0,255,239), -1)  # -1 rellena el arco
                    if 110>=angle2 and angle2>= 100:
                        cv.circle(frame, center2, 16, (0,255,239), -1)  # -1 rellena el arco
                    if 100>=angle1 and angle1>= 90:
                        cv.circle(frame, center1, 17, (0,255,188), -1)  # -1 rellena el arco
                    if 100>=angle2 and angle2>= 90:
                        cv.circle(frame, center2, 17, (0,255,188), -1)  # -1 rellena el arco
                    if 90>=angle1 and angle1>= 80:
                        cv.circle(frame, center1, 18, (0,255,154), -1)  # -1 rellena el arco
                    if 90>=angle2 and angle2>= 80:
                        cv.circle(frame, center2, 18, (0,255,154), -1)  # -1 rellena el arco
                    if angle1<=80:
                        cv.circle(frame, center1, 19, (0,255,60), -1)  # -1 rellena el arco
                    if angle2<=80:
                        cv.circle(frame, center2, 19, (0,255,60), -1)  # -1 rellena el arco            

                (flag, encodedImage) = cv.imencode(".jpg", frame)  
                
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                    bytearray(encodedImage) + b'\r\n')

def bicep_curl(cap):
    #Llamado de objetos mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    up = False
    down = False
    count = 0
    with mp_pose.Pose(static_image_mode=False) as pose:
        while True:            
            ret, frame = cap.read()
            height, width, _ = frame.shape
            if ret:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)                
                results = pose.process(frame)        
                if results.pose_landmarks is not None:
                    #Declaración del joint 0(nariz)
                    narizX = int(results.pose_landmarks.landmark[0].x * width)
                    narizY = int(results.pose_landmarks.landmark[0].y * height)

                    #Declaración del joint 11(hombro derecho)
                    hombroDerechoX = int(results.pose_landmarks.landmark[11].x * width)
                    hombroDerechoY = int(results.pose_landmarks.landmark[11].y * height)

                    #Declaración del joint 12(hombro izquierdo)
                    hombroIzquierdoX = int(results.pose_landmarks.landmark[12].x * width)
                    hombroIzquierdoY = int(results.pose_landmarks.landmark[12].y * height)

                    #Declaración del joint 13(codo derecho)
                    codoDerechoX = int(results.pose_landmarks.landmark[13].x * width)
                    codoDerechoY = int(results.pose_landmarks.landmark[13].y * height)

                    #Declaración del joint 14(codo izquierdo)
                    codoIzquierdoX = int(results.pose_landmarks.landmark[14].x * width)
                    codoIzquierdoY = int(results.pose_landmarks.landmark[14].y * height)

                    #Declaración del joint 15(muñeca derecha)
                    munecaDerechaX = int(results.pose_landmarks.landmark[15].x * width)
                    munecaDerechaY = int(results.pose_landmarks.landmark[15].y * height)

                    #Declaración del joint 16(muñeca izquierda)
                    munecaIzquierdaX = int(results.pose_landmarks.landmark[16].x * width)
                    munecaIzquierdaY = int(results.pose_landmarks.landmark[16].y * height)

                    #Declaración del joint 23 (cadera derecha)
                    caderaDerechaX = int(results.pose_landmarks.landmark[23].x * width)
                    caderaDerechaY = int(results.pose_landmarks.landmark[23].y * height)

                    #Declaración del joint 24 (cadera izquierda)
                    caderaIzquierdaX = int(results.pose_landmarks.landmark[24].x * width)
                    caderaIzquierdaY = int(results.pose_landmarks.landmark[24].y * height)

                    #Declaración del joint 25(rodilla derecha)
                    rodillaDerechaX = int(results.pose_landmarks.landmark[25].x * width)
                    rodillaDerechaY = int(results.pose_landmarks.landmark[25].y * height)

                    #Declaración del joint 26(rodilla izquierda)
                    rodillaIzquierdaX = int(results.pose_landmarks.landmark[26].x * width)
                    rodillaIzquierdaY = int(results.pose_landmarks.landmark[26].y * height)

                    nariz = np.array([narizX, narizY])
                    hombroDerecho = np.array([hombroDerechoX, hombroDerechoY])
                    hombroIzquierdo = np.array([hombroIzquierdoX, hombroIzquierdoY])
                    codoDerecho = np.array([codoDerechoX, codoDerechoY])
                    codoIzquierdo = np.array([codoIzquierdoX, codoIzquierdoY])
                    munecaDerecha = np.array([munecaDerechaX, munecaDerechaY])
                    munecaIzquierda = np.array([munecaIzquierdaX, munecaIzquierdaY])
                    caderaDerecha = np.array([caderaDerechaX, caderaDerechaY])
                    caderaIzquierda = np.array([caderaIzquierdaX, caderaIzquierdaY])
                    rodillaDerecha = np.array([rodillaDerechaX, rodillaDerechaY])
                    rodillaIzquierda = np.array([rodillaIzquierdaX, rodillaIzquierdaY])

                    #Declaración de lineas en base a los puntos de referencia
                    lineaHombroCodoDerecho = np.linalg.norm(hombroDerecho-codoDerecho)
                    lineaCodoMunecaDerecho = np.linalg.norm(codoDerecho-munecaDerecha)
                    lineaMunecaHombroDerecho = np.linalg.norm(munecaDerecha-hombroDerecho)

                    lineaHombroCodoIzquierdo = np.linalg.norm(hombroIzquierdo-codoIzquierdo)
                    lineaCodoMunecaIzquierdo = np.linalg.norm(codoIzquierdo-munecaIzquierda)
                    lineaMunecaHombroIzquierdo = np.linalg.norm(munecaIzquierda-hombroIzquierdo)

                    #Cálculo de ángulos en base al triángulo formado por los joint
                    angle1 = degrees(acos((lineaHombroCodoDerecho**2 + lineaCodoMunecaDerecho**2 - lineaMunecaHombroDerecho**2) / (2*lineaHombroCodoDerecho*lineaCodoMunecaDerecho)))
                    angle2 = degrees(acos((lineaHombroCodoIzquierdo**2 + lineaCodoMunecaIzquierdo**2 - lineaMunecaHombroIzquierdo**2) / (2*lineaHombroCodoIzquierdo*lineaCodoMunecaIzquierdo)))

                    angleElbow = (angle1 + angle2)/2     

                    #Dibujado de joints
                    cv.circle(frame, (hombroIzquierdoX, hombroIzquierdoY), 6, (0,0,255), 4)
                    cv.circle(frame, (codoIzquierdoX, codoIzquierdoY), 6, (255,0,0), 6)
                    cv.circle(frame, (munecaIzquierdaX, munecaIzquierdaY), 6, (0,0,255), 4)
                    cv.circle(frame, (hombroDerechoX, hombroDerechoY), 6, (0,0,255), 4)
                    cv.circle(frame, (codoDerechoX, codoDerechoY), 6, (255,0,0), 6)
                    cv.circle(frame, (munecaDerechaX, munecaDerechaY), 6, (0,0,255), 4)

                    #Dibujado de lineas entre los joints
                    cv.line(frame, (hombroIzquierdoX, hombroIzquierdoY), (codoIzquierdoX, codoIzquierdoY), (255,0,0), 20)
                    cv.line(frame, (codoIzquierdoX, codoIzquierdoY), (munecaIzquierdaX, munecaIzquierdaY), (255,0,0), 20)
                    cv.line(frame, (hombroIzquierdoX, hombroIzquierdoY), (munecaIzquierdaX, munecaIzquierdaY), (0,0,255), 5)
                    cv.line(frame, (hombroDerechoX, hombroDerechoY), (codoDerechoX, codoDerechoY), (255,0,0), 20)
                    cv.line(frame, (codoDerechoX, codoDerechoY), (munecaDerechaX, munecaDerechaY), (255,0,0), 20)
                    cv.line(frame, (hombroDerechoX, hombroDerechoY), (munecaDerechaX, munecaDerechaY), (0,0,255), 5)

                    #Impresión de la imagen final
                    cv.putText(frame, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angleElbow)), (100, 80), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angle2)), (codoDerechoX+30, codoDerechoY), 1, 1.5, (0, 255, 0), 2)
                    cv.putText(frame, str(int(angle1)), (codoIzquierdoX+30, codoIzquierdoY), 1, 1.5, (0, 255, 0), 2)

                    #contar la repetición de una flexión válida
                    if angleElbow >= 160:                        
                        th.Thread(target=text_to_speech, args=('bicep curl',)).start()
                    if angleElbow <= 160 and 130 <= angleElbow:
                        arrow_length = 100
                        arrow_thickness = 3
                        arrow_color = (0, 255, 0)  # Verde
                        cv.arrowedLine(frame, (width - arrow_length, arrow_length), (width - arrow_length, arrow_length * 2),arrow_color, thickness=arrow_thickness, tipLength=0.3)                    
                        up = True                
                    if up == True and down == False and angleElbow <= 90:                                        
                        # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                        th.Thread(target=text_to_speech, args=('Puedes bajar',)).start()
                        arrow_length = 100
                        arrow_thickness = 3
                        arrow_color = (0, 0, 255)  # Verde
                        x_start = frame.shape[1] - arrow_length
                        y_start = arrow_length * 2
                        x_end = frame.shape[1] - arrow_length
                        y_end = arrow_length
                        cv.arrowedLine(frame, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)
                        down=True
                        # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                        arrow_length = 100
                        arrow_thickness = 3
                        arrow_color = (0, 0, 255)  # Verde
                        x_start = frame.shape[1] - arrow_length
                        y_start = arrow_length * 2
                        x_end = frame.shape[1] - arrow_length
                        y_end = arrow_length
                        cv.arrowedLine(frame, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)
                    if up == True and down == True and angleElbow>=150:
                        count += 1
                        up = False
                        down = False
                        th.Thread(target=text_to_speech, args=('Repetición número ' + str(count),)).start()
                    if down == True and angleElbow<=65:
                        th.Thread(target=text_to_speech, args=('¡Está bajando demasiado!',)).start()
                    
                    # Dibujar el arco del área del ángulo
                    center1 = (codoIzquierdoX, codoIzquierdoY)  # Puedes ajustar el centro del arco según tus necesidades
                    center2 = (codoDerechoX, codoDerechoY)  # Puedes ajustar el centro del arco según tus necesidades
                    
                    # Dibuja los círculos en la imagen superpuesta

                    cv.circle(frame, center1, 20, (255,255,255), 1)  # -1 rellena el arco
                    cv.circle(frame, center2, 20, (255,255,255), 0)  # -1 rellena el arco                    
                    if angle1 >= 160:
                        cv.circle(frame, center1, 10, (0,0,255), -1)  # -1 rellena el arco
                    if angle2 >= 160:
                        cv.circle(frame, center2, 10, (0,0,255), -1)  # -1 rellena el arco
                    if 160>=angle1 and angle1>= 150:
                        cv.circle(frame, center1, 11, (0,65,255), -1)  # -1 rellena el arco                    
                    if 160>=angle2 and angle2>= 150:
                        cv.circle(frame, center2, 11, (0,65,255), -1)  # -1 rellena el arco
                    if 150>=angle1 and angle1>= 140:
                        cv.circle(frame, center1, 12, (0,119,255), -1)  # -1 rellena el arco                    
                    if 150>=angle2 and angle2>= 140:
                        cv.circle(frame, center2, 12, (0,119,255), -1)  # -1 rellena el arco
                    if 140>=angle1 and angle1>= 130:
                        cv.circle(frame, center1, 13, (0,154,230), -1)  # -1 rellena el arco
                    if 140>=angle2 and angle2>= 130:
                        cv.circle(frame, center2, 13, (0,154,230), -1)  # -1 rellena el arco
                    if 130>=angle1 and angle1>= 120:
                        cv.circle(frame, center1, 14, (0,205,255), -1)  # -1 rellena el arco                    
                    if 130>=angle2 and angle2>= 120:
                        cv.circle(frame, center2, 14, (0,205,255), -1)  # -1 rellena el arco
                    if 120>=angle1 and angle1>= 110:
                        cv.circle(frame, center1, 15, (0,230,255), -1)  # -1 rellena el arco
                    if 120>=angle2 and angle2>= 110:
                        cv.circle(frame, center2, 15, (0,230,255), -1)  # -1 rellena el arco
                    if 110>=angle1 and angle1>= 100:
                        cv.circle(frame, center1, 16, (0,255,239), -1)  # -1 rellena el arco
                    if 110>=angle2 and angle2>= 100:
                        cv.circle(frame, center2, 16, (0,255,239), -1)  # -1 rellena el arco
                    if 100>=angle1 and angle1>= 90:
                        cv.circle(frame, center1, 17, (0,255,188), -1)  # -1 rellena el arco
                    if 100>=angle2 and angle2>= 90:
                        cv.circle(frame, center2, 17, (0,255,188), -1)  # -1 rellena el arco
                    if 90>=angle1 and angle1>= 80:
                        cv.circle(frame, center1, 18, (0,255,154), -1)  # -1 rellena el arco
                    if 90>=angle2 and angle2>= 80:
                        cv.circle(frame, center2, 18, (0,255,154), -1)  # -1 rellena el arco
                    if angle1<=80:
                        cv.circle(frame, center1, 19, (0,255,60), -1)  # -1 rellena el arco
                    if angle2<=80:
                        cv.circle(frame, center2, 19, (0,255,60), -1)  # -1 rellena el arco            

                (flag, encodedImage) = cv.imencode(".jpg", frame)  
                
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                    bytearray(encodedImage) + b'\r\n')

