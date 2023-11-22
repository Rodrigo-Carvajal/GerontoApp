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
hombreDerechoX = int(results.pose_landmarks.landmark[11].x * width)
hombreDerechoY = int(results.pose_landmarks.landmark[11].y * height)

#Declaración del joint 12(hombro izquierdo)
hombreIzquierdoX = int(results.pose_landmarks.landmark[12].x * width)
hombreIzquierdoY = int(results.pose_landmarks.landmark[12].y * height)

#Declaración del joint 13(codo derecho)
codoDerechoX = int(results.pose_landmarks.landmark[13].x * width)
codoDerechoY = int(results.pose_landmarks.landmark[13].y * height)

#Declaración del joint 14(codo izquierdo)
codoIzquierdoX = int(results.pose_landmarks.landmark[14].x * width)
codoIzuiqerdoY = int(results.pose_landmarks.landmark[14].y * height)

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
hombroDerecho = np.array([hombreoDerecho, hombreoDerecho])
hombroIZquierdo = np.array([hombroIzquierdo, hombroIzquierdo])
codoDerecho = np.array([codoDerecho, codoDerecho])
codoIzquierdo = np.array([codoIzquierdo, codoIzquierdo])
munecaDerecha = np.array([munecaDerecha, munecaDerecha])
munecaIzquierda = np.array([munecaIzquierda, munecaIzquierda])
caderaDerecha = np.array([caderaDerecha, caderaDerecha])
caderaIzquierda = np.array([caderaIzquierda, caderaIzquierda])
rodillaDerecha = np.array([rodillaDerecha, rodillaDerecha])
rodillaIzquierda = np.array([rodillaIzquierda, rodillaIzquierda])
tobilloDerecho = np.array([tobilloDerecho, tobilloDerecho])
tobilloIzquierdo = np.array([tobilloIzquierdo, tobilloIzquierdo])
pieDerecho = np.array([pieDerecho, pieDerecho])
pieIzquierdo = np.array([pieIzquierdo, pieIzquierdo])

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
    return 'lateral raise'

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
                    rsw = int(results.pose_landmarks.landmark[11].x * width)
                    rsh = int(results.pose_landmarks.landmark[11].y * height)

                    #Declaración del joint 12(hombro izquierdo)
                    lsw = int(results.pose_landmarks.landmark[12].x * width)
                    lsh = int(results.pose_landmarks.landmark[12].y * height)
                    
                    #Declaración del joint 23(cadera derecha)
                    rhw = int(results.pose_landmarks.landmark[23].x * width)
                    rhh = int(results.pose_landmarks.landmark[23].y * height)
                    
                    #Declaración del joint 24(cadera izquierda)
                    lhw = int(results.pose_landmarks.landmark[24].x * width)
                    lhh = int(results.pose_landmarks.landmark[24].y * height)

                    #Declaración del joint 25(rodilla derecha)
                    rkw = int(results.pose_landmarks.landmark[25].x * width)
                    rkh = int(results.pose_landmarks.landmark[25].y * height)

                    #Declaración del joint 26(rodilla izquierda)
                    lkw = int(results.pose_landmarks.landmark[26].x * width)
                    lkh = int(results.pose_landmarks.landmark[26].y * height)

                    #Declaración del joint 27(tobillo derecha)
                    raw = int(results.pose_landmarks.landmark[27].x * width)
                    rah = int(results.pose_landmarks.landmark[27].y * height)
                    
                    #Declaración del joint 28(tobillo izquierda)
                    law = int(results.pose_landmarks.landmark[28].x * width)
                    lah = int(results.pose_landmarks.landmark[28].y * height)
                
                    #Declaración de puntos de referencia
                    hi = np.array([lsw, lsh])
                    hr = np.array([rsw, rsh])

                    ci = np.array([lhw, lhh])
                    cr = np.array([rhw, rhh])
                    
                    ti = np.array([law, lah])
                    tr = np.array([raw, rah])

                    ri = np.array([lkw, lkh])
                    rr = np.array([rkw, rkh])
                    
                    #Declaración de lineas en base a los puntos de referencia
                    l1 = np.linalg.norm(ri-ti)
                    l2 = np.linalg.norm(ci-ti)
                    l3 = np.linalg.norm(ci-ri)
                    l4 = np.linalg.norm(rr-tr)
                    l5 = np.linalg.norm(cr-tr)
                    l6 = np.linalg.norm(cr-rr)
                    l7 = np.linalg.norm(hi-ci)
                    l8 = np.linalg.norm(hr-cr)
                    l9 = np.linalg.norm(hi-ri)
                    l10 = np.linalg.norm(hr-rr)

                    #Cálculo de ángulos en base al tciángulo formado por los joints
                    angle1 = degrees(acos((l1**2 + l3**2 - l2**2) / (2*l1*l3)))
                    angle2 = degrees(acos((l4**2 + l6**2 - l5**2) / (2*l4*l6)))

                    angle3 = degrees(acos((l7**2 + l3**2 - l9**2) / (2*l7*l3)))
                    angle4 = degrees(acos((l8**2 + l6**2 - l10**2) / (2*l8*l6)))

                    angleKnee = (angle1 + angle2) / 2
                    angleHip = (angle3 + angle4) / 2


                    #Dibujado de joints
                    cv.circle(frame, (lhw, lhh), 6, (0,0,255), 4)
                    cv.circle(frame, (lkw, lkh), 6, (255,0,0), 6)
                    cv.circle(frame, (rhw, rhh), 6, (0,0,255), 4)
                    cv.circle(frame, (rkw, rkh), 6, (255,0,0), 6)
                    cv.circle(frame, (rsw, rsh), 6, (0,0,255), 4)
                    cv.circle(frame, (lsw, lsh), 6, (0,0,255), 4)

                    #Dibujado de lineas entre los joints
                    cv.line(frame, (lhw, lhh), (lkw, lkh), (255,0,0), 20)
                    cv.line(frame, (rhw, rhh), (rkw, rkh), (255,0,0), 20)
                    cv.line(frame, (lsw, lsh), (lhw, lhh), (255,0,0), 20)
                    cv.line(frame, (rsw, rsh), (rhw, rhh), (255,0,0), 20)
                    cv.line(frame, (rkw, rkh), (rsw, rsh), (0,0,255), 5)
                    cv.line(frame, (lkw, lkh), (lsw, lsh), (0,0,255), 5)
                
                    #Impresión de la imagen final
                    cv.rectangle(frame, (40,12), (90,50), (150,150,150), -1)
                    cv.putText(frame, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angleHip)), (100, 80), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angle2)), (rhw-20, rhh+50), 1, 1.5, (0, 255, 0), 2)
                    cv.putText(frame, str(int(angle1)), (lhw-20, lhh+50), 1, 1.5, (0, 255, 0), 2)

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
                        arrow_length = 100
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
                    center1 = (lhw, lhh)  # Puedes ajustar el centro del arco según tus necesidades
                    radius = 50  # Puedes ajustar el radio del arco según tus necesidades
                    cv.ellipse(frame, center1, (radius, radius), 0, 360 - angle1 / 2, 360 + angle1 / 2, (0, 0, 255), -1)  # -1 rellena el arco
                    
                    # Dibujar el arco del área del ángulo
                    center2 = (rhw, rhh)  # Puedes ajustar el centro del arco según tus necesidades
                    radius = 50  # Puedes ajustar el radio del arco según tus necesidades
                    cv.ellipse(frame, center2, (radius, radius), 0, 180 - angle2 / 2, 180 + angle2 / 2, (0, 0, 255), -1)  # -1 rellena el arco
                    
                    # Dibujar el arco del área del ángulo
                    center3 = (lkw, lkh)  # Puedes ajustar el centro del arco según tus necesidades
                    radius = 50  # Puedes ajustar el radio del arco según tus necesidades
                    cv.ellipse(frame, center3, (radius, radius), 0, 360 - angle1 / 2, 360 + angle1 / 2, (0, 0, 255), -1)  # -1 rellena el arco
                    
                    # Dibujar el arco del área del ángulo
                    center4 = (rkw, rkh)  # Puedes ajustar el centro del arco según tus necesidades
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
                    rsw = int(results.pose_landmarks.landmark[11]. x * width)
                    rsh = int(results.pose_landmarks.landmark[11]. y * height)

                    #Declaración del joint 12(hombro izquierdo) 
                    lsw = int(results.pose_landmarks.landmark[12].x * width)
                    lsh = int(results.pose_landmarks.landmark[12].y * height)

                    #Declaración del joint 13(codo derecho)
                    rew = int(results.pose_landmarks.landmark[13].x * width)
                    reh = int(results.pose_landmarks.landmark[13].y * height)

                    #Declaración del joint 14(codo izquierdo)
                    lew = int(results.pose_landmarks.landmark[14].x * width)
                    leh = int(results.pose_landmarks.landmark[14].y * height)

                    #Declaración del joint 15(muñeca derecha)
                    rww = int(results.pose_landmarks.landmark[15].x * width)
                    rwh = int(results.pose_landmarks.landmark[15].y * height)

                    #Declaración del joint 16(muñeca izquierda)
                    lww = int(results.pose_landmarks.landmark[16].x * width)
                    lwh = int(results.pose_landmarks.landmark[16].y * height)

                    #Declaración de puntos de referencia
                    hi = np.array([lsw, lsh])
                    hr = np.array([rsw, rsh])

                    ci = np.array([lew, leh])
                    cr = np.array([rew, reh])

                    mi = np.array([lww, lwh])
                    mr = np.array([rww, rwh])

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
                    cv.circle(frame, (lsw, lsh), 6, (0,0,255), 4)
                    cv.circle(frame, (lew, leh), 6, (255,0,0), 6)
                    cv.circle(frame, (lww, lwh), 6, (0,0,255), 4)
                    cv.circle(frame, (rsw, rsh), 6, (0,0,255), 4)
                    cv.circle(frame, (rew, reh), 6, (255,0,0), 6)
                    cv.circle(frame, (rww, rwh), 6, (0,0,255), 4)
                    
                    #Dibujado de lineas entre los joints
                    cv.line(frame, (lsw, lsh), (lew, leh), (255,0,0), 20)
                    cv.line(frame, (lew, leh), (lww, lwh), (255,0,0), 20)
                    cv.line(frame, (lsw, lsh), (lww, lwh), (0,0,255), 5)
                    cv.line(frame, (rsw, rsh), (rew, reh), (255,0,0), 20)
                    cv.line(frame, (rew, reh), (rww, rwh), (255,0,0), 20)
                    cv.line(frame, (rsw, rsh), (rww, rwh), (0,0,255), 5)

                    #Impresión de la imagen final
                    cv.putText(frame, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angleElbow)), (100, 50), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angle2)), (rew+30, reh), 1, 1.5, (0, 255, 0), 2)
                    cv.putText(frame, str(int(angle1)), (lew+30, leh), 1, 1.5, (0, 255, 0), 2)

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
                        threading.Thread(target=text_to_speech, args=('Repetición número ' + str(count),)).start()
                    if down == True and angleElbow<=65:
                        th.Thread(target=text_to_speech, args=('¡Está bajando demasiado!',)).start()
                    
                    # Dibujar el arco del área del ángulo
                    center1 = (lew, leh)  # Puedes ajustar el centro del arco según tus necesidades
                    center2 = (rew, reh)  # Puedes ajustar el centro del arco según tus necesidades
                    
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
                    rsw = int(results.pose_landmarks.landmark[11].x * width)
                    rsh = int(results.pose_landmarks.landmark[11].y * height)

                    #Declaración del joint 12(hombro izquierdo)
                    lsw = int(results.pose_landmarks.landmark[12].x * width)
                    lsh = int(results.pose_landmarks.landmark[12].y * height)

                    #Declaración del joint 13(codo derecho)
                    rew = int(results.pose_landmarks.landmark[13].x * width)
                    reh = int(results.pose_landmarks.landmark[13].y * height)

                    #Declaración del joint 14(codo izquierdo)
                    lew = int(results.pose_landmarks.landmark[14].x * width)
                    leh = int(results.pose_landmarks.landmark[14].y * height)

                    #Declaración del joint 15(muñeca derecha)
                    rww = int(results.pose_landmarks.landmark[15].x * width)
                    rwh = int(results.pose_landmarks.landmark[15].y * height)

                    #Declaración del joint 16(muñeca izquierda)
                    lww = int(results.pose_landmarks.landmark[16].x * width)
                    lwh = int(results.pose_landmarks.landmark[16].y * height)

                    #Declaración de puntos de referencia
                    hi = np.array([lsw, lsh])
                    hr = np.array([rsw, rsh])

                    ci = np.array([lew, leh])
                    cr = np.array([rew, reh])

                    mi = np.array([lww, lwh])
                    mr = np.array([rww, rwh])

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
                    cv.circle(frame, (lsw, lsh), 6, (0,0,255), 4)
                    cv.circle(frame, (lew, leh), 6, (255,0,0), 6)
                    cv.circle(frame, (lww, lwh), 6, (0,0,255), 4)
                    cv.circle(frame, (rsw, rsh), 6, (0,0,255), 4)
                    cv.circle(frame, (rew, reh), 6, (255,0,0), 6)
                    cv.circle(frame, (rww, rwh), 6, (0,0,255), 4)

                    #Dibujado de lineas entre los joints
                    cv.line(frame, (lsw, lsh), (lew, leh), (255,0,0), 20)
                    cv.line(frame, (lew, leh), (lww, lwh), (255,0,0), 20)
                    cv.line(frame, (lsw, lsh), (lww, lwh), (0,0,255), 5)
                    cv.line(frame, (rsw, rsh), (rew, reh), (255,0,0), 20)
                    cv.line(frame, (rew, reh), (rww, rwh), (255,0,0), 20)
                    cv.line(frame, (rsw, rsh), (rww, rwh), (0,0,255), 5)

                    #Impresión de la imagen final
                    cv.putText(frame, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angleElbow)), (100, 80), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angle2)), (rew+30, reh), 1, 1.5, (0, 255, 0), 2)
                    cv.putText(frame, str(int(angle1)), (lew+30, leh), 1, 1.5, (0, 255, 0), 2)

                    #contar la repetición de una flexión válida
                    if angleElbow >= 150:                        
                        th.Thread(target=text_to_speech, args=('OHP',)).start()
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
                    center1 = (lew, leh)  # Puedes ajustar el centro del arco según tus necesidades
                    center2 = (rew, reh)  # Puedes ajustar el centro del arco según tus necesidades
                    
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
                    nw = int(results.pose_landmarks.landmark[0].x * width)
                    nh = int(results.pose_landmarks.landmark[0].y * height)

                    #Declaración del joint 11(hombro derecho)
                    rsw = int(results.pose_landmarks.landmark[11].x * width)
                    rsh = int(results.pose_landmarks.landmark[11].y * height)

                    #Declaración del joint 12(hombro izquierdo)
                    lsw = int(results.pose_landmarks.landmark[12].x * width)
                    lsh = int(results.pose_landmarks.landmark[12].y * height)

                    #Declaración del joint 13(codo derecho)
                    rew = int(results.pose_landmarks.landmark[13].x * width)
                    reh = int(results.pose_landmarks.landmark[13].y * height)

                    #Declaración del joint 14(codo izquierdo)
                    lew = int(results.pose_landmarks.landmark[14].x * width)
                    leh = int(results.pose_landmarks.landmark[14].y * height)

                    #Declaración del joint 15(muñeca derecha)
                    rww = int(results.pose_landmarks.landmark[15].x * width)
                    rwh = int(results.pose_landmarks.landmark[15].y * height)

                    #Declaración del joint 16(muñeca izquierda)
                    lww = int(results.pose_landmarks.landmark[16].x * width)
                    lwh = int(results.pose_landmarks.landmark[16].y * height)

                    #Declaración del joint 23 (cadera derecha)
                    rhw = int(results.pose_landmarks.landmark[23].x * width)
                    rhh = int(results.pose_landmarks.landmark[23].y * height)
                    
                    #Declaración del joint 24 (cadera izquierda)
                    lhw = int(results.pose_landmarks.landmark[24].x * width)
                    lhh = int(results.pose_landmarks.landmark[24].y * height)

                    #Declaración de puntos de referencia
                    hi = np.array([lsw, lsh])
                    hr = np.array([rsw, rsh])

                    ci = np.array([lew, leh])
                    cr = np.array([rew, reh])

                    mi = np.array([lww, lwh])
                    mr = np.array([rww, rwh])

                    hi = np.array([rhw, rhh]) #Cadera izquierda
                    hr = np.array([lhw, lhh]) #Cadera derecha

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
                    cv.circle(frame, (lsw, lsh), 6, (0,0,255), 4)
                    cv.circle(frame, (lew, leh), 6, (255,0,0), 6)
                    cv.circle(frame, (lww, lwh), 6, (0,0,255), 4)
                    cv.circle(frame, (rsw, rsh), 6, (0,0,255), 4)
                    cv.circle(frame, (rew, reh), 6, (255,0,0), 6)
                    cv.circle(frame, (rww, rwh), 6, (0,0,255), 4)

                    #Dibujado de lineas entre los joints
                    cv.line(frame, (lsw, lsh), (lew, leh), (255,0,0), 20)
                    cv.line(frame, (lew, leh), (lww, lwh), (255,0,0), 20)
                    cv.line(frame, (lsw, lsh), (lww, lwh), (0,0,255), 5)
                    cv.line(frame, (rsw, rsh), (rew, reh), (255,0,0), 20)
                    cv.line(frame, (rew, reh), (rww, rwh), (255,0,0), 20)
                    cv.line(frame, (rsw, rsh), (rww, rwh), (0,0,255), 5)

                    #Impresión de la imagen final
                    cv.putText(frame, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angleElbow)), (100, 80), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angle2)), (rew+30, reh), 1, 1.5, (0, 255, 0), 2)
                    cv.putText(frame, str(int(angle1)), (lew+30, leh), 1, 1.5, (0, 255, 0), 2)

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
                    center1 = (lew, leh)  # Puedes ajustar el centro del arco según tus necesidades
                    center2 = (rew, reh)  # Puedes ajustar el centro del arco según tus necesidades
                    
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



