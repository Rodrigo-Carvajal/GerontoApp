import mediapipe as mp
import cv2 as cv
import numpy as np
from math import acos, degrees, hypot

def dibujar_articulaciones(cap):
    #Llamado de objetos mediapipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    with mp_pose.Pose(static_image_mode = False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            height, width, _ = frame.shape
            if ret:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)                
                results = pose.process(frame)

            if results.pose_landmarks:
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                
                if right_shoulder and right_elbow and right_wrist:
                    # Calcula los vectores entre las articulaciones
                    shoulder_to_elbow = (right_shoulder.x - right_elbow.x, right_shoulder.y - right_elbow.y)
                    wrist_to_elbow = (right_wrist.x - right_elbow.x, right_wrist.y - right_elbow.y)

                    # Calcula el ángulo entre los dos vectores usando el producto escalar
                    dot_product = shoulder_to_elbow[0] * wrist_to_elbow[0] + shoulder_to_elbow[1] * wrist_to_elbow[1]
                    shoulder_to_elbow_length = hypot(*shoulder_to_elbow)
                    wrist_to_elbow_length = hypot(*wrist_to_elbow)
                    angle_radians = acos(dot_product / (shoulder_to_elbow_length * wrist_to_elbow_length))

                    # Convierte el ángulo a grados
                    angle_degrees = degrees(angle_radians)

                    # Muestra el ángulo en la imagen
                    cv.putText(frame, f"Ángulo del codo: {angle_degrees:.2f} grados", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Dibuja las articulaciones
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

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
                    cv.rectangle(frame, (40,12), (55,50), (255,255,255), -1)
                    cv.putText(frame, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angleElbow)), (100, 80), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angle2)), (rew+30, reh), 1, 1.5, (0, 255, 0), 2)
                    cv.putText(frame, str(int(angle1)), (lew+30, leh), 1, 1.5, (0, 255, 0), 2)

                    #contar la repetición de una flexión válida
                    if angleElbow >= 150:
                        angleColor1 = (0,0,255)
                        angleColor2 = (0,0,255)
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
                    
                    # Dibujar el arco del área del ángulo
                    center1 = (lew, leh)  # Puedes ajustar el centro del arco según tus necesidades
                    center2 = (rew, reh)  # Puedes ajustar el centro del arco según tus necesidades                    
                    cv.circle(frame, center1, 35, (255,255,255), 1)  # -1 rellena el arco
                    cv.circle(frame, center2, 35, (255,255,255), 0)  # -1 rellena el arco                    
                    if angle1 >= 160:                        
                        cv.circle(frame, center1, 10, (0,255,0), -1)  # -1 rellena el arco
                    if angle2 >= 160:
                        cv.circle(frame, center2, 10, (0,255,0), -1)  # -1 rellena el arco                    
                    if 160>=angle1 and angle1>= 135:
                        cv.circle(frame, center1, 18, (0,180,255), -1)  # -1 rellena el arco                    
                    if 160>=angle2 and angle2>= 135:
                        cv.circle(frame, center2, 18, (0,180,255), -1)  # -1 rellena el arco
                    if 135>=angle1 and angle1>= 115:
                        cv.circle(frame, center1, 23, (0,85,230), -1)  # -1 rellena el arco
                    if 135>=angle2 and angle2>= 115:
                        cv.circle(frame, center2, 23, (0,85,230), -1)  # -1 rellena el arco
                    if 115>=angle1 and angle1>= 100:
                        cv.circle(frame, center1, 26, (0,80,255), -1)  # -1 rellena el arco                    
                    if 115>=angle2 and angle2>= 100:
                        cv.circle(frame, center2, 26, (0,80,255), -1)  # -1 rellena el arco
                    if 100>=angle1 and angle1>= 85:
                        cv.circle(frame, center1, 30, (0,75,255), -1)  # -1 rellena el arco
                    if 100>=angle2 and angle2>= 85:
                        cv.circle(frame, center2, 30, (0,75,255), -1)  # -1 rellena el arco
                    if angle1<=85:
                        cv.circle(frame, center1, 34, (0,25,255), -1)  # -1 rellena el arco
                    if angle2<=85:
                        cv.circle(frame, center2, 34, (0,25,255), -1)  # -1 rellena el arco
                    
                (flag, encodedImage) = cv.imencode(".jpg", frame)  
                
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                    bytearray(encodedImage) + b'\r\n')


