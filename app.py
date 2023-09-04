import tempfile
import streamlit as st
import streamlit_scrollable_textbox as stx
import mediapipe as mp
import numpy as np
import cv2 as cv
from math import degrees, acos
from functions import resize

# Función que evalua los abdominales en vivo
def crunches():
    #Llamado de objetos mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv.VideoCapture(0)
    frame_placeholder = st.empty()
    with mp_pose.Pose(static_image_mode=False) as pose:
        #Variables usadas para el control de la evaluación y conteo de sentadillas
        up = False
        down = False
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Video/Cámara no detectado")
            height, width, _ = frame.shape
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame_placeholder.image(frame,channels="RGB")
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
            
                #Declaración de puntos de referencia
                hi = np.array([lsw, lsh])
                hr = np.array([rsw, rsh])

                ci = np.array([lhw, lhh])
                cr = np.array([rhw, rhh])

                ri = np.array([lkw, lkh])
                rr = np.array([rkw, rkh])
                
                #Declaración de lineas en base a los puntos de referencia
                l1 = np.linalg.norm(ci-ri)
                l2 = np.linalg.norm(hi-ri)
                l3 = np.linalg.norm(hi-ci)
                l4 = np.linalg.norm(cr-rr)
                l5 = np.linalg.norm(hr-rr)
                l6 = np.linalg.norm(hr-cr)

                #Cálculo de ángulos en base al tciángulo formado por los joints
                angle1 = degrees(acos((l1**2 + l3**2 - l2**2) / (2*l1*l3)))
                angle2 = degrees(acos((l4**2 + l6**2 - l5**2) / (2*l4*l6)))

                angleHip = (angle1 + angle2)/2

                #Dibujado de joints importantes
                #Imágen auxiliar solo con el esqueleto
                aux_image= np.zeros(frame.shape, np.uint8)
                
                #Dibujado de joints
                cv.circle(frame, (lhw, lhh), 6, (0,0,255), 4)
                cv.circle(frame, (lkw, lkh), 6, (255,0,0), 6)
                cv.circle(frame, (rhw, rhh), 6, (0,0,255), 4)
                cv.circle(frame, (rkw, rkh), 6, (255,0,0), 6)
                cv.circle(frame, (rsw, rsh), 6, (0,0,255), 4)
                cv.circle(frame, (lsw, lsh), 6, (0,0,255), 4)

                #Dibujado de lineas entre los joints
                cv.line(aux_image, (lhw, lhh), (lkw, lkh), (255,0,0), 20)
                cv.line(aux_image, (rhw, rhh), (rkw, rkh), (255,0,0), 20)
                cv.line(aux_image, (lsw, lsh), (lhw, lhh), (255,0,0), 20)
                cv.line(aux_image, (rsw, rsh), (rhw, rhh), (255,0,0), 20)
                cv.line(aux_image, (rkw, rkh), (rsw, rsh), (0,0,255), 5)
                cv.line(aux_image, (lkw, lkh), (lsw, lsh), (0,0,255), 5)

                output = cv.addWeighted(frame, 1, aux_image, 0.8, 0)
            
                #Impresión de la imagen final
                cv.rectangle(output, (40,12), (90,50), (150,150,150), -1)
                cv.putText(output, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angleHip)), (100, 80), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angle2)), (rhw-20, rhh+50), 1, 1.5, (0, 255, 0), 2)
                cv.putText(output, str(int(angle1)), (lhw-20, lhh+50), 1, 1.5, (0, 255, 0), 2)

                #Contar la repetición de una sentadilla válida
                if angleHip >= 115: #Se encuentra acostado
                    # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                    arrow_length = 100
                    arrow_thickness = 3
                    arrow_color = (0, 255, 0)  # Verde
                    x_start = frame.shape[1] - arrow_length
                    y_start = arrow_length * 2
                    x_end = frame.shape[1] - arrow_length
                    y_end = arrow_length
                    cv.arrowedLine(output, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)
                    down = True
                if down == True and up == False and angleHip <= 110:
                    arrow_length = 100
                    arrow_thickness = 3
                    arrow_color = (0, 0, 255)  # Verde
                    cv.arrowedLine(output, (width - arrow_length, arrow_length), (width - arrow_length, arrow_length * 2),arrow_color, thickness=arrow_thickness, tipLength=0.3)                    
                    st.write("Fase concéntrica completada")
                    up = True
                    # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                    arrow_length = 100
                    arrow_thickness = 3
                    arrow_color = (0, 0, 255)  # Verde
                    cv.arrowedLine(output, (width - arrow_length, arrow_length), (width - arrow_length, arrow_length * 2),arrow_color, thickness=arrow_thickness, tipLength=0.3)
                if up == True and down == True  and angleHip >=115:
                    st.write("Fase excéntrica completada")
                    count += 1
                    st.write("Repeticiones válidas:", count)
                    up = False
                    down = False
                
                cv.imshow("Crunches video", resize(output, width=500))
            if cv.waitKey(1) & 0XFF == ord("q"):
                break
        cap.release()

# Función que evalua los abdominales en video
def crunches2(video):
    #Llamado de objetos mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    #Video a analizar
    cap = cv.VideoCapture(video)

    #Variables usadas para el control de la evaluación y conteo de sentadillas
    up = False
    down = False
    count = 0

    with mp_pose.Pose(static_image_mode=False) as pose:
        #Ciclo que se mantiene vivo la duración del video ingresado
        while True:
            ret, frame = cap.read()
            
            if ret == False:
                break
            height, width, _ = frame.shape
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

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
            
                #Declaración de puntos de referencia
                hi = np.array([lsw, lsh])
                hr = np.array([rsw, rsh])

                ci = np.array([lhw, lhh])
                cr = np.array([rhw, rhh])

                ri = np.array([lkw, lkh])
                rr = np.array([rkw, rkh])
                
                #Declaración de lineas en base a los puntos de referencia
                l1 = np.linalg.norm(ci-ri)
                l2 = np.linalg.norm(hi-ri)
                l3 = np.linalg.norm(hi-ci)
                l4 = np.linalg.norm(cr-rr)
                l5 = np.linalg.norm(hr-rr)
                l6 = np.linalg.norm(hr-cr)

                #Cálculo de ángulos en base al tciángulo formado por los joints
                angle1 = degrees(acos((l1**2 + l3**2 - l2**2) / (2*l1*l3)))
                angle2 = degrees(acos((l4**2 + l6**2 - l5**2) / (2*l4*l6)))

                angleHip = (angle1 + angle2)/2

                #Dibujado de joints importantes
                #Imágen auxiliar solo con el esqueleto
                aux_image= np.zeros(frame.shape, np.uint8)
                
                #Dibujado de joints
                cv.circle(frame, (lhw, lhh), 6, (0,0,255), 4)
                cv.circle(frame, (lkw, lkh), 6, (255,0,0), 6)
                cv.circle(frame, (rhw, rhh), 6, (0,0,255), 4)
                cv.circle(frame, (rkw, rkh), 6, (255,0,0), 6)
                cv.circle(frame, (rsw, rsh), 6, (0,0,255), 4)
                cv.circle(frame, (lsw, lsh), 6, (0,0,255), 4)

                #Dibujado de lineas entre los joints
                cv.line(aux_image, (lhw, lhh), (lkw, lkh), (255,0,0), 20)
                cv.line(aux_image, (rhw, rhh), (rkw, rkh), (255,0,0), 20)
                cv.line(aux_image, (lsw, lsh), (lhw, lhh), (255,0,0), 20)
                cv.line(aux_image, (rsw, rsh), (rhw, rhh), (255,0,0), 20)
                cv.line(aux_image, (rkw, rkh), (rsw, rsh), (0,0,255), 5)
                cv.line(aux_image, (lkw, lkh), (lsw, lsh), (0,0,255), 5)

                output = cv.addWeighted(frame, 1, aux_image, 0.8, 0)
            
                #Impresión de la imagen final
                cv.rectangle(output, (40,12), (90,50), (150,150,150), -1)
                cv.putText(output, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angleHip)), (100, 80), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angle2)), (rhw-20, rhh+50), 1, 1.5, (0, 255, 0), 2)
                cv.putText(output, str(int(angle1)), (lhw-20, lhh+50), 1, 1.5, (0, 255, 0), 2)

                #Contar la repetición de una sentadilla válida
                if angleHip >= 115: #Se encuentra acostado
                    # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                    arrow_length = 100
                    arrow_thickness = 3
                    arrow_color = (0, 255, 0)  # Verde
                    x_start = frame.shape[1] - arrow_length
                    y_start = arrow_length * 2
                    x_end = frame.shape[1] - arrow_length
                    y_end = arrow_length
                    cv.arrowedLine(output, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)
                    down = True
                if down == True and up == False and angleHip <= 110:
                    arrow_length = 100
                    arrow_thickness = 3
                    arrow_color = (0, 0, 255)  # Verde
                    cv.arrowedLine(output, (width - arrow_length, arrow_length), (width - arrow_length, arrow_length * 2),arrow_color, thickness=arrow_thickness, tipLength=0.3)                    
                    st.write("Fase concéntrica completada")
                    up = True
                    # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                    arrow_length = 100
                    arrow_thickness = 3
                    arrow_color = (0, 0, 255)  # Verde
                    cv.arrowedLine(output, (width - arrow_length, arrow_length), (width - arrow_length, arrow_length * 2),arrow_color, thickness=arrow_thickness, tipLength=0.3)
                if up == True and down == True  and angleHip >=115:
                    st.write("Fase excéntrica completada")
                    count += 1
                    st.write("Repeticiones válidas:", count)
                    up = False
                    down = False
                
                cv.imshow("Crunches video", resize(output, width=500))
            if cv.waitKey(1) & 0XFF == ord("q"):
                break
    cap.release()

# Función que evalua los curl de bicep en vivo
def bicepCurl():
    #Llamado de objetos mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv.VideoCapture(0)
    frame_placeholder = st.empty()
    with mp_pose.Pose(static_image_mode=False) as pose:
        #Variables usadas para el control de la evaluación y conteo de sentadillas
        up = False
        down = False
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Video/Cámara no detectado")
            height, width, _ = frame.shape
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame_placeholder.image(frame,channels="RGB")
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

                #Declaración del joint 23(cadera derecha)
                rhw = int(results.pose_landmarks.landmark[23].x * width)
                rhh = int(results.pose_landmarks.landmark[23].y * height)
                
                #Declaración del joint 24(cadera izquierda)
                lhw = int(results.pose_landmarks.landmark[24].x * width)
                lhh = int(results.pose_landmarks.landmark[24].y * height)

                #Declaración de puntos de referencia
                si = np.array([lsw, lsh])
                sr = np.array([rsw, rsh])

                ci = np.array([lew, leh])
                cr = np.array([rew, reh])

                mi = np.array([lww, lwh])
                mr = np.array([rww, rwh])

                hi = np.array([lhw, lhh])
                hr = np.array([rhw, rhh])

                #Declaración de lineas en base a los puntos de referencia
                l1 = np.linalg.norm(si-ci)
                l2 = np.linalg.norm(ci-mi)
                l3 = np.linalg.norm(si-mi)
                l4 = np.linalg.norm(sr-cr)
                l5 = np.linalg.norm(cr-mr)
                l6 = np.linalg.norm(sr-mr)
                l7 = np.linalg.norm(si-hi)
                l8 = np.linalg.norm(sr-hr)
                l9 = np.linalg.norm(hi-ci)
                l10 = np.linalg.norm(hr-cr)

                #Cálculo de ángulos en base al triángulo formado por los joints
                angle1 = degrees(acos((l1**2 + l2**2 - l3**2) / (2*l1*l2)))
                angle2 = degrees(acos((l4**2 + l5**2 - l6**2) / (2*l4*l5)))

                angle3 = degrees(acos((l7**2 + l1**2 - l9**2) / (2*l7*l1)))
                angle4 = degrees(acos((l8**2 + l4**2 - l10**2) / (2*l8*l4)))
                
                angleElbow = (angle1 + angle2)/2
                angleShoulder = (angle3 + angle4)/2
                
                #Imágen auxiliar solo con el esqueleto
                aux_image= np.zeros(frame.shape, np.uint8)
                
                #Dibujado de joints
                cv.circle(frame, (lsw, lsh), 6, (0,0,255), 4)
                cv.circle(frame, (lew, leh), 6, (255,0,0), 6)
                cv.circle(frame, (lww, lwh), 6, (0,0,255), 4)
                cv.circle(frame, (rsw, rsh), 6, (0,0,255), 4)
                cv.circle(frame, (rew, reh), 6, (255,0,0), 6)
                cv.circle(frame, (rww, rwh), 6, (0,0,255), 4)
                cv.circle(frame, (lhw, lhh), 6, (0,0,255), 4)
                cv.circle(frame, (rhw, rhh), 6, (0,0,255), 4)            

                #Dibujado de lineas entre los joints
                cv.line(aux_image, (lsw, lsh), (lew, leh), (255,0,0), 20)
                cv.line(aux_image, (lew, leh), (lww, lwh), (255,0,0), 20)
                cv.line(aux_image, (lsw, lsh), (lww, lwh), (0,0,255), 5)
                cv.line(aux_image, (lsw, lsh), (lhw, lhh), (255,0,0), 20)
                cv.line(aux_image, (rsw, rsh), (rew, reh), (255,0,0), 20)
                cv.line(aux_image, (rew, reh), (rww, rwh), (255,0,0), 20)
                cv.line(aux_image, (rsw, rsh), (rww, rwh), (0,0,255), 5)
                cv.line(aux_image, (rsw, rsh), (rhw, rhh), (255,0,0), 20)
                cv.line(aux_image, (rew, reh), (rhw, rhh), (0,0,255), 5)
                cv.line(aux_image, (lew, leh), (lhw, lhh), (0,0,255), 5)            

                output = cv.addWeighted(frame, 1, aux_image, 0.8, 0)

                #Impresión de la imagen final
                cv.rectangle(output, (40,12), (90,50), (150,150,150), -1)
                cv.putText(output, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angleElbow)), (100, 50), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angleShoulder)), (235, 50), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angle2)), (rew+30, reh), 1, 1.5, (0, 255, 0), 2)
                cv.putText(output, str(int(angle1)), (lew+30, leh), 1, 1.5, (0, 255, 0), 2)
                cv.putText(output, str(int(angle4)), (rsw+30, rsh), 1, 1.5, (0, 255, 0), 2)
                cv.putText(output, str(int(angle3)), (lsw+30, lsh), 1, 1.5, (0, 255, 0), 2)
                                
                #contar la repetición de una flexión válida
                if angleShoulder <= 20:
                    if angleElbow >=150:
                        # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                        arrow_length = 100
                        arrow_thickness = 3
                        arrow_color = (0, 255, 0)  # Verde
                        x_start = frame.shape[1] - arrow_length
                        y_start = arrow_length * 2
                        x_end = frame.shape[1] - arrow_length
                        y_end = arrow_length
                        cv.arrowedLine(output, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)
                        up = True
                    if up == True and down == False and angleElbow <= 60:
                        # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                        arrow_length = 100
                        arrow_thickness = 3
                        arrow_color = (0, 0, 255)  # Verde
                        cv.arrowedLine(output, (width - arrow_length, arrow_length), (width - arrow_length, arrow_length * 2),arrow_color, thickness=arrow_thickness, tipLength=0.3)
                        st.write("Fase excéntrica completada")
                        down = True
                        # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                        arrow_length = 100
                        arrow_thickness = 3
                        arrow_color = (0, 0, 255)  # Verde
                        cv.arrowedLine(output, (width - arrow_length, arrow_length), (width - arrow_length, arrow_length * 2),arrow_color, thickness=arrow_thickness, tipLength=0.3)
                    if up == True and down == True and angleElbow >= 150:
                        st.write("Fase concéntrica completada")
                        count += 1
                        st.write("Repeticiones válidas:", count)
                        up = False
                        down = False    
                
                cv.imshow("Bicep curl video", resize(output, width=500))
            if cv.waitKey(1) & 0XFF == ord("q"):
                break

        cap.release()

# Función que evalua los curl de bicep en video
def bicepCurl2(video):
    #Llamado de objetos mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    #Video a analizar
    cap = cv.VideoCapture(video)

    #Variables usadas para el control de la evaluación y conteo de sentadillas
    up = False
    down = False
    count = 0

    with mp_pose.Pose(static_image_mode=False) as pose:
        #Ciclo que se mantiene vivo la duración del video ingresado
        while True:
            ret, frame = cap.read()
            
            if ret == False:
                break
            height, width, _ = frame.shape
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

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

                #Declaración del joint 23(cadera derecha)
                rhw = int(results.pose_landmarks.landmark[23].x * width)
                rhh = int(results.pose_landmarks.landmark[23].y * height)
                
                #Declaración del joint 24(cadera izquierda)
                lhw = int(results.pose_landmarks.landmark[24].x * width)
                lhh = int(results.pose_landmarks.landmark[24].y * height)

                #Declaración de puntos de referencia
                si = np.array([lsw, lsh])
                sr = np.array([rsw, rsh])

                ci = np.array([lew, leh])
                cr = np.array([rew, reh])

                mi = np.array([lww, lwh])
                mr = np.array([rww, rwh])

                hi = np.array([lhw, lhh])
                hr = np.array([rhw, rhh])

                #Declaración de lineas en base a los puntos de referencia
                l1 = np.linalg.norm(si-ci)
                l2 = np.linalg.norm(ci-mi)
                l3 = np.linalg.norm(si-mi)
                l4 = np.linalg.norm(sr-cr)
                l5 = np.linalg.norm(cr-mr)
                l6 = np.linalg.norm(sr-mr)
                l7 = np.linalg.norm(si-hi)
                l8 = np.linalg.norm(sr-hr)
                l9 = np.linalg.norm(hi-ci)
                l10 = np.linalg.norm(hr-cr)

                #Cálculo de ángulos en base al triángulo formado por los joints
                angle1 = degrees(acos((l1**2 + l2**2 - l3**2) / (2*l1*l2)))
                angle2 = degrees(acos((l4**2 + l5**2 - l6**2) / (2*l4*l5)))

                angle3 = degrees(acos((l7**2 + l1**2 - l9**2) / (2*l7*l1)))
                angle4 = degrees(acos((l8**2 + l4**2 - l10**2) / (2*l8*l4)))
                
                angleElbow = (angle1 + angle2)/2
                angleShoulder = (angle3 + angle4)/2

                #Imágen auxiliar solo con el esqueleto
                aux_image= np.zeros(frame.shape, np.uint8)
                
                #Dibujado de joints
                cv.circle(frame, (lsw, lsh), 6, (0,0,255), 4)
                cv.circle(frame, (lew, leh), 6, (255,0,0), 6)
                cv.circle(frame, (lww, lwh), 6, (0,0,255), 4)
                cv.circle(frame, (rsw, rsh), 6, (0,0,255), 4)
                cv.circle(frame, (rew, reh), 6, (255,0,0), 6)
                cv.circle(frame, (rww, rwh), 6, (0,0,255), 4)
                cv.circle(frame, (lhw, lhh), 6, (0,0,255), 4)
                cv.circle(frame, (rhw, rhh), 6, (0,0,255), 4)            

                #Dibujado de lineas entre los joints
                cv.line(aux_image, (lsw, lsh), (lew, leh), (255,0,0), 20)
                cv.line(aux_image, (lew, leh), (lww, lwh), (255,0,0), 20)
                cv.line(aux_image, (lsw, lsh), (lww, lwh), (0,0,255), 5)
                cv.line(aux_image, (lsw, lsh), (lhw, lhh), (255,0,0), 20)
                cv.line(aux_image, (rsw, rsh), (rew, reh), (255,0,0), 20)
                cv.line(aux_image, (rew, reh), (rww, rwh), (255,0,0), 20)
                cv.line(aux_image, (rsw, rsh), (rww, rwh), (0,0,255), 5)
                cv.line(aux_image, (rsw, rsh), (rhw, rhh), (255,0,0), 20)
                cv.line(aux_image, (rew, reh), (rhw, rhh), (0,0,255), 5)
                cv.line(aux_image, (lew, leh), (lhw, lhh), (0,0,255), 5)            

                output = cv.addWeighted(frame, 1, aux_image, 0.8, 0)

                #Impresión de la imagen final
                cv.rectangle(output, (40,12), (90,50), (150,150,150), -1)
                cv.putText(output, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angleElbow)), (100, 50), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angleShoulder)), (235, 50), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angle2)), (rew+30, reh), 1, 1.5, (0, 255, 0), 2)
                cv.putText(output, str(int(angle1)), (lew+30, leh), 1, 1.5, (0, 255, 0), 2)
                cv.putText(output, str(int(angle4)), (rsw+30, rsh), 1, 1.5, (0, 255, 0), 2)
                cv.putText(output, str(int(angle3)), (lsw+30, lsh), 1, 1.5, (0, 255, 0), 2)
                                
                #contar la repetición de una flexión válida
                if angleShoulder <= 20:
                    if angleElbow >=150:
                        # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                        arrow_length = 100
                        arrow_thickness = 3
                        arrow_color = (0, 255, 0)  # Verde
                        x_start = frame.shape[1] - arrow_length
                        y_start = arrow_length * 2
                        x_end = frame.shape[1] - arrow_length
                        y_end = arrow_length
                        cv.arrowedLine(output, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)
                        up = True
                    if up == True and down == False and angleElbow <= 60:
                        # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                        arrow_length = 100
                        arrow_thickness = 3
                        arrow_color = (0, 0, 255)  # Verde
                        cv.arrowedLine(output, (width - arrow_length, arrow_length), (width - arrow_length, arrow_length * 2),arrow_color, thickness=arrow_thickness, tipLength=0.3)
                        st.write("Fase excéntrica completada")
                        down = True
                        # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                        arrow_length = 100
                        arrow_thickness = 3
                        arrow_color = (0, 0, 255)  # Verde
                        cv.arrowedLine(output, (width - arrow_length, arrow_length), (width - arrow_length, arrow_length * 2),arrow_color, thickness=arrow_thickness, tipLength=0.3)
                    if up == True and down == True and angleElbow >= 150:
                        st.write("Fase concéntrica completada")
                        count += 1
                        st.write("Repeticiones válidas:", count)
                        up = False
                        down = False    
                
                cv.imshow("Bicep curl video", resize(output, width=500))
            if cv.waitKey(1) & 0XFF == ord("q"):
                break

    cap.release()

# Función que evalua flexiones de brazos en vivo
def pushup():
    #Llamado de objetos mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv.VideoCapture(0)
    frame_placeholder = st.empty()
    with mp_pose.Pose(static_image_mode=False) as pose:
        #Variables usadas para el control de la evaluación y conteo de sentadillas
        up = False
        down = False
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Video/Cámara no detectado")
                break
            height, width, _ = frame.shape
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame_placeholder.image(frame,channels="RGB")
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
                
                #Imágen auxiliar solo con el esqueleto
                aux_image= np.zeros(frame.shape, np.uint8)

                #Dibujado de joints
                cv.circle(frame, (lsw, lsh), 6, (0,0,255), 4)
                cv.circle(frame, (lew, leh), 6, (255,0,0), 6)
                cv.circle(frame, (lww, lwh), 6, (0,0,255), 4)
                cv.circle(frame, (rsw, rsh), 6, (0,0,255), 4)
                cv.circle(frame, (rew, reh), 6, (255,0,0), 6)
                cv.circle(frame, (rww, rwh), 6, (0,0,255), 4)

                #Dibujado de lineas entre los joints
                cv.line(aux_image, (lsw, lsh), (lew, leh), (255,0,0), 20)
                cv.line(aux_image, (lew, leh), (lww, lwh), (255,0,0), 20)
                cv.line(aux_image, (lsw, lsh), (lww, lwh), (0,0,255), 5)
                cv.line(aux_image, (rsw, rsh), (rew, reh), (255,0,0), 20)
                cv.line(aux_image, (rew, reh), (rww, rwh), (255,0,0), 20)
                cv.line(aux_image, (rsw, rsh), (rww, rwh), (0,0,255), 5)                

                output = cv.addWeighted(frame, 1, aux_image, 0.8, 0)
                
                #Impresión de la imagen final
                cv.rectangle(output, (40,12), (90,50), (150,150,150), -1)
                cv.putText(output, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angleElbow)), (100, 80), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angle2)), (rew+30, reh), 1, 1.5, (0, 255, 0), 2)
                cv.putText(output, str(int(angle1)), (lew+30, leh), 1, 1.5, (0, 255, 0), 2)

                #contar la repetición de una flexión válida
                if angleElbow >= 150:
                    arrow_length = 100
                    arrow_thickness = 3
                    arrow_color = (0, 255, 0)  # Verde
                    cv.arrowedLine(output, (width - arrow_length, arrow_length), (width - arrow_length, arrow_length * 2),arrow_color, thickness=arrow_thickness, tipLength=0.3)                    
                    up = True                
                if up == True and down == False and angleElbow <= 90:                    
                    st.write("Fase concéntrica completada")
                    # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                    arrow_length = 100
                    arrow_thickness = 3
                    arrow_color = (0, 0, 255)  # Verde
                    x_start = frame.shape[1] - arrow_length
                    y_start = arrow_length * 2
                    x_end = frame.shape[1] - arrow_length
                    y_end = arrow_length
                    cv.arrowedLine(output, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)
                    down=True
                    # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                    arrow_length = 100
                    arrow_thickness = 3
                    arrow_color = (0, 0, 255)  # Verde
                    x_start = frame.shape[1] - arrow_length
                    y_start = arrow_length * 2
                    x_end = frame.shape[1] - arrow_length
                    y_end = arrow_length
                    cv.arrowedLine(output, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)
                if up == True and down == True and angleElbow>=150:
                    st.write("Fase excéntrica completada")
                    count += 1
                    st.write("Repeticiones válidas:", count)
                    up = False
                    down = False            
                
                cv.imshow("Push ups video", resize(output, width=500))
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()

# Función que evalua flexiones de brazo en video
def pushup2(video):
    #Llamado de objetos mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    #Video a analizar
    cap = cv.VideoCapture(video)

    #Variables usadas para el control de la evaluación y conteo de sentadillas
    up = False
    down = False
    count = 0

    with mp_pose.Pose(static_image_mode=False) as pose:
        #Ciclo que se mantiene vivo la duración del video ingresado
        while True:
            ret, frame = cap.read()
            
            if ret == False:
                break
            height, width, _ = frame.shape
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

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
                
                #Imágen auxiliar solo con el esqueleto
                aux_image= np.zeros(frame.shape, np.uint8)

                #Dibujado de joints
                cv.circle(frame, (lsw, lsh), 6, (0,0,255), 4)
                cv.circle(frame, (lew, leh), 6, (255,0,0), 6)
                cv.circle(frame, (lww, lwh), 6, (0,0,255), 4)
                cv.circle(frame, (rsw, rsh), 6, (0,0,255), 4)
                cv.circle(frame, (rew, reh), 6, (255,0,0), 6)
                cv.circle(frame, (rww, rwh), 6, (0,0,255), 4)

                #Dibujado de lineas entre los joints
                cv.line(aux_image, (lsw, lsh), (lew, leh), (255,0,0), 20)
                cv.line(aux_image, (lew, leh), (lww, lwh), (255,0,0), 20)
                cv.line(aux_image, (lsw, lsh), (lww, lwh), (0,0,255), 5)
                cv.line(aux_image, (rsw, rsh), (rew, reh), (255,0,0), 20)
                cv.line(aux_image, (rew, reh), (rww, rwh), (255,0,0), 20)
                cv.line(aux_image, (rsw, rsh), (rww, rwh), (0,0,255), 5)

                output = cv.addWeighted(frame, 1, aux_image, 0.8, 0)

                #contar la repetición de una flexión válida
                if angleElbow >= 150:
                    arrow_length = 100
                    arrow_thickness = 3
                    arrow_color = (0, 255, 0)  # Verde
                    cv.arrowedLine(output, (width - arrow_length, arrow_length), (width - arrow_length, arrow_length * 2),arrow_color, thickness=arrow_thickness, tipLength=0.3)                    
                    up = True                
                if up == True and down == False and angleElbow <= 90:                    
                    st.write("Fase concéntrica completada")
                    # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                    arrow_length = 100
                    arrow_thickness = 3
                    arrow_color = (0, 0, 255)  # Verde
                    x_start = frame.shape[1] - arrow_length
                    y_start = arrow_length * 2
                    x_end = frame.shape[1] - arrow_length
                    y_end = arrow_length
                    cv.arrowedLine(output, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)
                    down=True
                    # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                    arrow_length = 100
                    arrow_thickness = 3
                    arrow_color = (0, 0, 255)  # Verde
                    x_start = frame.shape[1] - arrow_length
                    y_start = arrow_length * 2
                    x_end = frame.shape[1] - arrow_length
                    y_end = arrow_length
                    cv.arrowedLine(output, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)
                if up == True and down == True and angleElbow>=150:
                    st.write("Fase excéntrica completada")
                    count += 1
                    st.write("Repeticiones válidas:", count)
                    up = False
                    down = False                               
            
                #Impresión de la imagen final
                cv.rectangle(output, (40,12), (90,50), (150,150,150), -1)
                cv.putText(output, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angleElbow)), (100, 80), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angle2)), (rew+30, reh), 1, 1.5, (0, 255, 0), 2)
                cv.putText(output, str(int(angle1)), (lew+30, leh), 1, 1.5, (0, 255, 0), 2)
                
                cv.imshow("Push ups video", resize(output, width=500))
            if cv.waitKey(1) & 0XFF == ord("q"):
                break
        html_string = "<h3>this is an html string</h3>"

        st.markdown(html_string, unsafe_allow_html=True)
    cap.release()

# Función que evalua sentadillas en vivo
def squat():
    #Llamado de objetos mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    cap = cv.VideoCapture(0)
    frame_placeholder = st.empty()
    with mp_pose.Pose(static_image_mode=False) as pose:
        #Variables usadas para el control de la evaluación y conteo de sentadillas
        up = False
        down = False
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Video/Cámara no detectado")
                break
            height, width, _ = frame.shape
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame_placeholder.image(frame,channels="RGB")
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

                
                #Dibujado de joints importantes
                #Imágen auxiliar solo con el esqueleto
                aux_image= np.zeros(frame.shape, np.uint8)
                
                #Dibujado de joints
                cv.circle(frame, (lhw, lhh), 6, (0,0,255), 4)
                cv.circle(frame, (lkw, lkh), 6, (255,0,0), 6)
                cv.circle(frame, (rhw, rhh), 6, (0,0,255), 4)
                cv.circle(frame, (rkw, rkh), 6, (255,0,0), 6)
                cv.circle(frame, (rsw, rsh), 6, (0,0,255), 4)
                cv.circle(frame, (lsw, lsh), 6, (0,0,255), 4)

                #Dibujado de lineas entre los joints
                cv.line(aux_image, (lhw, lhh), (lkw, lkh), (255,0,0), 20)
                cv.line(aux_image, (rhw, rhh), (rkw, rkh), (255,0,0), 20)
                cv.line(aux_image, (lsw, lsh), (lhw, lhh), (255,0,0), 20)
                cv.line(aux_image, (rsw, rsh), (rhw, rhh), (255,0,0), 20)
                cv.line(aux_image, (rkw, rkh), (rsw, rsh), (0,0,255), 5)
                cv.line(aux_image, (lkw, lkh), (lsw, lsh), (0,0,255), 5)

                output = cv.addWeighted(frame, 1, aux_image, 0.8, 0)
            
                #Impresión de la imagen final
                cv.rectangle(output, (40,12), (90,50), (150,150,150), -1)
                cv.putText(output, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angleHip)), (100, 80), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angle2)), (rhw-20, rhh+50), 1, 1.5, (0, 255, 0), 2)
                cv.putText(output, str(int(angle1)), (lhw-20, lhh+50), 1, 1.5, (0, 255, 0), 2)

                #Contar la repetición de una sentadilla válida
                if angleHip >= 115: #Se encuentra acostado
                    # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                    arrow_length = 100
                    arrow_thickness = 3
                    arrow_color = (0, 255, 0)  # Verde
                    cv.arrowedLine(output, (width - arrow_length, arrow_length), (width - arrow_length, arrow_length * 2),arrow_color, thickness=arrow_thickness, tipLength=0.3)
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
                    cv.arrowedLine(output, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)
                    st.write("Fase concéntrica completada")
                    up = True
                    # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                    arrow_length = 100
                    arrow_thickness = 3
                    arrow_color = (0, 0, 255)  # Verde
                    x_start = frame.shape[1] - arrow_length
                    y_start = arrow_length * 2
                    x_end = frame.shape[1] - arrow_length
                    y_end = arrow_length
                    cv.arrowedLine(output, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)                                        
                if up == True and down == True  and angleHip >=115:
                    st.write("Fase excéntrica completada")
                    count += 1
                    st.write("Repeticiones válidas:", count)
                    up = False
                    down = False

                cv.imshow("Squat", resize(output, width=500))
                
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
    
# Función que evalua sentadillas en video
def squat2(video):
    #Llamado de objetos mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    #Video a analizar
    cap = cv.VideoCapture(video)

    #Variables usadas para el control de la evaluación y conteo de sentadillas
    up = False
    down = False
    count = 0

    with mp_pose.Pose(static_image_mode=False) as pose:
        #Ciclo que se mantiene vivo la duración del video ingresado
        while True:
            ret, frame = cap.read()
            
            if ret == False:
                break
            height, width, _ = frame.shape
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

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
            
                #Declaración de puntos de referencia
                hi = np.array([lsw, lsh])
                hr = np.array([rsw, rsh])

                ci = np.array([lhw, lhh])
                cr = np.array([rhw, rhh])

                ri = np.array([lkw, lkh])
                rr = np.array([rkw, rkh])
                
                #Declaración de lineas en base a los puntos de referencia
                l1 = np.linalg.norm(ci-ri)
                l2 = np.linalg.norm(hi-ri)
                l3 = np.linalg.norm(hi-ci)
                l4 = np.linalg.norm(cr-rr)
                l5 = np.linalg.norm(hr-rr)
                l6 = np.linalg.norm(hr-cr)

                #Cálculo de ángulos en base al tciángulo formado por los joints
                angle1 = degrees(acos((l1**2 + l3**2 - l2**2) / (2*l1*l3)))
                angle2 = degrees(acos((l4**2 + l6**2 - l5**2) / (2*l4*l6)))

                angleHip = (angle1 + angle2)/2

                #Dibujado de joints importantes
                #Imágen auxiliar solo con el esqueleto
                aux_image= np.zeros(frame.shape, np.uint8)
                
                #Dibujado de joints
                cv.circle(frame, (lhw, lhh), 6, (0,0,255), 4)
                cv.circle(frame, (lkw, lkh), 6, (255,0,0), 6)
                cv.circle(frame, (rhw, rhh), 6, (0,0,255), 4)
                cv.circle(frame, (rkw, rkh), 6, (255,0,0), 6)
                cv.circle(frame, (rsw, rsh), 6, (0,0,255), 4)
                cv.circle(frame, (lsw, lsh), 6, (0,0,255), 4)

                #Dibujado de lineas entre los joints
                cv.line(aux_image, (lhw, lhh), (lkw, lkh), (255,0,0), 20)
                cv.line(aux_image, (rhw, rhh), (rkw, rkh), (255,0,0), 20)
                cv.line(aux_image, (lsw, lsh), (lhw, lhh), (255,0,0), 20)
                cv.line(aux_image, (rsw, rsh), (rhw, rhh), (255,0,0), 20)
                cv.line(aux_image, (rkw, rkh), (rsw, rsh), (0,0,255), 5)
                cv.line(aux_image, (lkw, lkh), (lsw, lsh), (0,0,255), 5)

                output = cv.addWeighted(frame, 1, aux_image, 0.8, 0)
            
                #Impresión de la imagen final
                cv.rectangle(output, (40,12), (90,50), (150,150,150), -1)
                cv.putText(output, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angleHip)), (100, 80), 1, 3.5, (0, 0, 255), 3)
                cv.putText(output, str(int(angle2)), (rhw-20, rhh+50), 1, 1.5, (0, 255, 0), 2)
                cv.putText(output, str(int(angle1)), (lhw-20, lhh+50), 1, 1.5, (0, 255, 0), 2)

                #Contar la repetición de una sentadilla válida
                if angleHip >= 115: #Se encuentra acostado
                    # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                    arrow_length = 100
                    arrow_thickness = 3
                    arrow_color = (0, 255, 0)  # Verde
                    cv.arrowedLine(output, (width - arrow_length, arrow_length), (width - arrow_length, arrow_length * 2),arrow_color, thickness=arrow_thickness, tipLength=0.3)
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
                    cv.arrowedLine(output, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)
                    st.write("Fase concéntrica completada")
                    up = True
                    # Dibujar la flecha apuntando hacia arriba en la esquina derecha del fotograma
                    arrow_length = 100
                    arrow_thickness = 3
                    arrow_color = (0, 0, 255)  # Verde
                    x_start = frame.shape[1] - arrow_length
                    y_start = arrow_length * 2
                    x_end = frame.shape[1] - arrow_length
                    y_end = arrow_length
                    cv.arrowedLine(output, (x_start, y_start), (x_end, y_end), arrow_color, thickness=arrow_thickness, tipLength=0.3)                                        
                if up == True and down == True  and angleHip >=115:
                    st.write("Fase excéntrica completada")
                    count += 1
                    st.write("Repeticiones válidas:", count)
                    up = False
                    down = False

                
                cv.imshow("Squat Video", resize(output, width=500))
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
 
def main():
    # Default sidebar
    ejercicios = ["Elige un ejercicio", "Sentadillas (Squats)", "Flexiones (Push ups)", "Abdominales (Crunches)", "Curl de bicep (Bicep curl) Curl"]

    st.sidebar.header("🏋️‍♂️🏋️‍♀️PersonApp Trainer🏋️‍♀️🏋️‍♂️")
    st.sidebar.subheader("Esta aplicación te ayuda a contabilizar las repeticiones y corregir la forma en ciertos ejercicios.")
    st.sidebar.caption("Asegurate de leer las indicaciones antes de usar la aplicación")
    live = st.sidebar.checkbox("¿Desea analizar a través de su cámara?")
    st.sidebar.divider()
    ejercicio = st.sidebar.selectbox("¿Qué ejercicio desea evaluar?", ejercicios)
    with st.sidebar.form('Upload',clear_on_submit=True):
        if not live:
            archivo = st.file_uploader("Suba su video a analizar (Solo 1 video).", type=['mp4', 'mov', 'avi'])
            submit = st.form_submit_button("Empezar análisis del video")
        
    if live:
        st.sidebar.divider()
        start = st.sidebar.checkbox("Comenzar el análisis")
        if ejercicio == ejercicios[0]:
            st.header("🏋️‍♂️🏋️‍♀️PersonApp Trainer🏋️‍♀️🏋️‍♂️")
            st.subheader("Esta aplicación te ayuda a contabilizar las repeticiones y corregir la forma en ciertos ejercicios.")
            st.caption("Para sacar el máximo provecho a esta aplicación se recomiendan las siguientes reglas:")        
            st.text("Reglas generales:")
            rules = """1) Al seleccionar algún ejercicio el análisis comenzará de inmediato, por lo que se recomienda que te encuentres en la posición ideal para el ejercicio a evaluar.
            2) Las indicaciones generales para cada ejercicio se indican más abajo.
            3) Para todos los ejercicios se recomienda que la persona se encuentre centrada en la cámara.
            4) Cada ejercicio consta de una fase excéntrica y concéntrica. Si vez que una repetición no se contó, es porque no cumpliste los requisitos de alguna de estas dos fases.
            5) Se sugiere evitar el uso de ropa holgada, ya que a veces genera problemas en la detección de articulaciones.
            """
            stx.scrollableTextbox(rules, height=250)

        if ejercicio == ejercicios[1] and not start:
            st.text("Reglas para sentadillas:")
            squatRules = """1) La persona debe ser visible desde un plano lateral o lateral/frontal.
            2) La posición incial debe ser de pie.
            """
            stx.scrollableTextbox(squatRules)
            st.text("Un referencia de como ubicarte frente a la cámara es la siguiente:")
            st.video("PersonApp/videos/squat2.mp4")
            
        if ejercicio == ejercicios[1] and start:
            squat()
            
        if ejercicio == ejercicios[2] and not start:
            st.text("Reglas para flexiones de brazos:")
            pushupRules = """1) La persona debe ser visible desde un plano frontal o lateral.
            2) La posición incial debe ser apoyado con las manos en el piso y los brazos extendidos.
            """
            stx.scrollableTextbox(pushupRules)
            st.text("Un referencia de como ubicarte frente a la cámara es la siguiente:")
            st.video("PersonApp/videos/push1.mp4")

        if ejercicio == ejercicios[2] and start:
            pushup()

        if ejercicio == ejercicios[3] and not start:
            st.text("Reglas para abdominales:")
            crunchesRules = """1) La persona debe ser visible desde un plano lateral.
            2) La persona se debe encontrar acostada de cúbito supino con las rodillas flectadas formando un ángulo de cadera mayor a 115 grados.
            """
            stx.scrollableTextbox(crunchesRules)
            st.text("Un referencia de como ubicarte frente a la cámara es la siguiente:")
            st.video("PersonApp/videos/abs3.mp4")

        if ejercicio == ejercicios[3] and start:
            crunches()

        if ejercicio == ejercicios[4] and not start:
            st.text("Reglas para curl de biceps:")
            bicepRules = """1) La persona debe ser visible desde un plano lateral o lateral/frontal.
            2) La posición incial debe ser de pie con los brazos completamente extendidos hacia abajo.
            3) Los codos se deben mantener pegados al cuerpo para evitar repeticiones inválidas.
            """
            stx.scrollableTextbox(bicepRules)
            st.text("Un referencia de como ubicarte frente a la cámara es la siguiente:")
            st.video("PersonApp/videos/bicep1.mp4")

        if ejercicio == ejercicios[4] and start:
            bicepCurl()

    else:
        if ejercicio == ejercicios[0]:            
                st.header("🏋️‍♂️🏋️‍♀️PersonApp Trainer🏋️‍♀️🏋️‍♂️")
                st.subheader("Esta aplicación te ayuda a contabilizar las repeticiones y corregir la forma en ciertos ejercicios.")
                st.divider()    

        if archivo and submit:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(archivo.read())
            cap = cv.VideoCapture(archivo.name)
            st.sidebar.text('Video original de entrada')
            st.sidebar.video(temp_file.name)    
            
            if ejercicio == ejercicios[1]:
                st.video(temp_file.name)
                st.divider()                
                squat2(temp_file.name)
                
            elif ejercicio == ejercicios[2]:
                st.video(temp_file.name)
                st.divider()
                pushup2(temp_file.name)

            elif ejercicio == ejercicios[3]:
                st.video(temp_file.name)
                st.divider()
                crunches2(temp_file.name)

            elif ejercicio == ejercicios[4]:
                st.video(temp_file.name)
                st.divider()
                bicepCurl2(temp_file.name)

if __name__ == '__main__':
    main()