from flask import Flask, render_template, Response
import cv2 as cv
import mediapipe as mp
from math import acos, degrees
import numpy as np


app = Flask(__name__)

def squat():
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

def pushup():
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
                        angleColor1 = (0,255,0)
                        angleColor2 = (0,255,0)
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
                        angleColor1 = (0,0,255)
                        angleColor2 = (0,0,255)
                        count += 1
                        up = False
                        down = False        

                    #Impresión de la imagen final
                    cv.rectangle(frame, (40,12), (90,50), (150,150,150), -1)
                    cv.putText(frame, str(count), (50, 50), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angleElbow)), (100, 80), 1, 3.5, (0, 0, 255), 3)
                    cv.putText(frame, str(int(angle2)), (rew+30, reh), 1, 1.5, (0, 255, 0), 2)
                    cv.putText(frame, str(int(angle1)), (lew+30, leh), 1, 1.5, (0, 255, 0), 2)

                    # Dibujar el arco del área del ángulo
                    center1 = (lew, leh)  # Puedes ajustar el centro del arco según tus necesidades
                    radius = 50  # Puedes ajustar el radio del arco según tus necesidades
                    cv.ellipse(frame, center1, (radius, radius), 0, 360 - angle1 / 2, 360 + angle1 / 2, angleColor1, -1)  # -1 rellena el arco
                    
                    # Dibujar el arco del área del ángulo
                    center2 = (rew, reh)  # Puedes ajustar el centro del arco según tus necesidades
                    radius = 50  # Puedes ajustar el radio del arco según tus necesidades
                    cv.ellipse(frame, center2, (radius, radius), 0, 180 - angle2 / 2, 180 + angle2 / 2, angleColor2, -1)  # -1 rellena el arco
                    
                (flag, encodedImage) = cv.imencode(".jpg", frame)  
                
                if not flag:
                    continue
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                    bytearray(encodedImage) + b'\r\n')
            

@app.route("/")
def main():
    #crunches2("GerontoApp/videos/abs3.mp4")    
    return render_template('main.html')

@app.route("/video_feed")
def video_feed():
    ejercicios = ['squat', 'pushup']
    ejercicio = ejercicios[0]
    if ejercicio == 'squat':
        return Response(squat(), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif ejercicio == 'pushup':
        return Response(pushup(), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')    


if __name__ == "__main__":
    cap = cv.VideoCapture(0, cv.CAP_MSMF)    
    app.run(debug=True, host="0.0.0.0", port=5001)
    cap.release()