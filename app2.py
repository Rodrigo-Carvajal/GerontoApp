from flask import Flask, render_template, Response
import cv2 as cv
import mediapipe as mp
from math import acos, degrees
import numpy as np
from exc import squat, pushup

app = Flask(__name__)

@app.route("/")
def main():
    #crunches2("GerontoApp/videos/abs3.mp4")    
    return render_template('main.html')

@app.route("/video_feed")
def video_feed():
    ejercicios = ['squat', 'pushup']
    ejercicio = ejercicios[1]
    if ejercicio == 'squat':
        return Response(squat(cap), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif ejercicio == 'pushup':
        return Response(pushup(cap), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')    


if __name__ == "__main__":
    cap = cv.VideoCapture(0, cv.CAP_MSMF)    
    app.run(debug=True, host="0.0.0.0", port=5001)
    cap.release()