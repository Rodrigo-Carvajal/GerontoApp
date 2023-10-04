from flask import Flask, render_template, Response
from flask_sqlalchemy import SQLAlchemy
import cv2 as cv
import mediapipe as mp
from math import acos, degrees
import numpy as np
from exc import squat, pushup

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('cam.html')

@app.route("/video_feed")
def video_feed():
    cap = cv.VideoCapture(0, cv.CAP_MSMF)
    ejercicios = ['squat', 'pushup']
    ejercicio = ejercicios[1]
    if ejercicio == 'squat':
        return Response(squat(cap), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif ejercicio == 'pushup':
        return Response(pushup(cap), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    cap.release()


with app.app_context():
    if __name__ == "__main__":        
        app.run(debug=True, host="0.0.0.0", port=5000)