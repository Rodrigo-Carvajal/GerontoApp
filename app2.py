from flask import Flask, render_template, Response
from flask_sqlalchemy import SQLAlchemy
import cv2 as cv
import mediapipe as mp
from math import acos, degrees
import numpy as np
from exc import squat, pushup

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///adult.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __init__(self, username, email):
        self.username = username
        self.email = email

# Define tu modelo de datos para la tabla Pacientes
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('patients', lazy=True))

    def __init__(self, name, user_id):
        self.name = name
        self.user_id = user_id

# Define tu modelo de datos para la tabla Ejercicios
class Exercise(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    description = db.Column(db.String(200))
    video_url = db.Column(db.String(200))  # Puedes almacenar URLs de video aquí

    def __init__(self, name, description, video_url):
        self.name = name
        self.description = description
        self.video_url = video_url


@app.route("/")
def main():
    pacientes = Patient.query.all()
    return render_template('main.html', pacientes=pacientes)



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
        db.create_all()
        app.run(debug=True, host="0.0.0.0", port=5000)