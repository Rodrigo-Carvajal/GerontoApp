from app import app, supabase, login_manager, csrf
from flask import render_template, Response, redirect, url_for, request, flash, Blueprint, session, jsonify
from flask_login import login_user, login_required, logout_user, current_user
import cv2 as cv
from app.functions import squat, pushup

from app.models import Usuario, Paciente, Ejercicio, Limitacion, Registro, Sesion

# Instanciación del blueprint
adultTrain = Blueprint('app', __name__)

"""
@app.route("/ruta", methods=['GET', 'POST'])
@login_required
def ruta():
    return render_template("views/.html") redirect(url_for(''))
"""

@app.route("/")
def main():
    return redirect(url_for('login'))

@app.route("/login", methods=['GET','POST'])
def login():
    if request.method == 'POST':
        nombreUsuario = request.form['nombreUsuario']
        password = request.form['password']
        data, count = supabase.table('Usuarios').select('*').eq('username', nombreUsuario).execute()
        if data:
            usuarios = data[1] 
            if usuarios:
                usuarioData = usuarios[0]
                if usuarioData["password"] == password:
                    user = Usuario(usuarioData["id"], usuarioData["username"], usuarioData["password"], usuarioData["rol"], usuarioData["nombre_completo"])
                    login_user(user)
                    if usuarioData["rol"] == 'Administrador':
                        return redirect(url_for('administrador_main'))
                    elif usuarioData["rol"] == 'Kinesiologo':
                        return redirect(url_for('kinesiologo_main'))
                flash ("Contraseña incorrecta", 'warning')
                return redirect(url_for('login'))
            flash ("Usuario no encontrado", 'danger')
            return redirect(url_for('login'))
        flash ("Usuario no encontrado", 'danger')
        return redirect(url_for('login'))
    return render_template('views/login.html')

@app.route("/logout", methods=['GET', 'POST'])
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route("/administrador_main", methods=['GET','POST'])
@login_required
def administrador_main():
    return render_template('views/admin/admin.html')

@app.route("/kinesiologo_main", methods=['GET','POST'])
@login_required
def kinesiologo_main():
    return render_template('views/kine/kine.html')

@app.route("/listar_pacientes", methods=['GET','POST'])
@login_required
def listar_pacientes():
    data, count = supabase.table('Pacientes').select('*').execute()
    pacientes = data[1]
    return render_template('views/kine/listarPacientes.html', pacientes=pacientes)

@app.route("/info_paciente/<int:id_paciente>", methods=['GET', 'POST'])
@login_required
def info_paciente(id_paciente):
    data, count = supabase.table('Pacientes').select('*').eq('id_paciente', id_paciente).execute()
    pacientes = data[1]
    pacienteData = pacientes[0]
    return render_template("views/kine/info_paciente.html", paciente=pacienteData)

@app.route("/crear_paciente/<int:id>", methods=['GET', 'POST'])
@login_required
def crear_paciente():
    if request.method == 'POST':
        pass
    return render_template("views/kine/crear_paciente.html")

@app.route("/crud_sesiones/<int:id_paciente>", methods=['GET', 'POST'])
@login_required
def crud_sesiones(id_paciente):
    data, count = supabase.table('Sesiones').select('*').eq('fk_id_paciente', id_paciente).execute()
    sesiones = data[1]
    return render_template('views/kine/sesiones.html', sesiones=sesiones)

@app.route("/video_template", methods=['GET','POST'])
@login_required
def video_template():    
    return render_template('views/kine/cam.html')

@app.route("/video_feed", methods=['GET','POST'])
@login_required
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

#Rutas de error
#404 y 500
@app.route('/error404')
@app.errorhandler(404)
def page_not_found(e):
    return render_template('views/404.html'), 404

@app.errorhandler(500)
def page_not_found(e):
    return render_template('views/500.html'), 500