from app import app, supabase, login_manager, csrf
from flask import render_template, Response, redirect, url_for, request, flash, Blueprint, session, jsonify
from flask_login import login_user, login_required, logout_user, current_user
import cv2 as cv
import pyttsx3

from app.functions import squat, pushup, ohp, bicep_curl, lateral_raise
from app.models import Usuario, Paciente, Ejercicio, Limitacion, Registro, Sesion

# Instanciación del blueprint
adultTrain = Blueprint('app', __name__)

# Ruta estándar
"""
@app.route("/ruta", methods=['GET', 'POST'])
@login_required
def ruta():
    return render_template("views/.html") redirect(url_for(''))
"""

######## Rutas comúnes:
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
    return render_template('views/otros/login.html')

@app.route("/logout", methods=['GET', 'POST'])
def logout():
    logout_user()
    return redirect(url_for('login'))

######## Rutas de error
#404 y 500
@app.route('/error404')
@app.errorhandler(404)
def page_not_found(e):
    return render_template('views/otros/404.html'), 404

@app.errorhandler(500)
def page_not_found(e):
    return render_template('views/otros/500.html'), 500

######## INICIO Rutas de Administrador

# Pantalla principal del administrador
@app.route("/administrador_main", methods=['GET','POST'])
@login_required
def administrador_main():
    return render_template('views/admin/admin.html')

### INICIO Ejercicios ###
# Crear y listar ejercicios
@app.route("/listar_ejercicios", methods=['GET', 'POST'])
@login_required
def listar_ejercicios():
    if request.method == 'POST':
        fk_id_usuario = request.form['fkIdUsuario']
        tipo = request.form['tipo']
        dificultad = request.form['dificultad']
        equipamiento = request.form['equipamiento']
        grupo_muscular = request.form['grupoMuscular']
        descripcion = request.form['descripcion']
        link_video = request.form['linkVideo']
        nombre = request.form['nombre']
        nuevoEjercicio = {'fk_id_usuario': fk_id_usuario, 'tipo': tipo, 'dificultad': dificultad, 'equipamiento': equipamiento, 'grupo_muscular': grupo_muscular, 'descripcion': descripcion, 'link_video': link_video, 'nombre': nombre}
        insert = supabase.table('Ejercicios').insert(nuevoEjercicio).execute()
        return redirect(url_for('listar_ejercicios'))
    ejercicios = supabase.table('Ejercicios').select('*').order('id', desc=False).execute()
    res = supabase.storage.list_buckets()
    print(res)
    return render_template("views/admin/CRUDejercicios/listarEjercicios.html", ejercicios=ejercicios.data)

@app.route("/eliminar_ejercicio/<int:id_ejercicio>", methods=['GET', 'POST'])
@login_required
def eliminar_ejercicio(id_ejercicio):
    delete = supabase.table('Ejercicios').delete().eq('id', id_ejercicio).execute()
    flash ('Ejercicio eliminado exitosamente', 'danger')
    return redirect(url_for('listar_ejercicios'))

@app.route("/editar_ejercicio/<int:id_ejercicio>", methods=['GET', 'POST'])
@login_required
def editar_ejercicio(id_ejercicio):
    if request.method == 'POST':
        fk_id_usuario = request.form['fkIdUsuario']        
        nombre = request.form['nombre']
        tipo = request.form['tipo']
        dificultad = request.form['dificultad']
        equipamiento = request.form['equipamiento']
        grupo_muscular = request.form['grupoMuscular']
        descripcion = request.form['descripcion']
        link_video = request.form['linkVideo']
        ejercicio = {'fk_id_usuario': fk_id_usuario, 'tipo': tipo, 'dificultad': dificultad, 'equipamiento': equipamiento, 'grupo_muscular': grupo_muscular, 'descripcion': descripcion, 'link_video': link_video, 'nombre': nombre}
        update = supabase.table('Ejercicios').update(ejercicio).eq('id', id_ejercicio).execute()
        return redirect(url_for('listar_ejercicios'))
    ejercicio = supabase.table('Ejercicios').select('*').eq('id', id_ejercicio).execute()
    if current_user.rol == 'Administrador':
        return render_template("views/admin/CRUDejercicios/editarEjercicio.html", ejercicio=ejercicio.data[0])
    elif current_user.rol == 'Kinesiologo':
        return render_template("views/kine/KCRUDejercicios/KeditarEjercicio.html", ejercicio=ejercicio.data[0])


@app.route("/listar_usuarios", methods=['GET', 'POST'])
@login_required
def listar_usuarios():
    return 'dasdasd' #render_template("views/.html") 

### FIN Ejercicios ###

######## FIN Rutas de Administrador

######## INICIO Rutas de Kinesiologo

@app.route("/kinesiologo_main", methods=['GET','POST'])
@login_required
def kinesiologo_main():    
    return render_template('views/kine/kine.html')

### INICIO pacientes ###
# Crear y listar pacientes
@app.route("/listar_pacientes/<int:id_kinesiologo>", methods=['GET','POST'])
@login_required
def listar_pacientes(id_kinesiologo): 
    limitaciones_posibles = ['ECcardiaca', 'ECrespiratoria', 'ECrenal', 'ECgastro', 'cognitiva', 'fisica']  # Reemplaza con los nombres reales de tus limitaciones   
    if request.method == 'POST':
        fk_id_kinesiologo = id_kinesiologo
        nombrePaciente = request.form['nombrePaciente']
        fechaNacimiento = request.form['fechaNacimiento']
        estatura = request.form['estatura']
        peso = request.form['peso']
        generoPaciente = request.form['genero']
        # Crear el diccionario para almacenar las limitaciones        
        limitaciones = {limitacion: limitacion in request.form.getlist('limitaciones') for limitacion in limitaciones_posibles}
        ecCardiaca = limitaciones['ECcardiaca']
        ecRespiratoria = limitaciones['ECrespiratoria']
        ecRenal = limitaciones['ECrenal']
        ecGastro = limitaciones['ECgastro']
        cognitiva = limitaciones['cognitiva']
        fisica = limitaciones['fisica']
        nuevoPaciente = {'fk_id_kinesiologo': fk_id_kinesiologo, 'fecha_nacimiento': fechaNacimiento, 'nombre_completo': nombrePaciente, 'genero': generoPaciente, 'peso': peso, 'estatura': estatura, 'EC_cardiaca': ecCardiaca, 'EC_respiratoria': ecRespiratoria, 'EC_renal': ecRenal, 'EC_gastrointestinal': ecGastro, 'cognitiva': cognitiva, 'fisica': fisica}
        insert = supabase.table('Pacientes').insert(nuevoPaciente).execute()    
        flash ('Paciente creado exitosamente', 'success')
        return redirect(url_for('listar_pacientes', id_kinesiologo=id_kinesiologo))
    pacientes = supabase.table('Pacientes').select('*').eq('fk_id_kinesiologo', id_kinesiologo).order('id_paciente', desc=False).execute()
    return render_template('views/kine/CRUDpacientes/listarPacientes.html', pacientes=pacientes.data)

# Mostrar info de paciente
@app.route("/info_paciente/<int:id_paciente>", methods=['GET', 'POST'])
@login_required
def info_paciente(id_paciente):
    paciente = supabase.table('Pacientes').select('*').eq('id_paciente', id_paciente).execute()
    return render_template("views/kine/CRUDpacientes/infoPaciente.html", paciente=paciente.data[0])

# Editar información de un paciente
@app.route("/editar_paciente/<int:id_paciente>", methods=['GET', 'POST'])
@login_required
def editar_paciente(id_paciente):
    limitaciones_posibles = ['ECcardiaca', 'ECrespiratoria', 'ECrenal', 'ECgastro', 'cognitiva', 'fisica']  # Reemplaza con los nombres reales de tus limitaciones   
    if request.method == 'POST':
        fk_id_kinesiologo = request.form['fkIdKinesiologo']
        nombrePaciente = request.form['nombrePaciente']
        fechaNacimiento = request.form['fechaNacimiento']
        estatura = request.form['estatura']
        peso = request.form['peso']
        generoPaciente = request.form['genero']
        # Crear el diccionario para almacenar las limitaciones        
        limitaciones = {limitacion: limitacion in request.form.getlist('limitaciones') for limitacion in limitaciones_posibles}
        ecCardiaca = limitaciones['ECcardiaca']
        ecRespiratoria = limitaciones['ECrespiratoria']
        ecRenal = limitaciones['ECrenal']
        ecGastro = limitaciones['ECgastro']
        cognitiva = limitaciones['cognitiva']
        fisica = limitaciones['fisica']        
        update = supabase.table('Pacientes').update(paciente).eq("id_paciente", id_paciente).execute()
        flash ('Paciente editado exitosamente', 'info')
        return redirect(url_for('listar_pacientes'))
    paciente = supabase.table('Pacientes').select('*').eq('id_paciente', id_paciente).execute()
    return render_template("views/kine/CRUDpacientes/editarPaciente.html", paciente=paciente.data[0])

# Eliminar un paciente
@app.route("/eliminar_paciente/<int:id_paciente>", methods=['GET', 'POST'])
@login_required
def eliminar_paciente(id_paciente):
    delete = supabase.table('Pacientes').delete().eq('id_paciente', id_paciente).execute()
    flash ('Paciente eliminado exitosamente', 'danger')
    return redirect(url_for('listar_pacientes', id_kinesiologo=current_user.get_id()))

### FIN pacientes ###

### INICIO sesiones ###
# Crear y listar sesiones
@app.route("/listar_sesiones/<int:id_paciente>", methods=['GET', 'POST'])
@login_required
def listar_sesiones(id_paciente):
    if request.method == 'POST':
        fk_id_paciente = id_paciente        
        fecha = request.form['fecha']
        objetivo = request.form['objetivo']
        evaluacion = request.form['evaluacion']
        comentarios = request.form['comentarios']
        sesion = {'fk_id_paciente': fk_id_paciente, 'fecha': fecha, 'objetivo': objetivo, 'evaluacion': evaluacion, 'comentarios': comentarios}
        insert =  supabase.table('Sesiones').insert(sesion).execute()
        flash('Sesión creada exitosamente', 'success')
        return redirect(url_for('listar_sesiones', id_paciente=id_paciente))
    sesiones = supabase.table('Sesiones').select('*').eq('fk_id_paciente', id_paciente).execute()
    return render_template('views/kine/CRUDsesiones/listarSesiones.html', sesiones=sesiones.data)

# Editar una sesión de un paciente
@app.route("/editar_sesion/<int:id_sesion>", methods=['GET', 'POST'])
@login_required
def editar_sesion(id_sesion):
    if request.method == 'POST':
        fk_id_paciente = request.form['idPaciente']
        fecha = request.form['fecha']
        objetivo = request.form['objetivo']
        evaluacion = request.form['evaluacion']
        comentarios = request.form['comentarios']
        sesion = {'fk_id_paciente': fk_id_paciente, 'fecha': fecha, 'objetivo': objetivo, 'evaluacion': evaluacion, 'comentarios': comentarios}
        update =  supabase.table('Sesiones').update(sesion).eq('id', id_sesion).execute()
        flash('Sesión editada exitosamente', 'info')
        return redirect(url_for('listar_sesiones', id_paciente=fk_id_paciente))
    sesion = supabase.table('Sesiones').select('*').eq('id', id_sesion).execute()
    return render_template("views/kine/CRUDsesiones/editarSesion.html", sesion=sesion.data[0])

# Eliminar una sesión de un paciente
@app.route("/eliminar_sesion/<int:id_sesion>/<int:id_paciente>", methods=['GET', 'POST'])
@login_required
def eliminar_sesion(id_sesion, id_paciente):
    data, count = supabase.table('Sesiones').delete().eq('id', id_sesion).execute()
    flash ("Sesión eliminada exitosamente", 'danger')
    return redirect(url_for('listar_sesiones', id_paciente=id_paciente))

### FIN sesiones ###

### INICIO rutinas ###
# Rutas de creación de rutinas
@app.route("/crear_rutina/<int:id_paciente>", methods=['GET','POST'])
@login_required
def crear_rutina(id_paciente):     
    if request.method == 'POST':
        fk_id_paciente = id_paciente
        tituloRutina = request.form['tituloRutina']
        objetivoRutina = request.form['objetivoRutina']
        tiempoRutina = request.form['tiempoRutina']
        notasRutina = request.form['notasRutina']
        rutina = {'fk_id_paciente' : id_paciente, 'titulo': tituloRutina, 'objetivo': objetivoRutina, 'duracion': tiempoRutina, 'notas': notasRutina}
        insert = supabase.table('Rutinas').insert(rutina).execute()
        flash ('Rutina creada exitosamente', 'success')
        return redirect(url_for('crear_rutina', id_paciente=id_paciente))
    rutinas = supabase.table('Rutinas').select('*').order('id', desc=False).execute()
    return render_template('views/kine/CRUDrutinas/crearRutina.html', rutinas=rutinas.data, id_paciente=id_paciente)

@app.route("/editar_rutina/<int:id_rutina>/<int:id_paciente>", methods=['GET','POST'])
@login_required
def editar_rutina(id_rutina, id_paciente):
    if request.method == 'POST':
        tituloRutina = request.form['tituloRutina']
        objetivoRutina = request.form['objetivoRutina']
        tiempoRutina = request.form['tiempoRutina']
        notasRutina = request.form['notasRutina']  
        rutina = {'titulo': tituloRutina, 'objetivo': objetivoRutina, 'duracion': tiempoRutina, 'notas': notasRutina}
        update =  supabase.table('Rutinas').update(rutina).eq('id', id_rutina).execute()
        flash ('Rutina creada exitosamente', 'success')
        return redirect(url_for('crear_rutina', id_paciente=id_paciente))
    rutina = supabase.table('Rutinas').select('*').eq('id', id_rutina).execute()
    return render_template('views/kine/CRUDrutinas/editarRutina.html', rutina = rutina.data[0])

@app.route("/eliminar_rutina/<int:id_rutina>/<int:id_paciente>", methods=['GET', 'POST'])
@login_required
def eliminar_rutina(id_rutina, id_paciente):
    delete = supabase.table('Rutinas').delete().eq('id', id_rutina).execute()
    flash ('Rutina eliminada exitosamente', 'danger')
    return redirect(url_for('crear_rutina', id_paciente=id_paciente)) 

### INICIO seleccionar ejercicio ###

@app.route("/seleccionar_ejercicio/<int:id_rutina>", methods=['GET','POST'])
@login_required
def seleccionar_ejercicio(id_rutina):
    ejercicios = supabase.table('Ejercicios').select('*').order('id', desc=False).execute()
    return render_template('views/kine/CRUDrutinas/seleccionarEjercicios.html', ejercicios = ejercicios.data)

### FIN seleccionar ejercicio ###

### FIN rutinas ###

### INICIO RTR ###

# Vista para elegir ejercicio, este entrega como argumento el nombre de ejercicio a evaluar para la ruta RTR
@app.route("/elegir_ejercicio", methods=['GET', 'POST'])
@login_required
def elegir_ejercicio():
    return render_template("views/kine/RTR/elegirEjercicio.html")

# Ruta de retroalimentación en tiempo real, esta envía el ejercicio elegido a la función que evalúa
@app.route("/RTR/<ejercicio>", methods=['GET','POST'])
@login_required
def RTR(ejercicio):
    return render_template('views/kine/RTR/cam.html', ejercicio=ejercicio)

# Esta ruta se ejecuta una vez que es llamada RTR para renderizar la RTR
@app.route("/video_feed/<ejercicio>", methods=['GET','POST'])
@login_required
def video_feed(ejercicio):
    cap = cv.VideoCapture(0, cv.CAP_MSMF)
    if ejercicio == 'squat':
        return Response(squat(cap), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif ejercicio == 'pushup':
        return Response(pushup(cap), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif ejercicio == 'ohp':
        return Response(ohp(cap), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif ejercicio == 'bicep_curl':
        return Response(bicep_curl(cap), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif ejercicio == 'lateral_raise':
        return Response(lateral_raise(cap), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    cap.release()

### FIN RTR ###

######## FIN Rutas de Kinesiologo

