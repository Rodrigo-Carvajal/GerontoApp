import mediapipe as mp
import cv2 as cv
import numpy as np
from math import acos, degrees
from app import login_manager, supabase
from flask_login import UserMixin
        
@login_manager.user_loader
def load_user(id):    
    data, count = supabase.table('Usuarios').select('*').eq('id', id).execute()
    usuarios = data[1]
    usuarioData = usuarios[0]
    return Usuario(usuarioData["id"], usuarioData["username"], usuarioData["password"], usuarioData["rol"], usuarioData["nombre_completo"])

class Usuario(UserMixin):    
    def __init__(self, id, username, password, rol, nombre_completo):
        self.id = id
        self.username = username        
        self.password = password
        self.rol = rol
        self.nombre_completo = nombre_completo

    def get_id(self):
        return self.id

class Paciente():    
    def __init__(self, id_paciente, fk_id_kinesiologo, fk_id_limitacion, fecha_nacimiento, nombre_completo, genero, peso, estatura):
        self.id_paciente = id_paciente
        self.fk_id_kinesiologo = fk_id_kinesiologo
        self.fk_id_limitacion = fk_id_limitacion
        self.fecha_nacimiento = fecha_nacimiento
        self.nombre_completo = nombre_completo
        self.genero = genero
        self.peso = peso
        self.estatura = estatura

    def get_limitacion(self):
        return self.fk_id_limitacion

class Ejercicio():
    def __init__(self, id, fk_id_usuario, fk_id_limitacion, tipo, dificultad, equipamiento, grupo_muscular, descripcion, link_video, nombre):
        self.id = id
        self.fk_id_usuario = fk_id_usuario
        self.fk_id_limitacion = fk_id_limitacion
        self.tipo = tipo
        self.dificultad = dificultad
        self.equipamiento = equipamiento
        self.grupo_muscular = grupo_muscular
        self.descripcion = descripcion
        self.link_video = link_video
        self.nombre = nombre
        
class Limitacion():
    def __init__(self, id, silla_de_ruedas, sin_brazos):
        self.id = id
        self.silla_de_ruedas = silla_de_ruedas
        self.sin_brazos = sin_brazos
        
class Registro():
    def __init__(self, id, fk_id_sesion, fk_id_ejercicio, serie, repeticiones, peso, duracion, comentarios, esfuerzo_percibido, evaluacion):
        self.id = id
        self.fk_id_sesion = fk_id_sesion
        self.fk_id_ejercicio = fk_id_ejercicio
        self.serie = serie
        self.repeticiones = repeticiones
        self.peso = peso
        self.duracion = duracion
        self.comentarios = comentarios
        self.esfuerzo_percibido = esfuerzo_percibido
        self.evaluacion = evaluacion
        
class Sesion():
    def __init__(self, id, fk_id_paciente, fecha, objetivo, evaluacion, comentarios):
        self.id = id
        self.fk_id_paciente = fk_id_paciente
        self.fecha = fecha
        self.objetivo = objetivo
        self.evaluacion = evaluacion
        self.comentarios = comentarios