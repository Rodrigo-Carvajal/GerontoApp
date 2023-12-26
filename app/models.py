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
