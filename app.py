# Importación de app:
from app import app, socketio
from config import config

# Ejecución de la aplicación:
if __name__ == '__main__':
    app.config.from_object(config['development'])
    app.run(host="0.0.0.0", port=5000, debug= True)