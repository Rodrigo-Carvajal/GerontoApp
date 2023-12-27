# Importación de app:
from app import app
from config import config
from waitress import serve
import webbrowser

# Ejecución de la aplicación:
if __name__ == '__main__':
    app.config.from_object(config['development'])
    #app.run(host="0.0.0.0", port=5000, debug= True)    
    webbrowser.open('http://127.0.0.1:5000/')
    serve(app, host='0.0.0.0', port=5000)