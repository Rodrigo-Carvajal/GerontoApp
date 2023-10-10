from flask import Flask
from flask_login import LoginManager
from config import config
from flask_wtf.csrf import CSRFProtect
from supabase import create_client, Client

# Creación de la aplicación flask y su secret key
app = Flask(__name__)
app.secret_key = 'jdamksdw-baq_#B#WFV-ZC#V_A@Q=d-kb1i41VFM!'
csrf = CSRFProtect(app)
csrf.init_app(app)

# Claves de supabase (Base de datos PostgreSQL)
supabase_url = 'https://iwzzyjvqvbgmkaybieeh.supabase.co'
supabase_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml3enp5anZxdmJnbWtheWJpZWVoIiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTY2MzA2MzQsImV4cCI6MjAxMjIwNjYzNH0.y6I6YCdFTW_JVCRHHrJVN9kFe1LXDgZPnl_pi9rrRoM'

# Cliente de supabase
supabase: Client = create_client(supabase_url, supabase_key)

login_manager = LoginManager()
login_manager.init_app(app)

# Importación de blueprint (Rutas):
from app.controllers import adultTrain
app.register_blueprint(adultTrain)