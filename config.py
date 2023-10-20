#Creación de clase Config:
class Config:
    SECRET_KEY = 'jdamksdw-baq_#B#WFV-ZC#V_A@Q=d-kb1i41VFM!'
    FLASK_APP = 'app'

#Clase usada para la configuración de la aplicación en desarrollo:
class DevelopmentConfig():
    DEBUG=True
    

#Diccionario de las distintas configuraciones existentes:    
config = {
    'development': DevelopmentConfig
    
}