import tempfile
import streamlit as st
import streamlit_scrollable_textbox as stx
import cv2 as cv
from functions import crunches, crunches2, bicepCurl, bicepCurl2, pushup, pushup2, squat, squat2

def main():
    # Default sidebar
    ejercicios = ["Elige un ejercicio", "Sentadillas (Squats)", "Flexiones (Push ups)", "Abdominales (Crunches)", "Curl de bicep (Bicep curl) Curl"]

    st.sidebar.header("🏋️‍♂️🏋️‍♀️PersonApp Trainer🏋️‍♀️🏋️‍♂️")
    st.sidebar.subheader("Esta aplicación te ayuda a contabilizar las repeticiones y corregir la forma en ciertos ejercicios.")
    st.sidebar.caption("Asegurate de leer las indicaciones antes de usar la aplicación")
    live = st.sidebar.checkbox("¿Desea analizar a través de su cámara?")
    st.sidebar.divider()
    ejercicio = st.sidebar.selectbox("¿Qué ejercicio desea evaluar?", ejercicios)
    with st.sidebar.form('Upload',clear_on_submit=True):
        if not live:
            archivo = st.file_uploader("Suba su video a analizar (Solo 1 video).", type=['mp4', 'mov', 'avi'])
            submit = st.form_submit_button("Empezar análisis del video")
        
    if live:
        st.sidebar.divider()
        start = st.sidebar.checkbox("Comenzar el análisis")
        if ejercicio == ejercicios[0]:
            st.header("🏋️‍♂️🏋️‍♀️PersonApp Trainer🏋️‍♀️🏋️‍♂️")
            st.subheader("Esta aplicación te ayuda a contabilizar las repeticiones y corregir la forma en ciertos ejercicios.")
            st.caption("Para sacar el máximo provecho a esta aplicación se recomiendan las siguientes reglas:")        
            st.text("Reglas generales:")
            rules = """1) Al seleccionar algún ejercicio el análisis comenzará de inmediato, por lo que se recomienda que te encuentres en la posición ideal para el ejercicio a evaluar.
            2) Las indicaciones generales para cada ejercicio se indican más abajo.
            3) Para todos los ejercicios se recomienda que la persona se encuentre centrada en la cámara.
            4) Cada ejercicio consta de una fase excéntrica y concéntrica. Si vez que una repetición no se contó, es porque no cumpliste los requisitos de alguna de estas dos fases.
            5) Se sugiere evitar el uso de ropa holgada, ya que a veces genera problemas en la detección de articulaciones.
            """
            stx.scrollableTextbox(rules, height=250)

        if ejercicio == ejercicios[1] and not start:
            st.text("Reglas para sentadillas:")
            squatRules = """1) La persona debe ser visible desde un plano lateral o lateral/frontal.
            2) La posición incial debe ser de pie.
            """
            stx.scrollableTextbox(squatRules)
            st.text("Un referencia de como ubicarte frente a la cámara es la siguiente:")
            st.video("GerontoApp/videos/squat2.mp4")
            
        if ejercicio == ejercicios[1] and start:
            squat()
            
        if ejercicio == ejercicios[2] and not start:
            st.text("Reglas para flexiones de brazos:")
            pushupRules = """1) La persona debe ser visible desde un plano frontal o lateral.
            2) La posición incial debe ser apoyado con las manos en el piso y los brazos extendidos.
            """
            stx.scrollableTextbox(pushupRules)
            st.text("Un referencia de como ubicarte frente a la cámara es la siguiente:")
            st.video("GerontoApp/videos/push1.mp4")

        if ejercicio == ejercicios[2] and start:
            pushup()

        if ejercicio == ejercicios[3] and not start:
            st.text("Reglas para abdominales:")
            crunchesRules = """1) La persona debe ser visible desde un plano lateral.
            2) La persona se debe encontrar acostada de cúbito supino con las rodillas flectadas formando un ángulo de cadera mayor a 115 grados.
            """
            stx.scrollableTextbox(crunchesRules)
            st.text("Un referencia de como ubicarte frente a la cámara es la siguiente:")
            st.video("GerontoApp/videos/abs3.mp4")

        if ejercicio == ejercicios[3] and start:
            st.text("Reglas para los abdominales:")
            bicep = """1) La persona debe encontrarse recostada desde un plano lateral.
            2)La posición inicial debe ser con la espalda apoyada en el suelo y las rodillas con ángulo de inclinación similar a 45 grados.
            """
            crunches()

        if ejercicio == ejercicios[4] and not start:
            st.text("Reglas para curl de biceps:")
            bicepRules = """1) La persona debe ser visible desde un plano lateral o lateral/frontal.
            2) La posición incial debe ser de pie con los brazos completamente extendidos hacia abajo.
            3) Los codos se deben mantener pegados al cuerpo para evitar repeticiones inválidas.
            """
            stx.scrollableTextbox(bicepRules)
            st.text("Un referencia de como ubicarte frente a la cámara es la siguiente:")
            st.video("GerontoApp/videos/bicep1.mp4")

        if ejercicio == ejercicios[4] and start:
            bicepCurl()

    else:
        if ejercicio == ejercicios[0]:            
                st.header("🏋️‍♂️🏋️‍♀️PersonApp Trainer🏋️‍♀️🏋️‍♂️")
                st.subheader("Esta aplicación te ayuda a contabilizar las repeticiones y corregir la forma en ciertos ejercicios.")
                st.divider()    

        if archivo and submit:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(archivo.read())
            cap = cv.VideoCapture(archivo.name)
            st.sidebar.text('Video original de entrada')
            st.sidebar.video(temp_file.name)    
            
            if ejercicio == ejercicios[1]:
                st.video(temp_file.name)
                st.divider()                
                squat2(temp_file.name)
                
            elif ejercicio == ejercicios[2]:
                st.video(temp_file.name)
                st.divider()
                pushup2(temp_file.name)

            elif ejercicio == ejercicios[3]:
                st.video(temp_file.name)
                st.divider()
                crunches2(temp_file.name)
                
                

            elif ejercicio == ejercicios[4]:
                st.video(temp_file.name)
                st.divider()
                bicepCurl2(temp_file.name)
                

if __name__ == '__main__':
    main()