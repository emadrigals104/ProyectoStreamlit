import streamlit as st
import pandas as pd
from io import StringIO
from PIL import Image


  
def cargar_archivo():
    st.markdown("# :computer: Mi Aplicación Streamlit Para Predecir El Abandono de Clientes")
    col_1,col_2,col_3 = st.columns([1, 2, 1])
    with col_2:
        st.image(image="images/customer-rating.png",width=300)
    st.markdown(
        """
        ### ¡Bienvenido a la aplicación de predicción de abandono de clientes! 
            
        Esta herramienta te ayudará a identificar a los clientes que tienen más probabilidades de abandonar tu negocio, lo que te permitirá tomar medidas preventivas y mejorar la retención de clientes.

        La aplicación utiliza un modelo de Machine Learning entrenado para analizar los patrones en los datos de tus clientes y predecir si abandonarán el servicio en un futuro próximo.

        Para comenzar, sigue estos sencillos pasos:

        1. Sube tu conjunto de datos de clientes utilizando la opción "Subir" que se encuentra en la parte de abajo. Asegúrate de que tu archivo esté en formato CSV.
        2. Puedes dirigirte a la página de Análisis Exploratorio de Datos para analizar los datos que se cargaron.
        3. En la página de Preprocesamiento puedes eliminar columnas o valores vacíos y atípicos
        4. Una vez realizado todo ese procedimiento puedes dirigirte a Entrenamiento para Entrenar tu modelo con los parámetros de cada modelo.
        2. Sube el modelo entrenado que utilizarás para realizar las predicciones. Asegúrate de que el modelo esté en formato PKL.
        3. Navega por las opciones en la barra lateral para ajustar los parámetros y personalizar tu análisis.
        4. Revisa las predicciones y las probabilidades generadas por el modelo.
        5. Descarga el informe con las predicciones y las probabilidades para cada cliente.

        ¡Buena suerte en tus esfuerzos para mejorar la retención de clientes y mantener a tus clientes satisfechos!
        """
    )
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        gif = Image.open("images/servicio-al-cliente.png")
        st.image(gif, width=300)
    st.markdown("### :open_file_folder: Sube al archivo que vas a entrenar")
    uploaded_file = st.file_uploader("Cargue su archivo CSV", type=["csv"])
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.df = pd.read_csv(uploaded_file)
        st.markdown('''
                    ## Acerca de los datos
                    ''')
        col_datos,col_datos2 = st.columns(2)
        with col_datos:
            st.markdown('''
                    **Genero:** si el cliente es hombre o mujer
                    
                    **PersonaMayor:** si el cliente es una persona mayor o no (1, 0)
                    
                    **Socio:** si el cliente tiene socio o no (Sí, No)
                    
                    **Dependientes:** si el cliente tiene dependientes o no (Sí, No)
                    
                    **Permanencia:** Indica el número de meses que el cliente ha estado en la empresa.
                    
                    **ServicioTelefonico:** Si el cliente tiene servicio telefónico o no (Sí, No)
                    
                    **VariasLineas:** si el cliente tiene múltiples líneas o no (Sí, No, Sin servicio telefónico)
                    
                    **ServicioInternet:** Proveedor de servicios de Internet del cliente (DSL, Fibra óptica, No)
                    
                    **SeguridadLinea:** si el cliente tiene seguridad en línea o no (Sí, No, Sin servicio de Internet)
                    
                    **CopiaSeguridadLinea:** si el cliente tiene copia de seguridad en línea o no (Sí, No, Sin servicio de Internet)
                ''')

        with col_datos2:
            st.markdown('''
                    **ProteccionDispositivo:** si el cliente tiene protección de dispositivo o no (Sí, No, Sin servicio de Internet)
                    
                    **SoporteTecnico:** si el cliente tiene soporte técnico o no (Sí, No, Sin servicio de Internet)
                    
                    **ServicioTV:** si el cliente tiene streaming de TV o no (Sí, No, Sin servicio de Internet)
                    
                    **ServicioPeliculas:** si el cliente tiene películas en streaming o no (Sí, No, Sin servicio de Internet)
                    
                    **Contrato:** El plazo del contrato del cliente (Mes a mes, Un año, Dos años)
                    
                    **FacturacionElectronica:** si el cliente tiene facturación electrónica o no (Sí, No)
                    
                    **MetodoPago:** el método de pago del cliente (Cheque electrónico, Cheque enviado por correo, Transferencia bancaria (automática), Tarjeta de crédito (automática))
                    
                    **RecargoMensual:** el monto cobrado al cliente mensualmente
                    
                    **TotalRecargo:** El monto total cobrado al cliente
                    
                    **Abondono:** Si el cliente abandonó o no (Sí o No)
                ''')

        st.write(f"Contenido del archivo {st.session_state.uploaded_file.type}")
        st.markdown("### :chart_with_upwards_trend: ¿Deseas ver el dataset?, da clic aquí abajo")
        mostrar_dataset = st.checkbox("Mostrar dataset")

        if mostrar_dataset:
            st.write(st.session_state.df)
    elif "uploaded_file" in st.session_state:
        
        
        st.markdown('''
                    ## Acerca de los datos
                    ''')
        col_datos3,col_datos4 = st.columns(2)
        with col_datos3:
            st.markdown('''
                    **Genero:** si el cliente es hombre o mujer
                    
                    **PersonaMayor:** si el cliente es una persona mayor o no (1, 0)
                    
                    **Socio:** si el cliente tiene socio o no (Sí, No)
                    
                    **Dependientes:** si el cliente tiene dependientes o no (Sí, No)
                    
                    **Permanencia:** Indica el número de meses que el cliente ha estado en la empresa.
                    
                    **ServicioTelefonico:** Si el cliente tiene servicio telefónico o no (Sí, No)
                    
                    **VariasLineas:** si el cliente tiene múltiples líneas o no (Sí, No, Sin servicio telefónico)
                    
                    **ServicioInternet:** Proveedor de servicios de Internet del cliente (DSL, Fibra óptica, No)
                    
                    **SeguridadLinea:** si el cliente tiene seguridad en línea o no (Sí, No, Sin servicio de Internet)
                    
                    **CopiaSeguridadLinea:** si el cliente tiene copia de seguridad en línea o no (Sí, No, Sin servicio de Internet)
                ''')

        with col_datos4:
            st.markdown('''
                    **ProteccionDispositivo:** si el cliente tiene protección de dispositivo o no (Sí, No, Sin servicio de Internet)
                    
                    **SoporteTecnico:** si el cliente tiene soporte técnico o no (Sí, No, Sin servicio de Internet)
                    
                    **ServicioTV:** si el cliente tiene streaming de TV o no (Sí, No, Sin servicio de Internet)
                    
                    **ServicioPeliculas:** si el cliente tiene películas en streaming o no (Sí, No, Sin servicio de Internet)
                    
                    **Contrato:** El plazo del contrato del cliente (Mes a mes, Un año, Dos años)
                    
                    **FacturacionElectronica:** si el cliente tiene facturación electrónica o no (Sí, No)
                    
                    **MetodoPago:** el método de pago del cliente (Cheque electrónico, Cheque enviado por correo, Transferencia bancaria (automática), Tarjeta de crédito (automática))
                    
                    **RecargoMensual:** el monto cobrado al cliente mensualmente
                    
                    **TotalRecargo:** El monto total cobrado al cliente
                    
                    **Abondono:** Si el cliente abandonó o no (Sí o No)
                ''')
        
        st.markdown(f"Archivo previamente subido: **{st.session_state.uploaded_file.name}**")
        st.markdown("### :chart_with_upwards_trend: ¿Deseas ver el dataset?, da clic aquí abajo")
        mostrar_dataset = st.checkbox("Mostrar dataset")
        if mostrar_dataset:
            st.write(st.session_state.df)
        



