import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px



@st.cache_data(persist="disk")
def columnas():
    if "data_entre" in st.session_state:
        if "boton_pasos_no_realizados" in st.session_state:
            if st.session_state.boton_pasos_no_realizados:
                data=st.session_state.data_entre
                columnas = data.columns
                st.session_state.columnas_actualizadas = True
                if st.session_state.boton_pasos_no_realizados == False:
                    data=st.session_state.data_entre
                    columnas = data.columns
                    st.session_state.columnas_actualizadas = False
    else:
      st.markdown("## No existen Columnas, debes entrenar el modelo")  
    return columnas

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Hoja1')
    writer.book.save(output)
    output.seek(0)
    return output

def prediccion():
    st.markdown("# :chart_with_upwards_trend: Predicción de Abandono de los Clientes")
    st.markdown("""
    En esta sección, puedes cargar un conjunto de datos con características de clientes y predecir si abandonarán los servicios o no. Nuestro modelo utiliza información demográfica y de uso del servicio para predecir la probabilidad de abandono.

    **Nota:**
    
    La sección de predicción tiene un archivo de un modelo entrenado por defecto que contiene las mejores configuraciones para la predicción.
    
    Si desea colocar su propio modelo entrenado debe entrenar el modelo, descargar el archivo PKL y subirlo en la parte de Archivo PKL

    **Instrucciones:**
    1. Puedes predecir manualmente con los controles colocados al lado izquierdo, podrás ingresar datos de un usuario y visualizar su predicción.
    2. Si deseas predecir un conjunto de archivos, carga el archivo CSV que contiene los datos de los clientes.
    2. Verifica que el archivo se haya cargado correctamente, visualizando el conjunto de datos.
    3. Una vez cargado la predicción se ejecuta automaticamente y podrás visualizar las predicciones.

    Una vez que obtengas las predicciones, podrás descargarlas en formato Excel y CSV y visualizar la información en la aplicación. Estas predicciones te permitirán tomar decisiones informadas sobre cómo retener a los clientes en riesgo de abandonar los servicios.
    """)

    st.markdown("""---""")
    st.markdown("### Suba el modelo entrenado, si no ha entrenado el modelo puede usar el modelo por defecto")
    modelo_entrenado = st.file_uploader("Suba el Modelo Entrenado",type=["pkl"])
    st.markdown("""---""")
    st.sidebar.header('Datos ingresados por el usuario')
    st.subheader('Datos ingresados por el usuario')
    datos_nuevos = st.sidebar.file_uploader("Cargue su Archivo CSV", type=["csv"])
    if modelo_entrenado is not None:
        carga_modelo = pickle.load(modelo_entrenado)
        if datos_nuevos is not None:
            dataset_ingresado = pd.read_csv(datos_nuevos)
            st.write(dataset_ingresado)
            columnas_comunes = dataset_ingresado.columns.intersection(columnas())
            dataset_nuevo = dataset_ingresado[columnas_comunes]
            if st.checkbox("Mostrar Dataset"):
                st.write(dataset_nuevo)
 
        else:
            column = columnas()
            boton_actualizar = st.button("Actualizar Columnas")
            if boton_actualizar:
                columnas.clear()
                column = columnas()
                if boton_actualizar == False:
                    columnas.clear()
                    column = columnas()
            datos_dicc = {}
            if "Genero" in column:
                genero = st.sidebar.selectbox('Genero',("Femenino","Masculino"))
                datos_dicc['Genero']= genero
            if "PersonaMayor" in column:
                PersonaMayor = st.sidebar.selectbox('¿Es Una Persona Adulta Mayor?(Si =1,No=0)', (0,1))
                datos_dicc["PersonaMayor"]=PersonaMayor
            if "Socio" in column:
                Socio = st.sidebar.selectbox('¿Eres socio?', ("Si","No"))
                datos_dicc["Socio"]=Socio
            if "Dependientes" in column:
                Dependientes = st.sidebar.selectbox('¿Eres Dependiente?',("Si","No"))
                datos_dicc["Dependientes"]=Dependientes
            if "Permanencia" in column:
                Permanencia = st.sidebar.slider('¿Cuantos Meses tienes de Contrato?',0,72,29)
                datos_dicc["Permanencia"]=Permanencia
            if "ServicioTelefonico" in column:
                ServicioTelefonico = st.sidebar.selectbox('¿Tienes Servicio Telefónico?', ("Si","No"))
                datos_dicc["ServicioTelefonico"]=ServicioTelefonico
            if "VariasLineas" in column:
                VariasLineas = st.sidebar.selectbox('¿Tiene Múlitples Líneas?', ("Si","No","Sin Servicio Telefónico"))
                datos_dicc["VariasLineas"]=VariasLineas
            if "ServicioInternet" in column:
                ServicioInternet = st.sidebar.selectbox('¿Que tipo de servicio de Internet Tiene?', ("DLS","No","Fibra Óptica"))
                datos_dicc["ServicioInternet"]=ServicioInternet
            if "SeguridadLinea" in column:
                SeguridadLinea = st.sidebar.selectbox('¿Tiene Seguridad En Línea?', ("Si","No","Sin Servicio de Internet"))
                datos_dicc["SeguridadLinea"]=SeguridadLinea
            if "CopiaSeguridadLinea" in column:
                CopiaSeguridadLinea = st.sidebar.selectbox('Tiene Copia de Seguridad en Línea ?', ("Si","No","Sin Servicio de Internet"))
                datos_dicc["CopiaSeguridadLinea"]=CopiaSeguridadLinea
            if "ProteccionDispositivo" in column:
                ProteccionDispositivo = st.sidebar.selectbox('¿Tiene Protección del Dispositivo?', ("Si","No","Sin Servicio de Internet"))
                datos_dicc["ProteccionDispositivo"]=ProteccionDispositivo
            if "ServicioTecnico" in column:
                ServicioTecnico = st.sidebar.selectbox('¿Tiene Soporte Técnico?', ("Si","No","Sin Servicio de Internet"))
                datos_dicc["ServicioTecnico"]=ServicioTecnico
            if "ServicioTV" in column:
                ServicioTV = st.sidebar.selectbox('¿Tiene Servicio de TV?', ("Si","No","Sin Servicio de Internet"))
                datos_dicc["ServicioTV"]=ServicioTV
            if "ServicioPeliculas" in column:
                ServicioPeliculas = st.sidebar.selectbox('¿Tiene Servicio de Películas?', ("Si","No","Sin Servicio de Internet"))
                datos_dicc["ServicioPeliculas"]=ServicioPeliculas
            if "Contrato" in column:
                Contrato = st.sidebar.selectbox('Tipo de Contrato del Cliente', ("Mensual","un anio","dos anios"))
                datos_dicc["Contrato"]=Contrato
            if "FacturacionElectronica" in column:
                FacturacionElectronica = st.sidebar.selectbox('¿Recibe Factura Electrónica?', ("Si","No"))
                datos_dicc["FacturacionElectronica"]=FacturacionElectronica
            if "MetodoPago" in column:
                MetodoPago = st.sidebar.selectbox('¿Cuál es el Metodo de Pago?', ("Cheque Electrónico","Cheque por Correo","Transferencia bancaria (automática)","Tarjeta de crédito (automática)"))
                datos_dicc["MetodoPago"]=MetodoPago
            if "RecargoMensual" in column:
                RecargoMensual = st.sidebar.number_input('Recargo Mensual',0.00,200.00,70.35)
                datos_dicc["RecargoMensual"]=RecargoMensual
            if "TotalRecargo" in column:
                TotalRecargo = st.sidebar.number_input('Recargo Anual',0.00,10000.00,1000.00)
                datos_dicc["TotalRecargo"]=TotalRecargo

            dataset_nuevo = pd.DataFrame(datos_dicc, index=[0])
            st.write('A la espera de que se cargue el archivo CSV. Actualmente usando parámetros de entrada de ejemplo (que se muestran a continuación).')
            st.write(dataset_nuevo)              
        
        for i in dataset_nuevo.select_dtypes(include='object').columns:
            dataset_nuevo[i] = LabelEncoder().fit_transform(dataset_nuevo[i])
        scaler = StandardScaler().fit(dataset_nuevo[["TotalRecargo"]])
        dataset_nuevo["TotalRecargo"] = scaler.transform(dataset_nuevo[["TotalRecargo"]])
        scaler = StandardScaler().fit(dataset_nuevo[["RecargoMensual"]])
        dataset_nuevo["RecargoMensual"] = scaler.transform(dataset_nuevo[["RecargoMensual"]])
        
        
        prediccion_modelo = carga_modelo.predict(dataset_nuevo)
        prediction_proba_modelo = carga_modelo.predict_proba(dataset_nuevo)
        
        if datos_nuevos is not None:
            df_abandono = pd.DataFrame(prediccion_modelo,columns=["Abandono"])
            df_abandono = df_abandono.applymap(lambda x: "No" if x == 0 else "Si")    
            df_abandono = pd.DataFrame(prediction_proba_modelo.argmax(axis=1), columns=["Abandono"])
            df_abandono = df_abandono.applymap(lambda x: "No" if x == 0 else "Si")
            probabilidades = np.where(df_abandono["Abandono"] == "No", prediction_proba_modelo[:, 0], prediction_proba_modelo[:, 1])

            df_resultado = pd.DataFrame({"Abandono": df_abandono["Abandono"], "Probabilidad": probabilidades})
            df_unido = pd.concat([dataset_ingresado,df_resultado],axis=1)
            csv_data = df_unido.to_csv(index=False)
            st.divider()
            st.markdown("### Datos con la Predicción")
            mostrar_prediccion = st.checkbox("Mostrar Predicción Final")
            if mostrar_prediccion:
                st.write(df_unido)
            st.divider()
            st.markdown("### Gráficos de la predicción")
            col_gra1, col_gra2 = st.columns((5,5))
            valores_categoricas = df_unido["Abandono"].value_counts()
            colorscale = px.colors.sequential.YlOrBr
            num_categorias = len(valores_categoricas.index)
            step_size = int(len(colorscale) / num_categorias)
            colores = colorscale[::step_size]
            
            
            with col_gra1:

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=valores_categoricas.index,
                    y=valores_categoricas,
                    text=valores_categoricas,
                    textposition='auto',
                    hovertemplate='%{x}: <br>valores_categoricas: %{y}',
                    marker=dict(color=colores)

                ))

                fig.update_layout(
                    title=f"Gráfico de Barras - Predicción",
                    xaxis_title="Género",
                    yaxis_title="valores_categoricas",
                    font=dict(size=12),
                    width=500,
                    height=500
                )

                st.plotly_chart(fig)

            
            with col_gra2:
                
                fig = go.Figure()
                
                fig.add_trace(go.Pie(
                    labels=valores_categoricas.index,
                    values=valores_categoricas.values,
                    textinfo='label+percent',
                    insidetextorientation='radial',
                    hovertemplate='%{label}: <br>valores_categoricas: %{value} <br>Porcentaje: %{percent}',
                    showlegend=True,
                    marker=dict(colors=colores)
                    
                ))

                fig.update_layout(
                    title=f"Gráfico Circular - Predicción",
                    font=dict(size=15),
                    width=500,
                    height=500
                )

                st.plotly_chart(fig)
            

            

            st.divider()
            st.markdown("### Descargar el Archivo Predecido en Diferentes Formatos")        
            st.download_button(
                label=":file_folder: Descargar El Archivo Excel",
                data=to_excel(df_unido),
                file_name='Reporte.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            st.download_button(
                label=":file_folder: Descargar El Archivo CSV",
                data=csv_data,
                file_name="reporte.csv",
                mime="text/csv"
                )
        else:
            col_nada_predi,col_nada_pro = st.columns((5,5))
            with col_nada_predi:
                st.subheader('Predicción')
                df_abandono = pd.DataFrame(prediccion_modelo,columns=["Abandono"])
                df_abandono = df_abandono.applymap(lambda x: "No" if x == 0 else "Si")
                st.write(df_abandono)
            with col_nada_pro:  
                st.subheader('Probabilidad de predicción')
                df_abandono = pd.DataFrame(prediction_proba_modelo.argmax(axis=1), columns=["Abandono"])
                df_abandono = df_abandono.applymap(lambda x: "No" if x == 0 else "Si")
                probabilidades = np.where(df_abandono["Abandono"] == "No", prediction_proba_modelo[:, 0], prediction_proba_modelo[:, 1])
                df_resultado = pd.DataFrame({"Abandono": df_abandono["Abandono"], "Probabilidad": probabilidades})
                st.write(df_resultado)
            for index,row in df_resultado.iterrows():
                abandono = row.iloc[0]
                probabilidad = row.iloc[1] * 100
                if abandono == "Si":
                    st.markdown(f"### La persona tiene una probabilidad del {probabilidad:.2f}% de que abandone los servicios.")
                else:
                    st.markdown(f"### La persona tiene una probabilidad del {probabilidad:.2f}% de que NO abandone los servicios.")                 
        
    else:
        
        carga_modelo = pickle.load(open('modelo_entrenado.pkl', 'rb'))
        if datos_nuevos is not None:
            column = ["PersonaMayor","Socio","Dependientes","Permanencia","ServicioTelefonico","VariasLineas","ServicioInternet",
                      "SeguridadLinea","CopiaSeguridadLinea","ProteccionDispositivo","ServicioTecnico","ServicioTV","ServicioPeliculas",
                      "Contrato","FacturacionElectronica","MetodoPago","RecargoMensual","TotalRecargo"]
            dataset_ingresado = pd.read_csv(datos_nuevos)
            if st.checkbox("Mostrar Datos Ingresados"):
                st.write(dataset_ingresado)
            columnas_comunes = dataset_ingresado.columns.intersection(column)
            dataset_nuevo = dataset_ingresado[columnas_comunes]
            st.divider()
            st.markdown("### Datos a Predecir")
            if st.checkbox("Mostrar Datos que ingresan a la Predicción"):
                st.write(dataset_nuevo)
            
            
            
        else:
            column = ["PersonaMayor","Socio","Dependientes","Permanencia","ServicioTelefonico","VariasLineas","ServicioInternet",
                      "SeguridadLinea","CopiaSeguridadLinea","ProteccionDispositivo","ServicioTecnico","ServicioTV","ServicioPeliculas",
                      "Contrato","FacturacionElectronica","MetodoPago","RecargoMensual","TotalRecargo"]
            datos_dicc = {}
            if "Genero" in column:
                genero = st.sidebar.selectbox('Genero',("Femenino","Masculino"))
                datos_dicc['Genero']= genero
            if "PersonaMayor" in column:
                PersonaMayor = st.sidebar.selectbox('¿Es Una Persona Adulta Mayor?(Si =1,No=0)', (0,1))
                datos_dicc["PersonaMayor"]=PersonaMayor
            if "Socio" in column:
                Socio = st.sidebar.selectbox('¿Eres socio?', ("Si","No"))
                datos_dicc["Socio"]=Socio
            if "Dependientes" in column:
                Dependientes = st.sidebar.selectbox('¿Eres Dependiente?',("Si","No"))
                datos_dicc["Dependientes"]=Dependientes
            if "Permanencia" in column:
                Permanencia = st.sidebar.slider('¿Cuantos Meses tienes de Contrato?',0,72,29)
                datos_dicc["Permanencia"]=Permanencia
            if "ServicioTelefonico" in column:
                ServicioTelefonico = st.sidebar.selectbox('¿Tienes Servicio Telefónico?', ("Si","No"))
                datos_dicc["ServicioTelefonico"]=ServicioTelefonico
            if "VariasLineas" in column:
                VariasLineas = st.sidebar.selectbox('¿Tiene Múlitples Líneas?', ("Si","No","Sin Servicio Telefónico"))
                datos_dicc["VariasLineas"]=VariasLineas
            if "ServicioInternet" in column:
                ServicioInternet = st.sidebar.selectbox('¿Que tipo de servicio de Internet Tiene?', ("DLS","No","Fibra Óptica"))
                datos_dicc["ServicioInternet"]=ServicioInternet
            if "SeguridadLinea" in column:
                SeguridadLinea = st.sidebar.selectbox('¿Tiene Seguridad En Línea?', ("Si","No","Sin Servicio de Internet"))
                datos_dicc["SeguridadLinea"]=SeguridadLinea
            if "CopiaSeguridadLinea" in column:
                CopiaSeguridadLinea = st.sidebar.selectbox('Tiene Copia de Seguridad en Línea ?', ("Si","No","Sin Servicio de Internet"))
                datos_dicc["CopiaSeguridadLinea"]=CopiaSeguridadLinea
            if "ProteccionDispositivo" in column:
                ProteccionDispositivo = st.sidebar.selectbox('¿Tiene Protección del Dispositivo?', ("Si","No","Sin Servicio de Internet"))
                datos_dicc["ProteccionDispositivo"]=ProteccionDispositivo
            if "ServicioTecnico" in column:
                ServicioTecnico = st.sidebar.selectbox('¿Tiene Soporte Técnico?', ("Si","No","Sin Servicio de Internet"))
                datos_dicc["ServicioTecnico"]=ServicioTecnico
            if "ServicioTV" in column:
                ServicioTV = st.sidebar.selectbox('¿Tiene Servicio de TV?', ("Si","No","Sin Servicio de Internet"))
                datos_dicc["ServicioTV"]=ServicioTV
            if "ServicioPeliculas" in column:
                ServicioPeliculas = st.sidebar.selectbox('¿Tiene Servicio de Películas?', ("Si","No","Sin Servicio de Internet"))
                datos_dicc["ServicioPeliculas"]=ServicioPeliculas
            if "Contrato" in column:
                Contrato = st.sidebar.selectbox('Tipo de Contrato del Cliente', ("Mensual","un anio","dos anios"))
                datos_dicc["Contrato"]=Contrato
            if "FacturacionElectronica" in column:
                FacturacionElectronica = st.sidebar.selectbox('¿Recibe Factura Electrónica?', ("Si","No"))
                datos_dicc["FacturacionElectronica"]=FacturacionElectronica
            if "MetodoPago" in column:
                MetodoPago = st.sidebar.selectbox('¿Cuál es el Metodo de Pago?', ("Cheque Electrónico","Cheque por Correo","Transferencia bancaria (automática)","Tarjeta de crédito (automática)"))
                datos_dicc["MetodoPago"]=MetodoPago
            if "RecargoMensual" in column:
                RecargoMensual = st.sidebar.number_input('Recargo Mensual',0.00,200.00,70.35)
                datos_dicc["RecargoMensual"]=RecargoMensual
            if "TotalRecargo" in column:
                TotalRecargo = st.sidebar.number_input('Recargo Anual',0.00,10000.00,1000.00)
                datos_dicc["TotalRecargo"]=TotalRecargo

            dataset_nuevo = pd.DataFrame(datos_dicc, index=[0])
            st.write('A la espera de que se cargue el archivo CSV. Actualmente usando parámetros de entrada (que se muestran a continuación).')
            st.write(dataset_nuevo)              
        
        for i in dataset_nuevo.select_dtypes(include='object').columns:
            dataset_nuevo[i] = LabelEncoder().fit_transform(dataset_nuevo[i])
        scaler = StandardScaler().fit(dataset_nuevo[["TotalRecargo"]])
        dataset_nuevo["TotalRecargo"] = scaler.transform(dataset_nuevo[["TotalRecargo"]])
        scaler = StandardScaler().fit(dataset_nuevo[["RecargoMensual"]])
        dataset_nuevo["RecargoMensual"] = scaler.transform(dataset_nuevo[["RecargoMensual"]])
        
        
        prediccion_modelo = carga_modelo.predict(dataset_nuevo)
        prediction_proba_modelo = carga_modelo.predict_proba(dataset_nuevo)
        
        if datos_nuevos is not None:
            df_abandono = pd.DataFrame(prediccion_modelo,columns=["Abandono"])
            df_abandono = df_abandono.applymap(lambda x: "No" if x == 0 else "Si")    
            df_abandono = pd.DataFrame(prediction_proba_modelo.argmax(axis=1), columns=["Abandono"])
            df_abandono = df_abandono.applymap(lambda x: "No" if x == 0 else "Si")
            probabilidades = np.where(df_abandono["Abandono"] == "No", prediction_proba_modelo[:, 0], prediction_proba_modelo[:, 1])

            df_resultado = pd.DataFrame({"Abandono": df_abandono["Abandono"], "Probabilidad": probabilidades})
            df_unido = pd.concat([dataset_ingresado,df_resultado],axis=1)
            csv_data = df_unido.to_csv(index=False)
            st.divider()
            st.markdown("### Datos con la Predicción")
            mostrar_prediccion = st.checkbox("Mostrar Predicción Final")
            if mostrar_prediccion:
                st.write(df_unido)
            st.divider()
            st.markdown("### Gráficos de la predicción")
            col_gra1, col_gra2 = st.columns((5,5))
            valores_categoricas = df_unido["Abandono"].value_counts()
            colorscale = px.colors.sequential.YlOrBr
            num_categorias = len(valores_categoricas.index)
            step_size = int(len(colorscale) / num_categorias)
            colores = colorscale[::step_size]
            
            
            with col_gra1:

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=valores_categoricas.index,
                    y=valores_categoricas,
                    text=valores_categoricas,
                    textposition='auto',
                    hovertemplate='%{x}: <br>valores_categoricas: %{y}',
                    marker=dict(color=colores)

                ))

                fig.update_layout(
                    title=f"Gráfico de Barras - Predicción",
                    xaxis_title="Género",
                    yaxis_title="valores_categoricas",
                    font=dict(size=12),
                    width=500,
                    height=500
                )

                st.plotly_chart(fig)

            
            with col_gra2:
                
                fig = go.Figure()
                
                fig.add_trace(go.Pie(
                    labels=valores_categoricas.index,
                    values=valores_categoricas.values,
                    textinfo='label+percent',
                    insidetextorientation='radial',
                    hovertemplate='%{label}: <br>valores_categoricas: %{value} <br>Porcentaje: %{percent}',
                    showlegend=True,
                    marker=dict(colors=colores)
                    
                ))

                fig.update_layout(
                    title=f"Gráfico Circular - Predicción",
                    font=dict(size=15),
                    width=500,
                    height=500
                )

                st.plotly_chart(fig)
            

            

            st.divider()
            st.markdown("### Descargar el Archivo Predecido en Diferentes Formatos")        
            st.download_button(
                label=":file_folder: Descargar El Archivo Excel",
                data=to_excel(df_unido),
                file_name='Reporte.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            st.download_button(
                label=":file_folder: Descargar El Archivo CSV",
                data=csv_data,
                file_name="reporte.csv",
                mime="text/csv"
                )
                
        else:
            col_nada_predi,col_nada_pro = st.columns((5,5))
            with col_nada_predi:
                st.subheader('Predicción')
                df_abandono = pd.DataFrame(prediccion_modelo,columns=["Abandono"])
                df_abandono = df_abandono.applymap(lambda x: "No" if x == 0 else "Si")
                st.write(df_abandono)
            with col_nada_pro:  
                st.subheader('Probabilidad de predicción')
                df_abandono = pd.DataFrame(prediction_proba_modelo.argmax(axis=1), columns=["Abandono"])
                df_abandono = df_abandono.applymap(lambda x: "No" if x == 0 else "Si")
                probabilidades = np.where(df_abandono["Abandono"] == "No", prediction_proba_modelo[:, 0], prediction_proba_modelo[:, 1])
                df_resultado = pd.DataFrame({"Abandono": df_abandono["Abandono"], "Probabilidad": probabilidades})
                st.write(df_resultado)
            for index,row in df_resultado.iterrows():
                abandono = row.iloc[0]
                probabilidad = row.iloc[1] * 100
                if abandono == "Si":
                    st.markdown(f"### La persona tiene una probabilidad del {probabilidad:.2f}% de que abandone los servicios.")
                else:
                    st.markdown(f"### La persona tiene una probabilidad del {probabilidad:.2f}% de que NO abandone los servicios.")
        
        