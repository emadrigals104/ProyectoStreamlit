def preprocesamiento():
    import streamlit as st
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.impute import KNNImputer
    from sklearn.ensemble import IsolationForest
    

    st.markdown("# Preprocesamiento de Datos")


    if "df" not in st.session_state:
        st.image("images/upload-cloud-data.png", width=300)
        st.write("Debe Ingresar el dataset primero, Dirijase a la pagina principal.") 
    else:
        if st.session_state.get("original_df") is None:
            st.session_state.original_df = st.session_state.df.copy()
        st.markdown("### Si desea restablecer el dataset aplaste el botón de abajo")
        reset_button = st.button("Restablecer Preprocesado")
        if reset_button:
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.predf = st.session_state.df.copy()
            st.write("El dataset ha sido restablecido.")

        if "predf" not in st.session_state:
            st.session_state.predf = st.session_state.df.copy()
        st.session_state.data = st.session_state.predf
        
        st.divider()
        col_nulo,col_trans = st.columns(2)
        with col_nulo:
            st.markdown("### Eliminar Columnas")
            col_eli,col_bo_eli = st.columns(2)
            with col_eli:              
                st.session_state.seleccion_nulo = st.multiselect("Escoja las variables a eliminar:", st.session_state.data.columns)
            with col_bo_eli:
                st.write("")
                st.write("")
                st.session_state.boton_eliminar = st.button("Eliminar")
            if st.session_state.boton_eliminar:
                st.session_state.data = st.session_state.data.drop(st.session_state.seleccion_nulo, axis=1)
                st.session_state.predf = st.session_state.predf.drop(st.session_state.seleccion_nulo, axis=1)
                st.write(st.session_state.data)
                
        with col_trans:
            st.markdown("### Transformar Valores Vacios Categóricos")
            col_selec, col_boton = st.columns(2)
            with col_selec:
                seleccion_cate = st.multiselect("Escoja la(s) variable(s):", st.session_state.data.select_dtypes(include='object').columns)
                seleccion_metodo_cate = st.selectbox("Elija la técnica para transformar",["SimpleImputer"])
                seleccion_tecnica_cate = st.selectbox("Elija el método",("most_frequent","constant"))
                if seleccion_tecnica_cate == "constant":
                    fill_value = st.text_input("Con qué valor desea reemplazar")
                else:
                    fill_value = None
            with col_boton:
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                boton_quitarcate = st.button("Reemplazar")
                if boton_quitarcate:
                    if seleccion_metodo_cate == "SimpleImputer":
                        categorico = SimpleImputer(strategy=seleccion_tecnica_cate,fill_value=fill_value)
                        categorico.fit_transform(st.session_state.data[seleccion_cate])
                        st.write("Valores nulos reemplazados")                 
            
        st.divider()
        col_label,col_standar=st.columns(2)
        with col_label:
            st.markdown("### Transformar Valores Vacios Numéricos")
            col_selec, col_boton = st.columns(2)
            with col_selec:
                seleccion_nume = st.multiselect("Escoja la(s) variable(s):", st.session_state.data.select_dtypes(include=['int',"float"]).columns)
                seleccion_metodo_nume = st.selectbox("Elija la técnica para transformar",("SimpleImputer","KNNImputer","IterativeImputer"))
                if seleccion_metodo_nume =="SimpleImputer":
                    seleccion_tecnica_nume = st.selectbox("Elija el método",("mean","median"))
                elif seleccion_metodo_nume == "KNNImputer":
                    seleccion_knn = st.slider("Escoja el número de muestras",0,100,5,key="seleccion_knn")
                    
                
            with col_boton:
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                boton_quitarnume = st.button("Reemplazar Nulos")
                if boton_quitarnume:
                    if seleccion_metodo_nume =="SimpleImputer":                         
                        categorico = SimpleImputer(strategy=seleccion_tecnica_nume)
                        categorico.fit_transform(st.session_state.data[seleccion_nume])
                        st.write("Valores nulos reemplazados")
                    elif seleccion_metodo_nume == "KNNImputer":
                        categorico = KNNImputer(n_neighbors=seleccion_knn)
                        categorico.fit_transform(st.session_state.data[seleccion_nume])
                        st.write("Valores nulos reemplazados")    
                           
        with col_standar:
            st.markdown("### Eliminar valores Atípicos") 
            col_eliminar,col_boton_eliminar = st.columns(2)    
            with col_eliminar:
                variables_numericas = st.multiselect("Elija las variables numéricas", options=st.session_state.data.select_dtypes(include=['float', 'int']).columns)
                selector_atipico = st.selectbox("Elija el metodo",("Boxplot","Insolation Forest"))
                if selector_atipico == "Boxplot":
                    def eliminar_valores_atipicos_iqr(df, columnas, factor_iqr=1.5):
                        for columna in columnas:
                            Q1 = df[columna].quantile(0.25)
                            Q3 = df[columna].quantile(0.75)
                            IQR = Q3 - Q1
                            limite_inferior = Q1 - factor_iqr * IQR
                            limite_superior = Q3 + factor_iqr * IQR
                            df = df[(df[columna] >= limite_inferior) & (df[columna] <= limite_superior)]
                        return df
                    with col_boton_eliminar:
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        st.write("")
                        if st.button("Eliminar valores atípicos"):
                            st.session_state.data = eliminar_valores_atipicos_iqr(st.session_state.data, variables_numericas)
                            st.write("Valores atípicos eliminados.")
                            st.write(st.session_state.data)
                else:
                    if st.button("Eliminar valores atípicos"):
                        modelo_isolation_forest = IsolationForest(random_state=0)
                        st.session_state.data = st.session_state.data[variables_numericas]
                        modelo_isolation_forest.fit(st.session_state.data)
                        st.write("Valores atípicos eliminados.")
                        st.write(st.session_state.data)
                        
            st.markdown("### Eliminar Valores Vacíos")
            if st.button("Eliminar filas vacías"):
                st.session_state.data = st.session_state.data.dropna()
                st.write(st.session_state.data)
                
        st.divider()
        st.markdown('''
                    # Proceso Interno
                    Existe un proceso interno que realiza la conversión de las columnas categóricas y algunas configuraciones más que se deben realizar.
                    
                    Para que se realicen de manera automática debe pulsar el botón de aquí abajo
                    ''')
        st.session_state.boton_pasos_no_realizados = st.button("Pulse aquí cuando haya terminado")
        if st.session_state.boton_pasos_no_realizados:
            for i in st.session_state.data.select_dtypes(include='object').columns:
                st.session_state.data[i] = LabelEncoder().fit_transform(st.session_state.data[i])
                    
            scaler = StandardScaler().fit(st.session_state.data[["TotalRecargo"]])
            st.session_state.data["TotalRecargo"] = scaler.transform(st.session_state.data[["TotalRecargo"]])
            scaler = StandardScaler().fit(st.session_state.data[["RecargoMensual"]])
            st.session_state.data["RecargoMensual"] = scaler.transform(st.session_state.data[["RecargoMensual"]])
 
            
        
        
        
                
                

                