import math

def sturges_rule(data):
    n = len(data)
    k = 1 + math.log2(n)
    return int(k)


def eda():
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from io import StringIO
   
    st.markdown("# :bar_chart: Análisis Explotario de Datos")
    st.write("""
        En este análisis exploratorio de datos, nos enfocaremos en 
        comprender las variables y patrones subyacentes que 
        afectan el abandono de los clientes. 
    """)
    st.divider()

    if "df" not in st.session_state:
        st.image("images/upload-cloud-data.png", width=300)
        st.write("Debe Ingresar el dataset primero, Dirijase a la pagina principal.") 
    else:
        data = st.session_state.df
        seleccion_grafica_cate = st.sidebar.selectbox('Selecciona una Variable Categórica', list(data.select_dtypes(include='object').columns))
        seleccion_grafica_nume = st.sidebar.selectbox('Selecciona una Variable Numérica', list(data.select_dtypes(exclude='object').columns))
        
        
        
        # Primera fila
        st.markdown('## Metricas de los Datos')
        col1, col2, col3, col4,col5 = st.columns(5)
        col1.metric("Número de Filas", data.shape[0])
        col2.metric("Número de Columnas", data.shape[1])
        col3.metric("Datos Duplicados", data.duplicated().sum())
        col4.metric("Variables Categóricas",data.select_dtypes(include='object').shape[1])
        col5.metric("Variables Numéricas",data.select_dtypes(exclude='object').shape[1])

        st.divider()
        st.markdown('## Gráficos de los Datos')
        c1, c2 = st.columns((5,5))
        valores_categoricas = data[seleccion_grafica_cate].value_counts()
        valores_numericas = data[seleccion_grafica_nume]
        colorscale = px.colors.sequential.YlOrBr
        num_categorias = len(valores_categoricas.index)
        step_size = int(len(colorscale) / num_categorias)
        colores = colorscale[::step_size]
        
        
        with c1:
            import plotly.graph_objects as go
            import streamlit as st

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
                title=f"Gráfico de Barras - {seleccion_grafica_cate}",
                xaxis_title="Género",
                yaxis_title="valores_categoricas",
                font=dict(size=12),
                width=500,
                height=500
            )

            st.plotly_chart(fig)

        
        with c2:
            
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
                title=f"Gráfico Circular - {seleccion_grafica_cate}",
                font=dict(size=15),
                width=500,
                height=500
            )

            st.plotly_chart(fig)
            
            


        import plotly.graph_objects as go
        import streamlit as st
        st.divider()
        gra1,gra2 = st.columns((5,5))
        fig = go.Figure()
        with gra1:
            fig_box = go.Figure()

            for variable, color in zip(data[seleccion_grafica_cate].unique(), colores):
                fig_box.add_trace(go.Box(
                    x=data[seleccion_grafica_cate][data[seleccion_grafica_cate] == variable],
                    y=data[seleccion_grafica_nume][data[seleccion_grafica_cate] == variable],
                    name=variable,
                    marker=dict(color=color),
                    hovertemplate='%{x}: %{y}'
                ))

            fig_box.update_layout(
                title=f"Gráfico Boxplot - {seleccion_grafica_cate}",
                xaxis_title=seleccion_grafica_cate,
                yaxis_title=seleccion_grafica_nume,
                font=dict(size=12),
                width=500,
                height=500
            )

            st.plotly_chart(fig_box)
        import plotly.graph_objects as go
        import streamlit as st
        with gra2:
            fig_hist = go.Figure()
            k = sturges_rule(valores_numericas)
            fig_hist.add_trace(go.Histogram(
                x=valores_numericas,
                nbinsx=k,
                marker=dict(color=colores[0]),
                hovertemplate='Edad: %{x}<br>valores_categoricas: %{y}'
            ))

            fig_hist.update_layout(
                title=f"Histograma - {seleccion_grafica_cate}",
                xaxis_title=seleccion_grafica_cate,
                yaxis_title=seleccion_grafica_nume,
                font=dict(size=12),
                width=500,
                height=500
            )

            st.plotly_chart(fig_hist)
        gra3,gra4 = st.columns((5,5))
        with gra3:
            import plotly.figure_factory as ff
            data_corr = data.corr(numeric_only=True)
            z = data_corr.values.round(2)
            x = data_corr.columns.tolist()
            y = data_corr.index.tolist()
            fig_heatmap = go.Figure()
            fig_heatmap = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='YlOrBr', annotation_text=z)
            fig_heatmap.update_layout(
                title=f"Mapa de Calor - Matriz de correlación",
                font=dict(size=15),
                width=500,
                height=600
            )
            st.plotly_chart(fig_heatmap)
            
            
        with gra4: 
            fig = go.Figure()
            
            colorscale = px.colors.sequential.YlOrBr
            num_categorias = len(data[seleccion_grafica_cate].unique())
            step_size = int(len(colorscale) / num_categorias)
            colores = colorscale[::step_size]

            for i, category in enumerate(data[seleccion_grafica_cate].unique()):
                fig.add_trace(go.Violin(y=data[data[seleccion_grafica_cate] == category][seleccion_grafica_nume],
                                        x=[category] * len(data[data[seleccion_grafica_cate] == category]),
                                        name=category,
                                        box_visible=True,
                                        meanline_visible=True,
                                        hovertemplate=f'{seleccion_grafica_cate}: %{x}<br>{seleccion_grafica_nume}: %{y}',
                                        line_color=colores[i % len(colores)]))

            fig.update_layout(
                title=f"Gráfico de violín - {seleccion_grafica_cate}",
                xaxis_title=seleccion_grafica_cate,
                yaxis_title=seleccion_grafica_nume,
                font=dict(size=12),
                width=500,
                height=600
            )

            st.plotly_chart(fig)
        binary_df = data.isnull().astype(int)

        # Crear un heatmap con Plotly
        fig = go.Figure()

        fig.add_trace(go.Heatmap(z=binary_df.values,
                                x=binary_df.columns,
                                y=binary_df.index,
                                colorscale='YlOrBr',
                                showscale=False))

        fig.update_layout(
            title="Datos nulos en los Datos",
            xaxis_title="Columnas",
            yaxis_title="Índice",
            font=dict(size=12),
            width=1100,
            height=600
        )

        st.plotly_chart(fig)
        
        st.divider()
        st.markdown('## Información sobre los Datos')
        col_resu1, col_resu2 = st.columns(2)
        with col_resu1:
            st.markdown("### Resumen Conciso de los Datos")
            if st.checkbox("Mostrar Resumen"):
                info = StringIO()
                data.info(buf=info)
                st.text(str(info.getvalue()))
        
        with col_resu2:
            st.markdown("### Datos Nulos por Columnas")
            if st.checkbox("Mostrar Datos Nulos"):
                st.write(data.isnull().sum().sort_values(ascending=False))
                
        st.divider()       
        col_resu3, col_resu4 = st.columns((2))
        with col_resu3:
            st.markdown("### Datos Únicos por Columnas")
            if st.checkbox("Mostrar Datos Únicos"):
                st.write(data.nunique())
        with col_resu4:
            st.markdown("### Estadística Descriptiva de los Datos")
            if st.checkbox("Mostrar Estadística"):
                st.write(data.describe().round(2))
                
        st.divider()
        col_resu5,col_resu6 = st.columns(2)
        with col_resu5:
            st.markdown("### Correlación Entre Variables Numéricas")
            if st.checkbox("Mostrar Correlación"):  
                st.write(data.corr(numeric_only=True))


        


