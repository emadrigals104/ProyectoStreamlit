import streamlit as st
import pandas as pd


def reporte():
    st.markdown("# :bar_chart: Reporte de los Datos")
    st.markdown('''
    En esta sección, presentamos un reporte detallado y visual creado en Power BI. Este reporte proporciona información valiosa y una visión general del análisis de los datos.

    La integración de Power BI con Streamlit permite a los usuarios aprovechar las capacidades de ambas herramientas, ofreciendo visualizaciones dinámicas y una experiencia de usuario mejorada.

    A continuación, encontrarás el reporte de Power BI que hemos preparado para este análisis. Puedes interactuar con las visualizaciones y explorar diferentes aspectos de los datos. Si deseas obtener más información sobre alguna visualización específica, simplemente coloca el cursor sobre ella y aparecerá información adicional.

    Para obtener los mejores resultados, te recomendamos visualizar este reporte en pantalla completa. Para hacerlo, haz clic en el icono de pantalla completa en la esquina inferior derecha del reporte.

    ¡Disfruta explorando el reporte y descubriendo información útil para tu análisis!''')
    st.markdown('''
                <iframe title="Transporte y Logistica - Instructor" width="1100" height="673.5" src="https://app.powerbi.com/view?r=eyJrIjoiYzJkOWRiYzctMDE4NS00NTcyLTkwMGYtMDQ0NzkwMTZiZmVmIiwidCI6ImQyZGZkYjc0LWZiNDItNGMzYi04ZGFkLTZiNzg1NzlmMmM4ZCIsImMiOjR9" frameborder="0" allowFullScreen="true"></iframe>
                ''', unsafe_allow_html=True)


