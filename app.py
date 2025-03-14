# -*- coding: utf-8 -*-
"""
App de Streamlit para Conversión Archivística
"""

import streamlit as st
import pandas as pd
import os
import io
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(__file__))

from ira_atom_v2 import ISADConverter  # Asumiendo que el código está en isad_converter.py

# Cargar variables de entorno
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Configuración inicial
st.set_page_config(page_title="Conversor ISAD(G)", page_icon="📚", layout="wide")

# Título de la aplicación
st.title("🖋️ Conversor de Documentos Archivísticos a ISAD(G)")
st.markdown("---")
st.write("🔍 Depuración: La aplicación ha cargado hasta aquí correctamente.")

# Sidebar para configuración
with st.sidebar:
    st.header("Configuración")
    if HF_API_TOKEN:
        st.success("Token de Hugging Face cargado correctamente desde el entorno.")
    else:
        st.error("No se encontró el token de Hugging Face en el entorno. Verifica tu configuración.")
    st.markdown("---")
    st.info("Sube tu archivo CSV y descarga los formatos ISAD(G) listos para importar.")

# Componente principal
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file:
    converter = ISADConverter()
    
    # Procesamiento de datos
    with st.spinner("Procesando archivo..."):
        try:
            # Crear directorio temporal
            temp_dir = "temp_results"
            os.makedirs(temp_dir, exist_ok=True)
            output_base = os.path.join(temp_dir, "resultado_isad")
            
            # Guardar archivo temporal
            with open(os.path.join(temp_dir, "temp_input.csv"), "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Ejecutar conversión
            if converter.process(os.path.join(temp_dir, "temp_input.csv"), output_base):
                # Leer resultados
                result_csv = pd.read_csv(f"{output_base}.csv")
                result_excel = pd.read_excel(f"{output_base}.xlsx")
                
                # Mostrar vista previa
                st.subheader("Vista Previa de los Datos Convertidos")
                st.dataframe(result_csv.head(), use_container_width=True)
                
                # Botones de descarga
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_buffer = io.StringIO()
                    result_csv.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="Descargar CSV ISAD(G)",
                        data=csv_buffer.getvalue(),
                        file_name="resultado_isad.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        result_excel.to_excel(writer, index=False)
                    st.download_button(
                        label="Descargar Excel ISAD(G)",
                        data=excel_buffer.getvalue(),
                        file_name="resultado_isad.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Limpieza de archivos temporales
                os.remove(f"{output_base}.csv")
                os.remove(f"{output_base}.xlsx")
                os.remove(os.path.join(temp_dir, "temp_input.csv"))
                
            else:
                st.error("Error en el procesamiento. Verifica el formato del archivo.")
                
        except Exception as e:
            st.error(f"Error crítico: {str(e)}")
            st.stop()

# Mensaje de instrucciones
else:
    st.markdown("""
    ### Instrucciones de Uso:
    1. **Sube tu archivo CSV** con los documentos archivísticos
    2. Espera a que se complete el procesamiento (¡usamos IA para mejorar los metadatos!)
    3. **Descarga los resultados** en formato CSV o Excel listos para importar
    4. 🚀 ¡Listo para preservar tu patrimonio documental!
    
    ---
    **Formato CSV Requerido:**
    - Debe contener columnas: `Signatura`, `Fecha cronica`, `Institucion`, etc.
    - Codificación: UTF-8
    - Separador: Comas
    """)

# Notas técnicas
st.markdown("---")
st.caption("v1.0 - Herramienta desarrollada para el Archivo Histórico Riva-Agüero PUCP")
