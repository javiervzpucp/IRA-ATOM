# -*- coding: utf-8 -*-
"""
App de Streamlit para Conversión Archivística
"""

import streamlit as st
import pandas as pd
import os
import io
import re
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema import HumanMessage
import warnings

# Cargar variables de entorno
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suprimir advertencias de depreciación
warnings.simplefilter("ignore", category=FutureWarning)

# Clase integrada para conversión
class ISADConverter:
    def __init__(self):
        self.column_mapping = {
            'signatura': 'referenceCode',
            'fechacronica': 'date',
            'institucion': 'title',
            'categoria': 'scopeAndContent',
            'pais': 'country',
            'observaciones': 'physicalDescription'
        }
        
        # Configurar Mixtral
        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            huggingfacehub_api_token=HF_API_TOKEN,
            temperature=0.3,
            max_new_tokens=150
        )

    def _normalize_column_names(self, df):
        df.columns = [col.strip().lower().replace(' ', '') for col in df.columns]
        return df.rename(columns=self.column_mapping)

    def _normalize_date_with_ai(self, date_str):
        """Usa Mixtral para fechas complejas"""
        prompt = f"""Convierte esta fecha a formato ISO 8601:
        Fecha original: {date_str}
        Formato ISO: """
        
        try:
            response = self.llm.invoke(prompt).strip()
            if re.match(r'\d{4}-\d{2}-\d{2}', response):
                return response
        except Exception as e:
            logger.warning(f"Error AI: {str(e)}")
        return None

    def _normalize_date(self, date_str):
        try:
            date_str = re.sub(r'\.-', '-', str(date_str).lower())
            
            # Intentar parseo automático
            if re.match(r'^\d{4}$', date_str):
                return f"{date_str}-01-01"
            
            parsed_date = datetime.strptime(date_str, "%Y-%b-%d")
            return parsed_date.strftime("%Y-%m-%d")
        
        except:
            # Fallback con IA
            ai_date = self._normalize_date_with_ai(date_str)
            return ai_date if ai_date else datetime.now().strftime("%Y-%m-%d")

    def _generate_title_with_ai(self, row):
        """Mejora títulos usando contexto"""
        prompt = f"""Genera un título archivístico formal en español usando:
        - Institución: {row.get('institucion','')}
        - Categoría: {row.get('categoria','')}
        - País: {row.get('pais','')}
        Título: """
        
        try:
            return self.llm.invoke(prompt).strip().replace('"','')
        except:
            return row.get('institucion', 'Documento sin título')

    def process(self, input_file, output_base):
        try:
            df = pd.read_csv(input_file, skiprows=1, dtype=str, header=0)
            df = self._normalize_column_names(df)
            
            df['title'] = df.apply(
                lambda x: self._generate_title_with_ai(x) if pd.isna(x.get('title')) else x['title'], 
                axis=1
            )
            
            df['date'] = df['date'].apply(self._normalize_date)
            df['levelOfDescription'] = 'Documento'
            
            os.makedirs(os.path.dirname(output_base), exist_ok=True)
            df.to_csv(f"{output_base}.csv", index=False, encoding='utf-8-sig')
            df.to_excel(f"{output_base}.xlsx", index=False, engine='openpyxl')
            
            logger.info(f"Procesamiento completado: {len(df)} registros")
            return True
        
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return False

# Configuración inicial de la app
st.set_page_config(page_title="Conversor ISAD(G)", page_icon="📚", layout="wide")

st.title("🖋️ Conversor de Documentos Archivísticos a ISAD(G)")
st.markdown("---")

with st.sidebar:
    st.header("Configuración")
    if HF_API_TOKEN:
        st.success("Token de Hugging Face cargado correctamente desde el entorno.")
    else:
        st.error("No se encontró el token de Hugging Face en el entorno. Verifica tu configuración.")
    st.markdown("---")
    st.info("Sube tu archivo CSV y descarga los formatos ISAD(G) listos para importar.")

uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file:
    converter = ISADConverter()
    with st.spinner("Procesando archivo..."):
        try:
            temp_dir = "temp_results"
            os.makedirs(temp_dir, exist_ok=True)
            output_base = os.path.join(temp_dir, "resultado_isad")
            
            with open(os.path.join(temp_dir, "temp_input.csv"), "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if converter.process(os.path.join(temp_dir, "temp_input.csv"), output_base):
                result_csv = pd.read_csv(f"{output_base}.csv")
                result_excel = pd.read_excel(f"{output_base}.xlsx")
                
                st.subheader("Vista Previa de los Datos Convertidos")
                st.dataframe(result_csv.head(), use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    csv_buffer = io.StringIO()
                    result_csv.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                    st.download_button("Descargar CSV ISAD(G)", data=csv_buffer.getvalue(), file_name="resultado_isad.csv", mime="text/csv")
                with col2:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        result_excel.to_excel(writer, index=False)
                    st.download_button("Descargar Excel ISAD(G)", data=excel_buffer.getvalue(), file_name="resultado_isad.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"Error crítico: {str(e)}")
            st.stop()
else:
    st.markdown("""
    ### Instrucciones de Uso:
    1. **Sube tu archivo CSV** con los documentos archivísticos
    2. Espera a que se complete el procesamiento (¡usamos IA para mejorar los metadatos!)
    3. **Descarga los resultados** en formato CSV o Excel listos para importar
    4. 🚀 ¡Listo para preservar tu patrimonio documental!
    """)
    st.markdown("---")
    st.caption("v1.0 - Herramienta desarrollada para el Archivo Histórico Riva-Agüero PUCP")
