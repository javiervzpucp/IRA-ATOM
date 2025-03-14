# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 18:47:46 2025

@author: jveraz
"""

# -*- coding: utf-8 -*-
"""
Conversor CSV a ISAD(G) con Mixtral - Versión Mejorada
"""

import pandas as pd
import logging
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema import HumanMessage
import warnings

# Suprimir advertencias de depreciación
warnings.simplefilter("ignore", category=FutureWarning)

# Cargar variables de entorno
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
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
            # Cargar y procesar datos base
            df = pd.read_csv(input_file, skiprows=1, dtype=str, header=0)
            df = self._normalize_column_names(df)
            
            # Mejorar títulos con IA
            df['title'] = df.apply(
                lambda x: self._generate_title_with_ai(x) if pd.isna(x.get('title')) else x['title'], 
                axis=1
            )
            
            # Procesar fechas
            df['date'] = df['date'].apply(self._normalize_date)
            
            # Campos obligatorios
            df['levelOfDescription'] = 'Documento'
            
            # Guardar resultados
            os.makedirs(os.path.dirname(output_base), exist_ok=True)
            df.to_csv(f"{output_base}.csv", index=False, encoding='utf-8-sig')
            df.to_excel(f"{output_base}.xlsx", index=False, engine='openpyxl')
            
            logger.info(f"Procesamiento completado: {len(df)} registros")
            return True
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return False

#if __name__ == "__main__":
#    converter = ISADConverter()
#    if converter.process("Diplomas.csv", "resultados/salida_ia"):
#        print("✅ Transformación exitosa con mejoras de IA")
#    else:
#        print("❌ Error en el proceso")