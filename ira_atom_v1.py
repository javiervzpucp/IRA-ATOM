# -*- coding: utf-8 -*-
"""
Conversor CSV a ISAD(G) - Versión Final Estable
Corrección de error de sintaxis y salida limpia
"""

import pandas as pd
import logging
import os
from datetime import datetime
import re

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

    def _normalize_column_names(self, df):
        """Normaliza nombres de columnas para matching preciso"""
        df.columns = [col.strip().lower().replace(' ', '') for col in df.columns]
        return df.rename(columns=self.column_mapping)

    def _normalize_date(self, date_str):
        """Convierte formatos complejos a ISO 8601"""
        try:
            # Corrección del error de sintaxis
            date_str = re.sub(r'\.-', '-', str(date_str).lower())  # Paréntesis cerrado
            if re.match(r'^\d{4}$', date_str):
                return f"{date_str}-01-01"
            return datetime.strptime(date_str, "%Y-%b-%d").strftime("%Y-%m-%d")
        except Exception as e:
            logger.warning(f"Fecha no válida: {date_str} - Usando fecha actual")
            return datetime.now().strftime("%Y-%m-%d")

    def process(self, input_file, output_base):
        try:
            # 1. Cargar datos omitiendo fila inicial sobrante
            df = pd.read_csv(input_file, skiprows=1, dtype=str, header=0)
            
            # 2. Normalizar nombres de columnas
            df = self._normalize_column_names(df)
            
            # 3. Validar y transformar fechas
            df['date'] = df['date'].apply(self._normalize_date)
            
            # 4. Campos obligatorios
            df['levelOfDescription'] = 'Documento'
            
            # 5. Crear rutas de salida
            os.makedirs(os.path.dirname(output_base), exist_ok=True)
            csv_output = f"{output_base}.csv"
            excel_output = f"{output_base}.xlsx"
            
            # 6. Guardar archivos
            df.to_csv(csv_output, index=False, encoding='utf-8-sig')
            df.to_excel(excel_output, index=False, engine='openpyxl')
            
            logger.info(f"✅ Archivos generados:\n- CSV: {csv_output}\n- Excel: {excel_output}")
            return True

        except Exception as e:
            logger.error(f"❌ Error crítico: {str(e)}", exc_info=True)
            return False

if __name__ == "__main__":
    converter = ISADConverter()
    if converter.process("Diplomas.csv", "resultados/salida_final"):
        print("✅ Transformación exitosa")
    else:
        print("❌ Error en la transformación")