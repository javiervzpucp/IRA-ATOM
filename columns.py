# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 15:21:04 2025

@author: jveraz
"""

import pandas as pd
import os
from dotenv import load_dotenv
import re
from huggingface_hub import InferenceClient  # Importar InferenceClient
from collections import Counter
import random  # Para seleccionar filas aleatorias

# Cargar variables de entorno
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HF_API_TOKEN")

# Inicializar el generador de texto (Mixtral)
generator_client = InferenceClient(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    token=HUGGINGFACE_API_KEY
)

# Cache para almacenar respuestas del LLM y evitar consultas repetidas
llm_cache = {}

# Función para detectar si una columna es de títulos usando reglas
def es_columna_titulo(columna, datos_columna):
    # Regla 1: Verificar si el nombre de la columna contiene palabras clave
    palabras_clave = ["titulo", "título", "title"]  # Variantes de "título"
    if any(palabra in columna.lower() for palabra in palabras_clave):
        return True
    
    # Regla 2: Verificar si los valores en la columna son cadenas de texto
    if all(isinstance(valor, str) for valor in datos_columna):
        # Regla 3: Verificar si la longitud promedio de los valores es mayor a un umbral
        longitud_promedio = sum(len(str(valor)) for valor in datos_columna) / len(datos_columna)
        if longitud_promedio > 10:  # Umbral de longitud
            return True
    
    # Si no se cumple ninguna regla, devolver False
    return False

# Función para confirmar con un LLM si una columna es de títulos
def confirmar_con_llm_si_es_titulo(datos_columna):
    try:
        # Tomar una muestra de los datos de la columna
        muestra = datos_columna[:5]  # Tomar los primeros 5 valores
        prompt = f"""
        ¿La siguiente lista de valores corresponde a títulos de documentos o registros?
        Los títulos suelen ser frases descriptivas, no códigos ni números.
        Valores: {muestra}
        Responde solo con "Sí" o "No".
        """
        
        # Llamada al modelo Mixtral
        response = generator_client.text_generation(
            prompt,
            max_new_tokens=5,  # Limitar la longitud de la respuesta
            temperature=0.1     # Sin aleatoriedad
        )
        
        # Extraer la respuesta
        respuesta = response.strip().lower()
        return respuesta == "sí" or respuesta == "si"
    except Exception as e:
        print(f"Error al confirmar con LLM: {e}")
        return False

# Función para detectar la columna de títulos en el DataFrame
def detectar_columna_titulo(df):
    # Lista de variantes posibles de la columna de títulos
    variantes_titulo = ["titulo", "título", "title"]
    
    # Buscar la columna de títulos (insensible a mayúsculas/minúsculas)
    for columna in df.columns:
        # Convertir el nombre de la columna a minúsculas para comparación
        columna_lower = columna.lower()
        
        # Verificar si el nombre de la columna coincide con alguna variante
        if columna_lower in variantes_titulo:
            return columna  # Devolver el nombre original de la columna
    
    # Si no se encuentra ninguna columna de títulos, usar reglas heurísticas y LLM
    for columna in df.columns:
        datos_columna = df[columna].dropna().tolist()  # Obtener datos no nulos de la columna
        if es_columna_titulo(columna, datos_columna):
            return columna
        elif confirmar_con_llm_si_es_titulo(datos_columna):
            return columna
    
    return None  # Si no se detecta ninguna columna de títulos

# Función para detectar la columna de fecha
def detectar_columna_fecha(df):
    # Lista de posibles nombres de columnas de fecha
    posibles_nombres = ["año", "fecha", "date", "year", "fechas"]
    
    # Buscar la columna de fecha por nombre
    for columna in df.columns:
        if any(nombre in columna.lower() for nombre in posibles_nombres):
            return columna
    
    # Si no se encuentra por nombre, buscar por formato de datos
    for columna in df.columns:
        # Verificar si los datos parecen ser fechas (por ejemplo, contienen años)
        muestra = df[columna].dropna().head(5).tolist()  # Tomar una muestra de 5 valores
        if any(re.match(r'\d{4}', str(valor)) for valor in muestra):  # Buscar años de 4 dígitos
            return columna
    
    return None  # Si no se detecta ninguna columna de fecha

# Función para normalizar fechas usando Mixtral
def normalizar_fecha_con_mixtral(texto_fecha):
    try:
        # Prompt para el modelo con las reglas específicas
        prompt = f"""
        Normaliza la siguiente fecha siguiendo estas reglas:
        1. Si solo hay un año (por ejemplo, "1984"), no agregues ni el mes ni el día. Devuelve solo el año.
        2. Si hay un año con un signo de interrogación (por ejemplo, "1984?"), no lo modifiques. Devuelve el año con el signo de interrogación.
        3. Si la celda está vacía, devuelve "No disponible". No inventes ningún dato.
        4. Si la fecha ya está en formato AAAA-MM-DD, AAAA-MM o AAAA, devuélvela tal como está.
        5. Si no se puede normalizar, devuelve "No disponible".

        Fecha: {texto_fecha}
        """
        
        # Llamada al modelo Mixtral
        response = generator_client.text_generation(
            prompt,
            max_new_tokens=15,  # Limitar la longitud de la respuesta
            temperature=0.1     # Sin aleatoriedad
        )
        
        # Extraer la respuesta
        fecha_normalizada = response.strip()
        
        # Verificar que la fecha tenga un formato válido
        if re.match(r'^\d{4}(\?)?$|^\d{4}-\d{2}(-\d{2})?$|^No disponible$', fecha_normalizada):
            return fecha_normalizada
        else:
            return "No disponible"
    except Exception as e:
        print(f"Error al normalizar la fecha: {e}")
        return "No disponible"

# Función para estandarizar fechas (combina regex y Mixtral)
def estandarizar_fecha(fecha):
    # Expresiones regulares para identificar formatos comunes
    patron_aaaa_mm_dd = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
    patron_aaaa_mm = re.compile(r'(\d{4})-(\d{2})')
    patron_aaaa = re.compile(r'(\d{4})')
    patron_aaaa_interrogacion = re.compile(r'(\d{4})\?')
    patron_aaaa_mm_interrogacion = re.compile(r'(\d{4})-(\d{2})-(\?)')  # Nuevo patrón para "AAAA-MM-?"
    patron_aaaa_interrogacion_simple = re.compile(r'(\d{2})-(\?)')  # Nuevo patrón para "AA-?"

    if pd.isna(fecha) or fecha == "":
        return "No disponible"
    
    # Convertir a string
    fecha = str(fecha)

    # Intentar coincidir con los patrones comunes
    if patron_aaaa_mm_dd.match(fecha):
        return fecha  # Ya está en el formato correcto
    elif patron_aaaa_mm.match(fecha):
        return fecha  # Devolver solo año y mes, sin inventar el día
    elif patron_aaaa.match(fecha):
        return fecha  # Devolver solo el año, sin inventar mes o día
    elif patron_aaaa_interrogacion.match(fecha):
        return fecha  # Devolver el año con el signo de interrogación
    elif patron_aaaa_mm_interrogacion.match(fecha):
        return fecha.replace("?", "")  # Eliminar el "?" pero mantener el guion
    elif patron_aaaa_interrogacion_simple.match(fecha):
        return fecha.replace("?", "")  # Eliminar el "?" pero mantener el guion
    else:
        # Si no coincide con ningún patrón, usar Mixtral para interpretar
        return normalizar_fecha_con_mixtral(fecha)

# Función para predecir el tipo de evento ("creación" o "acumulación")
def predecir_event_type(fecha):
    if pd.isna(fecha) or fecha == "No disponible":
        return "No disponible"
    
    # Convertir la fecha a minúsculas para facilitar la comparación
    fecha_lower = str(fecha).lower()
    
    # Términos que indican "acumulación"
    terminos_acumulacion = ["registro", "acumulación", "acumulacion", "ingreso", "proceso"]
    
    # Verificar si la fecha contiene algún término de "acumulación"
    if any(termino in fecha_lower for termino in terminos_acumulacion):
        return "acumulación"
    else:
        return "creación"  # Por defecto, es "creación"

# Función para determinar el eventType más común en 5 filas aleatorias
def determinar_event_type_con_votacion_global(df):
    # Seleccionar 5 filas aleatorias de todo el dataset
    filas_muestras = df.sample(n=min(5, len(df)), random_state=42)  # Selección aleatoria
    
    # Obtener los eventType de las filas seleccionadas
    event_types = filas_muestras['eventType'].tolist()
    
    # Contar la frecuencia de cada eventType
    contador_event_types = Counter(event_types)
    
    # Obtener el eventType más común y su frecuencia
    event_type_comun, count = contador_event_types.most_common(1)[0]
    
    # Si el eventType más común aparece en al menos 3 filas, asignarlo a todas las filas
    if count >= 3:
        return event_type_comun
    else:
        return "creación"  # Si no hay consenso, asignar "creación" por defecto

# Función para consultar al LLM el nivel de descripción
def consultar_llm_para_level_of_description(titulo, fecha):
    try:
        # Prompt para el modelo con las reglas específicas
        prompt = f"""
        Determina el nivel de descripción adecuado para el siguiente documento basándote en su título y fecha.
        Los niveles de descripción posibles son:
        - Fondo. Ejemplo: "Archivo de la Familia Pérez" o "Archivo de la Compañía XYZ".
        - Subfondo. Ejemplo: "Departamento de Contabilidad" dentro del "Archivo de la Compañía XYZ".
        - Sección. Ejemplo: "Correspondencia administrativa" o "Registros financieros".
        - Subserie. Ejemplo: "Facturas de proveedores" dentro de la serie "Registros financieros".
        - Expediente. Ejemplo: "Expediente de contratación de empleados para el año 1995".
        - Documento. Ejemplo: "Carta de renuncia de Juan Pérez, fechada el 15 de marzo de 1995".
        - Colección. Ejemplo: "Colección de fotografías históricas de la ciudad".
        - Subcolección. Ejemplo: "Fotografías de edificios históricos" dentro de la "Colección de fotografías históricas de la ciudad".
        - Fondo Agrupado. Ejemplo: "Registros del Ministerio de Educación".
        - Subgrupo de Fondo. Ejemplo: "Registros de la Dirección de Primaria" dentro del "Registro del Ministerio de Educación".
        - Otro. Ejemplo: Descripciones que no siguen una estructura jerárquica tradicional.

        Título: {titulo}
        Fecha: {fecha}

        Responde solo con el nivel de descripción más adecuado.
        """
        
        # Llamada al modelo Mixtral
        response = generator_client.text_generation(
            prompt,
            max_new_tokens=10,  # Limitar la longitud de la respuesta
            temperature=0.1     # Sin aleatoriedad
        )
        
        # Extraer la respuesta
        nivel_descripcion = response.strip()
        
        # Verificar que la respuesta sea un nivel de descripción válido
        niveles_validos = [
            "Fondo", "Subfondo", "Sección", "Subserie", "Expediente",
            "Documento", "Colección", "Subcolección", "Fondo Agrupado",
            "Subgrupo de Fondo", "Otro"
        ]
        if nivel_descripcion in niveles_validos:
            return nivel_descripcion
        else:
            return "Otro"
    except Exception as e:
        print(f"Error al consultar el LLM: {e}")
        return "Otro"

# Función para determinar el nivel de descripción usando votación global
def determinar_level_of_description_con_votacion_global(df, columna_titulo):
    # Seleccionar 5 filas aleatorias de todo el dataset
    filas_muestras = df.sample(n=min(5, len(df)), random_state=42)  # Selección aleatoria
    
    # Consultar al LLM para las filas de muestra
    niveles = []
    for _, fila in filas_muestras.iterrows():  # Usar iterrows para acceder a las filas
        nivel = consultar_llm_para_level_of_description(fila[columna_titulo], fila['date'])
        niveles.append(nivel)
    
    # Contar la frecuencia de cada nivel de descripción
    contador_niveles = Counter(niveles)
    
    # Si al menos 3 filas tienen el mismo nivel, asignar ese nivel a todas las filas
    nivel_descripcion, count = contador_niveles.most_common(1)[0]
    if count >= 3:
        return nivel_descripcion
    else:
        return "Otro"  # Si no hay consenso, asignar "Otro"

# Función para asignar el nivel de descripción a todas las filas
def asignar_level_of_description_global(df, columna_titulo):
    # Determinar el nivel de descripción usando votación global
    nivel_descripcion = determinar_level_of_description_con_votacion_global(df, columna_titulo)
    
    # Asignar el nivel de descripción a todas las filas
    df['levelOfDescription'] = nivel_descripcion
    return df

# Leer el archivo Excel
df = pd.read_excel('Novenas Sánchez-Concha La Combe.xlsx', sheet_name='Novenas')

# Detectar la columna de títulos
columna_titulo = detectar_columna_titulo(df)
if columna_titulo:
    print(f"Columna de títulos detectada: {columna_titulo}")
else:
    print("No se detectó ninguna columna de títulos.")

# Detectar la columna de fecha
columna_fecha = detectar_columna_fecha(df)
if columna_fecha:
    print(f"Columna de fecha detectada: {columna_fecha}")
else:
    print("No se detectó ninguna columna de fecha.")

# Formatear la columna de fecha y crear la columna 'date'
df['date'] = df[columna_fecha].apply(estandarizar_fecha)

# Crear la columna 'eventType' y asignarle valores basados en la columna 'date'
df['eventType'] = df['date'].apply(predecir_event_type)

# Determinar el eventType más común usando votación global
event_type_global = determinar_event_type_con_votacion_global(df)

# Asignar el eventType más común a todas las filas
df['eventType'] = event_type_global

# Asignar el nivel de descripción a todas las filas
df = asignar_level_of_description_global(df, columna_titulo)

# Crear un nuevo DataFrame con las columnas requeridas
resultados = df[['Id', 'date', columna_titulo, 'eventType', 'levelOfDescription']].rename(
    columns={'Id': 'identifier', columna_titulo: 'title'}
)

# Guardar el nuevo DataFrame en un archivo Excel
resultados.to_excel('columnas_estandarizadas.xlsx', index=False)