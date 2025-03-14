# -*- coding: utf-8 -*-
"""
Transformador de datos archivísticos a estándares ISAD(G) y AtoM
Versión Completa 1.0 - Incluye todas las funcionalidades
"""

import os
import logging
import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEndpoint
from rdflib import Graph, URIRef, Literal, Namespace, RDF, XSD
from sentence_transformers import SentenceTransformer, util
import spacy
from langdetect import detect, DetectorFactory
import langid
import country_converter as coco
from geopy.geocoders import Nominatim
from rapidfuzz import process, fuzz
from unidecode import unidecode
from spacy.tokens import Token  # Añadir este import
from spacy.language import Language  # Import necesario para componentes

# Configuración inicial
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
DetectorFactory.seed = 0
load_dotenv()

# Constantes y Namespaces
ISAD = Namespace("http://archivematica.org/isad/")
DCTERMS = Namespace("http://purl.org/dc/terms/")
SCHEMA = Namespace("http://schema.org/")
ARCH = Namespace("http://purl.org/archival/vocab/arch#")

class ISADProcessor:
    def __init__(self, lang: str = 'es'):
        self.lang = lang
        self.nlp = self._load_spacy_model()
        self.embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
        self.geolocator = Nominatim(user_agent="isad_processor_v2")
        self.country_converter = coco.CountryConverter()
        self._load_isad_structure()
        self._init_llm()

    def _load_spacy_model(self):
        """Carga el modelo de SpaCy con personalizaciones para español"""
        try:
            if self.lang == 'es':
                nlp = spacy.load("es_core_news_lg")
                self._add_spanish_components(nlp)
                return nlp
            return spacy.load("en_core_web_lg")
        except IOError:
            logger.error("Modelo SpaCy no encontrado. Ejecutar: python -m spacy download es_core_news_lg")
            raise

    def _add_spanish_components(self, nlp):
        """Añade componentes personalizados para procesamiento en español"""
        from spacy.lang.es import Spanish
        
        # Configurar pipeline para fechas
        nlp.add_pipe("merge_entities")
        
        @Spanish.component("date_normalizer")
        def date_normalizer(doc):
            for ent in doc.ents:
                if ent.label_ == "DATE":
                    ent._.normalized_date = self._normalize_spanish_date(ent.text)
            return doc
        
        Token.set_extension("normalized_date", default=None,force=True)
        nlp.add_pipe("date_normalizer", after="ner")

    def _load_isad_structure(self):
        """Carga la estructura ISAD(G) con metadata extendida"""
        self.ISAD_GROUPS = {
            "Identificación": {
                "fields": ["referenceCode", "title", "date", "levelOfDescription"],
                "mandatory": True,
                "description_es": "Información esencial para identificar la unidad de descripción"
            },
            "Contexto": {
                "fields": ["custodialHistory", "archivalHistory"],
                "description_es": "Historia archivística y procedencia del material"
            },
            "Contenido": {
                "fields": ["scopeAndContent", "extent", "systemOfArrangement"],
                "description_es": "Descripción del contenido y estructura"
            },
            "Acceso": {
                "fields": ["conditionsGoverningAccess", "conditionsGoverningReproduction"],
                "description_es": "Condiciones de acceso y uso"
            },
            "Documentación Relacionada": {
                "fields": ["relatedUnitsOfDescription", "publicationNote"],
                "description_es": "Material relacionado y publicaciones"
            }
        }

        self.ISAD_FIELDS = {
            "referenceCode": {
                "description_es": "Código único de referencia según sistema institucional",
                "mandatory": True,
                "data_type": XSD.string,
                "validation": r"^[A-Z]{2,4}-\d{3,}(/\d+)*$",
                "examples": ["AH-MP-001", "F-003/005"]
            },
            "title": {
                "description_es": "Título formal de la unidad de descripción",
                "mandatory": True,
                "data_type": XSD.string,
                "validation": r".{10,255}",
                "examples": ["Correspondencia oficial, 1890-1905"]
            },
            "date": {
                "description_es": "Fechas de creación del material en formato ISO 8601",
                "mandatory": True,
                "data_type": XSD.date,
                "validation": r"\d{4}(-\d{2}(-\d{2})?)?",
                "examples": ["1890-1905", "1923-05-15"]
            },
            "levelOfDescription": {
                "description_es": "Nivel jerárquico en la estructura archivística",
                "mandatory": True,
                "data_type": XSD.string,
                "allowed_values": ["Fondo", "Sección", "Serie", "Expediente", "Documento"]
            },
            "scopeAndContent": {
                "description_es": "Resumen del contenido y características principales",
                "data_type": XSD.string,
                "min_length": 50
            }
        }

    def _init_llm(self):
        """Inicializa el modelo de lenguaje grande"""
        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            huggingfacehub_api_token=os.getenv("HF_API_TOKEN"),
            temperature=0.3,
            max_new_tokens=500,
            timeout=30
        )

    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Carga y limpia datos de entrada"""
        try:
            # Detectar formato y cargar
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, dtype=str, keep_default_na=False, 
                               on_bad_lines='warn', encoding_errors='replace')
            else:
                df = pd.read_excel(file_path, dtype=str, keep_default_na=False)
            
            # Limpiar datos
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Eliminar columnas sin nombre
            df = df.dropna(axis=1, how='all')  # Eliminar columnas totalmente vacías
            df = df.map(lambda x: unidecode(str(x).strip()) if x else x)
            
            return df.replace({'': pd.NA, 'NaN': pd.NA}).dropna(axis=0, how='all')
        except Exception as e:
            logger.error(f"Error cargando archivo: {str(e)}")
            raise

    def _semantic_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mapeo semántico mejorado para español"""
        column_mappings = {}
        
        for col in df.columns:
            original_col = col
            col = unidecode(col).lower().strip()
            
            # Paso 1: Detección de patrones comunes
            pattern_matches = {
                r'(cod|ref)': 'referenceCode',
                r'titulo': 'title',
                r'fecha': 'date',
                r'nivel': 'levelOfDescription',
                r'contenido': 'scopeAndContent'
            }
            
            for pattern, field in pattern_matches.items():
                if re.search(pattern, col, re.IGNORECASE):
                    column_mappings[original_col] = field
                    logger.info(f"Mapeo por patrón: {original_col} -> {field}")
                    break
            else:
                # Paso 2: Búsqueda semántica
                col_embedding = self.embedding_model.encode(col)
                similarities = {
                    field: util.cos_sim(col_embedding, 
                                     self.embedding_model.encode(meta['description_es'])).item()
                    for field, meta in self.ISAD_FIELDS.items()
                }
                best_match = max(similarities, key=similarities.get)
                
                # Paso 3: Validación con fuzzy matching
                fuzzy_match = process.extractOne(col, 
                                               list(self.ISAD_FIELDS.keys()), 
                                               scorer=fuzz.token_set_ratio)
                
                if similarities[best_match] >= 0.5 or fuzzy_match[1] >= 60:
                    final_match = best_match if similarities[best_match] > 0.5 else fuzzy_match[0]
                    column_mappings[original_col] = final_match
                    logger.info(f"Mapeo semántico: {original_col} -> {final_match}")
                else:
                    logger.warning(f"Columna no mapeada: {original_col}")

        # Aplicar mapeos y eliminar columnas no relevantes
        df = df.rename(columns=column_mappings)
        valid_fields = [f for f in self.ISAD_FIELDS if f in df.columns]
        return df[valid_fields]

    def _normalize_spanish_date(self, date_str: str) -> Optional[str]:
        """Normalización avanzada de fechas en español"""
        month_map = {
            'ene': '01', 'feb': '02', 'mar': '03', 'abr': '04', 'may': '05', 
            'jun': '06', 'jul': '07', 'ago': '08', 'sep': '09', 'oct': '10',
            'nov': '11', 'dic': '12'
        }
        
        patterns = [
            # Formato: 15 de marzo de 2023
            (r'(\d{1,2})\s+de\s+([a-z]+)\s+de\s+(\d{4})', 
             lambda m: f"{m.group(3)}-{month_map[m.group(2)[:3].lower()]}-{m.group(1).zfill(2)}"),
    
            # Formato: marzo 2023 (corregido)
            (r'([a-z]+)\s+(\d{4})', 
            lambda m: f"{m.group(2)}-{month_map[m.group(1)[:3].lower()]}-01"),
    
            # Formato: 2023-03-15
            (r'(\d{4}-\d{2}-\d{2})', lambda m: m.group(1)),
    
            # Formato: 03/2023
            (r'(\d{2})/(\d{4})', lambda m: f"{m.group(2)}-{m.group(1)}-01"),
    
            # Solo año
            (r'\b(\d{4})\b', lambda m: f"{m.group(1)}-01-01")
            ]
        
        date_str = unidecode(date_str.lower())
        for pattern, converter in patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    return converter(match)
                except (KeyError, ValueError):
                    continue
        return None

    def _enrich_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enriquecimiento de metadatos con NLP y geocodificación"""
        # Normalización de fechas
        if 'date' in df.columns:
            df['date'] = df['date'].apply(self._normalize_spanish_date)
        
        # Detección de idioma
        if 'scopeAndContent' in df.columns:
            df['language'] = df['scopeAndContent'].apply(self._detect_language)
        
        # Extracción de entidades
        if 'scopeAndContent' in df.columns:
            entity_cols = df['scopeAndContent'].apply(self._extract_entities).apply(pd.Series)
            df = pd.concat([df, entity_cols], axis=1)
        
        # Geocodificación
        if 'placeAccessPoints' in df.columns:
            df['country'] = df['placeAccessPoints'].apply(self._geocode_location)
        
        return df

    def _detect_language(self, text: str) -> Optional[str]:
        """Detección robusta de idioma para textos cortos"""
        try:
            if pd.isna(text) or len(text) < 20:
                return None
                
            # Forzar prioridad a español
            langid.set_languages(['es', 'en', 'pt', 'fr'])
            lang, _ = langid.classify(text)
            return lang if detect(text) == lang else None
        except:
            return None

    def _extract_entities(self, text: str) -> Dict:
        """Extracción de entidades en documentos archivísticos en español"""
        doc = self.nlp(text[:100000])  # Limitar a 100KB por rendimiento
        
        entities = {
            'personas': [],
            'instituciones': [],
            'lugares': [],
            'fechas': [],
            'tipos_documentales': []
        }
        
        # Entidades nombradas
        for ent in doc.ents:
            if ent.label_ in ("PER", "PERSON"):
                entities['personas'].append(ent.text)
            elif ent.label_ in ("ORG", "INST"):
                entities['instituciones'].append(ent.text)
            elif ent.label_ in ("LOC", "LUG"):
                entities['lugares'].append(ent.text)
            elif ent.label_ == "DATE" and ent._.normalized_date:
                entities['fechas'].append(ent._.normalized_date)
        
        # Detección de tipos documentales
        tipos = [
            'carta', 'informe', 'decreto', 'ley', 'contrato', 
            'acta', 'memorándum', 'telegrama', 'fotografía'
        ]
        entities['tipos_documentales'] = [t for t in tipos if t in text.lower()]
        
        # Enriquecimiento con LLM
        if not entities['tipos_documentales']:
            prompt = f"""Identifica tipos documentales en este texto (solo nombres en español):
            {text[:1500]}
            Respuesta en formato: Tipo1, Tipo2, Tipo3"""
            
            try:
                response = self.llm.invoke(prompt).strip()
                entities['tipos_documentales'] = [t.strip() for t in response.split(',')[:3]]
            except Exception as e:
                logger.error(f"Error en LLM: {str(e)}")
        
        return {k: ', '.join(sorted(set(v))) if v else None for k, v in entities.items()}

    def _geocode_location(self, location: str) -> Optional[str]:
        """Geocodificación con múltiples estrategias"""
        try:
            # Primero usar country_converter
            country = self.country_converter.convert(names=[location], to='name_short')
            if country != 'not found':
                return country
                
            # Fallback a geopy
            geo = self.geolocator.geocode(location, exactly_one=True, language='es')
            if geo:
                return self.country_converter.convert(names=[geo.address.split(',')[-1].strip()], 
                                                    to='name_short')
        except Exception as e:
            logger.warning(f"Error geocodificando {location}: {str(e)}")
            return None

    def _generate_rdf(self, df: pd.DataFrame) -> Graph:
        """Generación de RDF con validación de tipos"""
        g = Graph()
        g.bind("isad", ISAD)
        g.bind("dcterms", DCTERMS)
        
        for _, row in df.iterrows():
            if not row.get('referenceCode'):
                continue
                
            subject = URIRef(f"{ISAD}{row['referenceCode']}")
            g.add((subject, RDF.type, ISAD.ArchivalDescription))
            
            for field in self.ISAD_FIELDS:
                value = row.get(field)
                if pd.notna(value):
                    predicate = ISAD[field]
                    dtype = self.ISAD_FIELDS[field].get('data_type', XSD.string)
                    
                    try:
                        literal = Literal(value, datatype=dtype)
                    except:
                        literal = Literal(str(value))
                    
                    g.add((subject, predicate, literal))
        
        return g

    def process(self, input_path: str, output_dir: str) -> bool:
        """Ejecuta el pipeline completo de transformación"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            
            # 1. Carga y limpieza
            df = self._load_data(input_path)
            
            # 2. Mapeo semántico
            df = self._semantic_mapping(df)
            
            # 3. Validación de campos obligatorios
            missing = [f for f, m in self.ISAD_FIELDS.items() 
                     if m.get('mandatory') and f not in df.columns]
            if missing:
                raise ValueError(f"Campos obligatorios faltantes: {', '.join(missing)}")
            
            # 4. Enriquecimiento
            df = self._enrich_metadata(df)
            
            # 5. Generación de salidas
            # CSV
            csv_path = os.path.join(output_dir, f"{base_name}_isad.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            # RDF
            rdf_graph = self._generate_rdf(df)
            rdf_path = os.path.join(output_dir, f"{base_name}_isad.ttl")
            rdf_graph.serialize(destination=rdf_path, format='turtle')
            
            # Excel con validaciones
            excel_path = os.path.join(output_dir, f"{base_name}_isad.xlsx")
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Datos')
                
                # Configurar validaciones
                workbook = writer.book
                worksheet = writer.sheets['Datos']
                
                # Validación para levelOfDescription
                allowed_values = self.ISAD_FIELDS['levelOfDescription']['allowed_values']
                col_idx = df.columns.get_loc('levelOfDescription')
                worksheet.data_validation(1, col_idx, 1048576, col_idx, {
                    'validate': 'list',
                    'source': allowed_values,
                    'error_title': 'Valor inválido',
                    'error_message': f"Valores permitidos: {', '.join(allowed_values)}"
                })
            
            logger.info(f"Proceso completado exitosamente en {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error en el procesamiento: {str(e)}", exc_info=True)
            return False

if __name__ == "__main__":
    processor = ISADProcessor(lang='es')
    result = processor.process(
        input_path="Diplomas.csv",
        output_dir="resultados"
    )
    print("✅ Transformación exitosa" if result else "❌ Error en la transformación")