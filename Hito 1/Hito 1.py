# ==========================================
# ÍNDICE DE CALIDAD VIAL (ICV)
# CASO: San Miguel, Santiago, Chile
# Enfoque: Data Science + Ingeniería Urbana
# Adaptado para incluir:
# - RUIDO (proxy vial)
# - LUMINARIAS
# - TRAFICO_PROMEDIO (proxy vial)
# ==========================================

import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np

ox.settings.use_cache = True
ox.settings.log_console = True

# ==========================================
# 1. PARÁMETROS
# ==========================================
place_name = 'San Miguel, Santiago, Chile'
CRS_PROY = 'EPSG:32719'

# Ponderaciones
W_CONECTIVIDAD = 0.22
W_ACCESIBILIDAD = 0.22
W_JERARQUIA = 0.16
W_PEATONAL = 0.15
W_EQUIPAMIENTO = 0.10
W_RUIDO = 0.05
W_TRAFICO = 0.10

# ==========================================
# 2. FUNCIONES AUXILIARES
# ==========================================
def primer_valor(x):
    if isinstance(x, list):
        return x[0]
    return x

def parsear_numero_simple(x, default=np.nan):
    x = primer_valor(x)
    if pd.isna(x):
        return default
    try:
        texto = str(x).strip().lower().replace(',', '.')
        numero = ''.join(ch for ch in texto if ch.isdigit() or ch == '.')
        return float(numero) if numero != '' else default
    except:
        return default

# ==========================================
# 3. RED VIAL
# ==========================================
G = ox.graph_from_place(place_name, network_type='drive')
nodes, edges = ox.graph_to_gdfs(G)

nodes = nodes.to_crs(CRS_PROY)
edges = edges.to_crs(CRS_PROY)

# ==========================================
# 4. CONECTIVIDAD
# ==========================================
degree_dict = dict(G.degree())

nodes['conectividad'] = nodes.index.map(degree_dict)
nodes['conectividad'] = nodes['conectividad'].fillna(0)

if nodes['conectividad'].max() == nodes['conectividad'].min():
    nodes['conectividad_norm'] = 0
else:
    nodes['conectividad_norm'] = (
        nodes['conectividad'] - nodes['conectividad'].min()
    ) / (nodes['conectividad'].max() - nodes['conectividad'].min())

# ==========================================
# 5. JERARQUÍA VIAL
# ==========================================
def clasificar_jerarquia(highway):
    highway = primer_valor(highway)

    if highway in ['motorway', 'trunk', 'primary']:
        return 1.0
    elif highway in ['secondary']:
        return 0.7
    elif highway in ['tertiary']:
        return 0.5
    else:
        return 0.3

edges['jerarquia'] = edges['highway'].apply(clasificar_jerarquia)

jerarquia_nodos = edges.groupby('u')['jerarquia'].mean()
nodes['jerarquia'] = nodes.index.map(jerarquia_nodos)
nodes['jerarquia'] = nodes['jerarquia'].fillna(0)

if nodes['jerarquia'].max() == nodes['jerarquia'].min():
    nodes['jerarquia_norm'] = 0
else:
    nodes['jerarquia_norm'] = (
        nodes['jerarquia'] - nodes['jerarquia'].min()
    ) / (nodes['jerarquia'].max() - nodes['jerarquia'].min())

# ==========================================
# 6. ACCESIBILIDAD A SERVICIOS
# ==========================================
tags_servicios = {'amenity': True}

servicios = ox.features_from_place(place_name, tags_servicios)
servicios = servicios[servicios.geometry.notnull()].copy()

if len(servicios) == 0:
    print('⚠ No hay servicios disponibles')
    nodes['dist_servicio'] = np.nan
    nodes['accesibilidad'] = 0
    nodes['accesibilidad_norm'] = 0
else:
    servicios = servicios.to_crs(CRS_PROY)
    servicios['geometry'] = servicios.geometry.centroid

    nodes['dist_servicio'] = nodes.geometry.apply(
        lambda x: servicios.distance(x).min()
    )

    nodes['accesibilidad'] = 1 / (nodes['dist_servicio'] + 1)

    if nodes['accesibilidad'].max() == nodes['accesibilidad'].min():
        nodes['accesibilidad_norm'] = 0
    else:
        nodes['accesibilidad_norm'] = (
            nodes['accesibilidad'] - nodes['accesibilidad'].min()
        ) / (nodes['accesibilidad'].max() - nodes['accesibilidad'].min())

print('✔ Accesibilidad calculada')

# ==========================================
# 7. INFRAESTRUCTURA PEATONAL
# ==========================================
tags_walk = {'highway': ['footway', 'pedestrian', 'path']}

walkways = ox.features_from_place(place_name, tags_walk)
walkways = walkways[walkways.geometry.notnull()].copy()

if len(walkways) == 0:
    print('⚠ No hay infraestructura peatonal')
    nodes['peatonal'] = 0
    nodes['peatonal_norm'] = 0
else:
    walkways = walkways.to_crs(CRS_PROY)

    def densidad_peatonal(punto):
        buffer = punto.buffer(200)
        return walkways.intersects(buffer).sum()

    nodes['peatonal'] = nodes.geometry.apply(densidad_peatonal)

    if nodes['peatonal'].max() == nodes['peatonal'].min():
        nodes['peatonal_norm'] = 0
    else:
        nodes['peatonal_norm'] = (
            nodes['peatonal'] - nodes['peatonal'].min()
        ) / (nodes['peatonal'].max() - nodes['peatonal'].min())

print('✔ Infraestructura peatonal calculada')

# ==========================================
# 8. DENSIDAD DE EQUIPAMIENTO
# ==========================================
tags_pois = {'amenity': True, 'shop': True}

pois = ox.features_from_place(place_name, tags_pois)
pois = pois[pois.geometry.notnull()].copy()

if len(pois) == 0:
    print('⚠ No hay equipamientos disponibles')
    nodes['equipamiento'] = 0
    nodes['equipamiento_norm'] = 0
else:
    pois = pois.to_crs(CRS_PROY)
    pois['geometry'] = pois.geometry.centroid

    def densidad_pois(punto):
        buffer = punto.buffer(300)
        return pois.intersects(buffer).sum()

    nodes['equipamiento'] = nodes.geometry.apply(densidad_pois)

    if nodes['equipamiento'].max() == nodes['equipamiento'].min():
        nodes['equipamiento_norm'] = 0
    else:
        nodes['equipamiento_norm'] = (
            nodes['equipamiento'] - nodes['equipamiento'].min()
        ) / (nodes['equipamiento'].max() - nodes['equipamiento'].min())

print('✔ Equipamiento calculado')

# ==========================================
# 9. RUIDO VIAL (PROXY)
# ==========================================
def score_ruido_highway(highway):
    highway = primer_valor(highway)
    if highway in ['motorway', 'trunk', 'primary']:
        return 1.0
    elif highway in ['secondary']:
        return 0.8
    elif highway in ['tertiary']:
        return 0.6
    else:
        return 0.3

edges['ruido_highway'] = edges['highway'].apply(score_ruido_highway)

if 'maxspeed' in edges.columns:
    edges['maxspeed_num'] = edges['maxspeed'].apply(parsear_numero_simple)
else:
    edges['maxspeed_num'] = np.nan

def score_ruido_velocidad(v):
    if pd.isna(v):
        return 0.5
    elif v >= 80:
        return 1.0
    elif v >= 60:
        return 0.8
    elif v >= 40:
        return 0.6
    else:
        return 0.3

edges['ruido_velocidad'] = edges['maxspeed_num'].apply(score_ruido_velocidad)

if 'lanes' in edges.columns:
    edges['lanes_num'] = edges['lanes'].apply(parsear_numero_simple)
else:
    edges['lanes_num'] = np.nan

def score_ruido_pistas(n):
    if pd.isna(n):
        return 0.5
    elif n >= 4:
        return 1.0
    elif n >= 2:
        return 0.7
    else:
        return 0.4

edges['ruido_pistas'] = edges['lanes_num'].apply(score_ruido_pistas)

edges['ruido_tramo'] = (
    0.5 * edges['ruido_highway'] +
    0.3 * edges['ruido_velocidad'] +
    0.2 * edges['ruido_pistas']
)

ruido_nodos = edges.groupby('u')['ruido_tramo'].mean()
nodes['ruido'] = nodes.index.map(ruido_nodos)
nodes['ruido'] = nodes['ruido'].fillna(edges['ruido_tramo'].mean())

if nodes['ruido'].max() == nodes['ruido'].min():
    nodes['ruido_norm'] = 0
else:
    nodes['ruido_norm'] = (
        nodes['ruido'] - nodes['ruido'].min()
    ) / (nodes['ruido'].max() - nodes['ruido'].min())

print('✔ Ruido vial calculado como proxy')

# ==========================================
# 10. LUMINARIAS
# ==========================================
tags_luminaria = {'highway': 'street_lamp'}
luminarias = ox.features_from_place(place_name, tags_luminaria)
luminarias = luminarias[luminarias.geometry.notnull()].copy()

if len(luminarias) == 0:
    print('⚠ No hay luminarias street_lamp disponibles')
    nodes['cant_luminarias'] = 0
else:
    luminarias = luminarias.to_crs(CRS_PROY)
    luminarias['geometry'] = luminarias.geometry.centroid

    def contar_luminarias_nodo(punto, radio=250):
        buffer = punto.buffer(radio)
        lum_cercanas = luminarias[luminarias.intersects(buffer)]
        return len(lum_cercanas)

    nodes['cant_luminarias'] = nodes.geometry.apply(contar_luminarias_nodo)

print('✔ Luminarias calculadas')

# ==========================================
# 11. TRAFICO PROMEDIO (PROXY)
# Basado en highway + maxspeed + lanes
# ==========================================
def score_trafico_highway(highway):
    highway = primer_valor(highway)
    if highway in ['motorway', 'trunk']:
        return 1.0
    elif highway in ['primary']:
        return 0.9
    elif highway in ['secondary']:
        return 0.7
    elif highway in ['tertiary']:
        return 0.5
    else:
        return 0.3

edges['trafico_highway'] = edges['highway'].apply(score_trafico_highway)

def score_trafico_velocidad(v):
    if pd.isna(v):
        return 0.5
    elif v >= 80:
        return 1.0
    elif v >= 60:
        return 0.8
    elif v >= 40:
        return 0.6
    else:
        return 0.3

edges['trafico_velocidad'] = edges['maxspeed_num'].apply(score_trafico_velocidad)

def score_trafico_pistas(n):
    if pd.isna(n):
        return 0.5
    elif n >= 4:
        return 1.0
    elif n >= 3:
        return 0.85
    elif n >= 2:
        return 0.65
    else:
        return 0.35

edges['trafico_pistas'] = edges['lanes_num'].apply(score_trafico_pistas)

edges['trafico_tramo'] = (
    0.5 * edges['trafico_highway'] +
    0.25 * edges['trafico_velocidad'] +
    0.25 * edges['trafico_pistas']
)

trafico_nodos = edges.groupby('u')['trafico_tramo'].mean()
nodes['trafico_promedio'] = nodes.index.map(trafico_nodos)
nodes['trafico_promedio'] = nodes['trafico_promedio'].fillna(edges['trafico_tramo'].mean())

if nodes['trafico_promedio'].max() == nodes['trafico_promedio'].min():
    nodes['trafico_promedio_norm'] = 0
else:
    nodes['trafico_promedio_norm'] = (
        nodes['trafico_promedio'] - nodes['trafico_promedio'].min()
    ) / (nodes['trafico_promedio'].max() - nodes['trafico_promedio'].min())

print('✔ Trafico promedio calculado como proxy')

# ==========================================
# 12. ÍNDICE DE CALIDAD VIAL (ICV)
# ruido entra negativo, trafico entra positivo
# ==========================================
nodes['ICV'] = (
    W_CONECTIVIDAD * nodes['conectividad_norm'] +
    W_ACCESIBILIDAD * nodes['accesibilidad_norm'] +
    W_JERARQUIA * nodes['jerarquia_norm'] +
    W_PEATONAL * nodes['peatonal_norm'] +
    W_EQUIPAMIENTO * nodes['equipamiento_norm'] +
    W_RUIDO * (1 - nodes['ruido_norm']) +
    W_TRAFICO * nodes['trafico_promedio_norm']
)

# ==========================================
# 13. CLASIFICACIÓN OPCIONAL
# ==========================================
def clasificar_icv(valor):
    if valor >= 0.75:
        return 'Alta'
    elif valor >= 0.50:
        return 'Media'
    else:
        return 'Baja'

nodes['categoria_icv'] = nodes['ICV'].apply(clasificar_icv)

# ==========================================
# 14. EXPORTACIÓN
# ==========================================
nodes.to_file('indice_calidad_vial_nodos.shp')
edges.to_file('red_vial_tramos_con_ruido_y_trafico.shp')

if len(luminarias) > 0:
    luminarias.to_file('luminarias_san_miguel.shp')

nodes.drop(columns='geometry').to_csv('indice_calidad_vial_nodos.csv', index=True)

print('✔ Índice de Calidad Vial generado')

# ==========================================
# 15. VISUALIZACIÓN RÁPIDA
# ==========================================
nodes.plot(column='ICV', cmap='viridis', legend=True)

# ==========================================
# 16. REFLEXIÓN
# ==========================================
print("""
Este índice es un modelo simplificado.

Desafíos para mejorar:
- Incorporar accidentes
- Usar centralidad real (betweenness)
- Reemplazar 'ruido' por una capa acústica real
- Reemplazar 'trafico_promedio' por aforos o TMDA real
- Validar cobertura de luminarias en OSM
- Incluir datos socioeconómicos
- Validar ponderaciones

Ojo:
- 'ruido' es una aproximación basada en estructura vial
- 'trafico_promedio' es un proxy derivado de highway, maxspeed y lanes
""")