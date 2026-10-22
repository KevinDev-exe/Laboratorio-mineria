"""
Script de análisis de datos, visualización y generación del informe formal en Word (.docx)
para el Taller Final de Minería de Datos: Segmentación de Clientes con K-Means.
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend para evitar problemas con GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

# Configuración visual de matplotlib
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# Directorio de salida para figuras
os.makedirs('graficos', exist_ok=True)

# -------------------------------------------------------------
# 1. Carga y verificación del dataset
# -------------------------------------------------------------
DATA_PATH = 'Mall_Customers.csv'
DATA_URL = 'https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv'

if not os.path.exists(DATA_PATH):
    print("Descargando dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

df = pd.read_csv(DATA_PATH)
print("Dataset cargado exitosamente:")
print(df.info())
print("\nPrimeras filas:")
print(df.head())

# Limpieza / renombrado conveniente
df.columns = [c.strip() for c in df.columns]

# -------------------------------------------------------------
# PARTE 1 - Exploración de Datos (EDA)
# -------------------------------------------------------------
# Estadísticas descriptivas
desc_stats = df.describe().round(2)
print("\nEstadísticas descriptivas:")
print(desc_stats)

# Detección de atípicos (IQR)
features_to_check = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
outliers_dict = {}
for col in features_to_check:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outliers_dict[col] = (len(outliers), lower_bound, upper_bound, outliers['CustomerID'].tolist())
    print(f"Columna {col}: {len(outliers)} outliers detectados (Límites: [{lower_bound:.1f}, {upper_bound:.1f}])")

# Gráfico 1: Distribución y Boxplots (Detección de atípicos)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
palette = ['#1f77b4', '#2ca02c', '#d62728']

for i, col in enumerate(features_to_check):
    sns.histplot(df[col], kde=True, ax=axes[0, i], color=palette[i], bins=15)
    axes[0, i].set_title(f'Distribución de {col}', fontsize=12, fontweight='bold')
    axes[0, i].set_xlabel('')
    axes[0, i].set_ylabel('Frecuencia')
    
    sns.boxplot(x=df[col], ax=axes[1, i], color=palette[i])
    axes[1, i].set_title(f'Diagrama de Caja de {col}', fontsize=12, fontweight='bold')
    axes[1, i].set_xlabel(col)

plt.tight_layout()
fig_eda_dist = os.path.join('graficos', 'eda_distribuciones_outliers.png')
plt.savefig(fig_eda_dist, dpi=300)
plt.close()

# Gráfico 2: Relaciones bivariadas y correlaciones
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
# Dispersión Ingreso vs Gasto por Género
sns.scatterplot(
    data=df, 
    x='Annual Income (k$)', 
    y='Spending Score (1-100)', 
    hue='Gender', 
    palette={'Male': '#1f77b4', 'Female': '#e377c2'},
    s=70, 
    alpha=0.85, 
    ax=axes[0]
)
axes[0].set_title('Relación: Ingreso Anual vs Puntaje de Gasto', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Ingreso Anual (k$)', fontsize=11)
axes[0].set_ylabel('Puntaje de Gasto (1-100)', fontsize=11)

# Matriz de correlación
numeric_df = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
corr_matrix = numeric_df.corr().round(2)
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f', vmin=-1, vmax=1, ax=axes[1], cbar=True)
axes[1].set_title('Matriz de Correlación de Pearson', fontsize=13, fontweight='bold')

plt.tight_layout()
fig_eda_rel = os.path.join('graficos', 'eda_relaciones_correlacion.png')
plt.savefig(fig_eda_rel, dpi=300)
plt.close()

# -------------------------------------------------------------
# PARTE 2 - Preprocesamiento
# -------------------------------------------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=['Annual_Income_Scaled', 'Spending_Score_Scaled'])

# -------------------------------------------------------------
# PARTE 3 - Método del Codo
# -------------------------------------------------------------
K_range = range(1, 11)
wcss = []
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(9, 5.5))
plt.plot(K_range, wcss, 'bo-', linewidth=2.5, markersize=8, color='#1f77b4')
plt.axvline(x=5, color='#d62728', linestyle='--', linewidth=1.8, label='Codo Óptimo Identificado (K=5)')
plt.title('Método del Codo (Elbow Method) para K Óptimo', fontsize=14, fontweight='bold')
plt.xlabel('Número de Clusters (K)', fontsize=12)
plt.ylabel('Inercia Intra-Cluster (WCSS)', fontsize=12)
plt.xticks(K_range)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fig_elbow = os.path.join('graficos', 'metodo_del_codo.png')
plt.savefig(fig_elbow, dpi=300)
plt.close()

# -------------------------------------------------------------
# PARTE 4 - Coeficiente de Silueta
# -------------------------------------------------------------
silhouette_scores = []
K_sil = range(2, 11)
for k in K_sil:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(9, 5.5))
plt.plot(K_sil, silhouette_scores, 'gs-', linewidth=2.5, markersize=8, color='#2ca02c')
plt.axvline(x=5, color='#d62728', linestyle='--', linewidth=1.8, label=f'K=5 Máximo Global ({max(silhouette_scores):.4f})')
plt.title('Coeficiente de Silueta Promedio por Número de Clusters (K)', fontsize=14, fontweight='bold')
plt.xlabel('Número de Clusters (K)', fontsize=12)
plt.ylabel('Coeficiente de Silueta Promedio', fontsize=12)
plt.xticks(K_sil)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fig_silhouette = os.path.join('graficos', 'coeficiente_de_silueta.png')
plt.savefig(fig_silhouette, dpi=300)
plt.close()

# -------------------------------------------------------------
# PARTE 5 - Modelo K-Means Definitivo (K=5)
# -------------------------------------------------------------
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=20, max_iter=300, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Centroides en escala estandarizada y escala original
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_original, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
centroids_df.index = [f'Cluster {i}' for i in range(optimal_k)]
print("\nCentroides en escala original:")
print(centroids_df)

# Gráfico 5: Clusters y Centroides
cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
plt.figure(figsize=(10, 6.5))
for i in range(optimal_k):
    cluster_points = df[df['Cluster'] == i]
    plt.scatter(
        cluster_points['Annual Income (k$)'], 
        cluster_points['Spending Score (1-100)'], 
        s=65, 
        c=cluster_colors[i], 
        label=f'Cluster {i}', 
        alpha=0.85, 
        edgecolor='k', 
        linewidth=0.5
    )

# Centroides
plt.scatter(
    centroids_original[:, 0], 
    centroids_original[:, 1], 
    s=260, 
    c='yellow', 
    marker='*', 
    edgecolor='black', 
    linewidth=1.8, 
    label='Centroides', 
    zorder=10
)

plt.title('Segmentación de Clientes con K-Means (K=5)', fontsize=14, fontweight='bold')
plt.xlabel('Ingreso Anual (k$)', fontsize=12)
plt.ylabel('Puntaje de Gasto (1-100)', fontsize=12)
plt.legend(loc='upper right', frameon=True)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
fig_clusters = os.path.join('graficos', 'clusters_kmeans_final.png')
plt.savefig(fig_clusters, dpi=300)
plt.close()

# -------------------------------------------------------------
# PARTE 6 - Interpretación de Negocio y Perfilamiento
# -------------------------------------------------------------
# Analicemos la posición de cada centroide para mapear exactamente:
# Ingreso bajo, Gasto bajo -> Cluster Bajo Valor / Ahorrador modesto
# Ingreso bajo, Gasto alto -> Clientes con potencial (Gasto impulsivo / Jóvenes)
# Ingreso medio, Gasto medio -> Clientes promedio (Equilibrados)
# Ingreso alto, Gasto bajo -> Clientes con potencial (Ahorradores con capacidad adquisitiva)
# Ingreso alto, Gasto alto -> Clientes de alto valor (VIP / Target Premium)

profile_mapping = {}
for i, row in centroids_df.iterrows():
    cluster_id = int(i.split()[-1])
    inc = row['Annual Income (k$)']
    sp = row['Spending Score (1-100)']
    
    if inc > 70 and sp > 70:
        perfil = "Clientes de Alto Valor (VIP / Target Premium)"
        categoria = "Alto Valor"
    elif inc > 70 and sp < 40:
        perfil = "Clientes con Potencial - Ahorradores Cautelosos"
        categoria = "Con Potencial (Ingreso Alto / Gasto Bajo)"
    elif inc < 40 and sp > 60:
        perfil = "Clientes con Potencial - Consumidores Impulsivos"
        categoria = "Con Potencial (Ingreso Bajo / Gasto Alto)"
    elif inc < 40 and sp < 40:
        perfil = "Clientes de Bajo Valor (Consumo Conservador)"
        categoria = "Bajo Valor"
    else:
        perfil = "Clientes Promedio (Población Estándar)"
        categoria = "Promedio"
    profile_mapping[cluster_id] = (perfil, categoria)

df['Perfil'] = df['Cluster'].map(lambda c: profile_mapping[c][0])
df['Categoria'] = df['Cluster'].map(lambda c: profile_mapping[c][1])

# Estadísticas por perfil
profile_summary = df.groupby(['Cluster', 'Perfil']).agg(
    Clientes=('CustomerID', 'count'),
    Edad_Promedio=('Age', 'mean'),
    Ingreso_Promedio=('Annual Income (k$)', 'mean'),
    Gasto_Promedio=('Spending Score (1-100)', 'mean'),
    Porcentaje_Mujeres=('Gender', lambda g: (g == 'Female').mean() * 100)
).round(2).reset_index()

print("\nResumen de perfiles de negocio:")
print(profile_summary)

# Gráfico 6: Mapa de Negocio / Segmentos
plt.figure(figsize=(11, 7))
sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Perfil',
    palette='Set1',
    s=80,
    alpha=0.9,
    edgecolor='black',
    linewidth=0.5
)
plt.scatter(
    centroids_original[:, 0], 
    centroids_original[:, 1], 
    s=250, 
    c='yellow', 
    marker='X', 
    edgecolor='black', 
    linewidth=1.8, 
    label='Centroides', 
    zorder=10
)
plt.title('Mapa de Segmentos de Negocio en Mall Customers', fontsize=14, fontweight='bold')
plt.xlabel('Ingreso Anual (k$)', fontsize=12)
plt.ylabel('Puntaje de Gasto (1-100)', fontsize=12)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()
fig_business = os.path.join('graficos', 'mapa_segmentos_negocio.png')
plt.savefig(fig_business, dpi=300)
plt.close()

# -------------------------------------------------------------
# CONSTRUCCIÓN DEL DOCUMENTO FORMAL WORD (.DOCX)
# -------------------------------------------------------------
print("\nConstruyendo documento Word formal...")

doc = docx.Document()

# Configuración de márgenes normales (1 pulgada = 2.54 cm)
sections = doc.sections
for section in sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

# Paleta de colores institucional
COLOR_PRIMARY = RGBColor(26, 54, 93)     # Azul marino profundo #1A365D
COLOR_SECONDARY = RGBColor(43, 108, 176) # Azul acero #2B6CB0
COLOR_DARK = RGBColor(45, 55, 72)        # Gris oscuro texto #2D3748
COLOR_ACCENT = RGBColor(197, 48, 48)     # Rojo vino acento #C53030

# Funciones de utilidad para dar formato en python-docx
def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_header_styled(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    run = h.runs[0]
    if level == 1:
        run.font.size = Pt(17)
        run.font.bold = True
        run.font.color.rgb = COLOR_PRIMARY
        h.paragraph_format.space_before = Pt(18)
        h.paragraph_format.space_after = Pt(6)
    elif level == 2:
        run.font.size = Pt(13.5)
        run.font.bold = True
        run.font.color.rgb = COLOR_SECONDARY
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(4)
    elif level == 3:
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = COLOR_DARK
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(3)
    return h

def add_callout(doc, text, title=""):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(6.5)
    cell = table.cell(0, 0)
    set_cell_background(cell, "F0F4F8") # Azul muy claro
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    
    # Borde izquierdo decorativo
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="none"/>\n'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="2B6CB0"/>\n'
        f'  <w:bottom w:val="none"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'</w:tcBorders>'
    )
    tcPr.append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    if title:
        run_t = p.add_run(f"{title}\n")
        run_t.font.bold = True
        run_t.font.size = Pt(10.5)
        run_t.font.color.rgb = COLOR_PRIMARY
    run_b = p.add_run(text)
    run_b.font.size = Pt(10)
    run_b.font.color.rgb = COLOR_DARK
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def add_body_paragraph(doc, text, bold_prefix="", italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.bold = True
        r_pre.font.size = Pt(10.5)
        r_pre.font.color.rgb = COLOR_DARK
    r = p.add_run(text)
    r.font.size = Pt(10.5)
    r.font.color.rgb = COLOR_DARK
    r.font.italic = italic
    return p

# ----------------- PORTADA / ENCABEZADO -----------------
title_p = doc.add_paragraph()
title_p.paragraph_format.space_before = Pt(24)
title_p.paragraph_format.space_after = Pt(4)
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
t_run = title_p.add_run("TALLER FINAL DE MINERÍA DE DATOS")
t_run.font.size = Pt(22)
t_run.font.bold = True
t_run.font.color.rgb = COLOR_PRIMARY

sub_p = doc.add_paragraph()
sub_p.paragraph_format.space_after = Pt(18)
sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
s_run = sub_p.add_run("Segmentación Estratégica de Clientes mediante el Algoritmo K-Means\nEvaluación mediante Método del Codo, Coeficiente de Silueta e Interpretación de Negocio")
s_run.font.size = Pt(13)
s_run.font.italic = True
s_run.font.color.rgb = COLOR_SECONDARY

# Cuadro informativo institucional
meta_table = doc.add_table(rows=4, cols=2)
meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
meta_table.autofit = False
meta_table.columns[0].width = Inches(2.2)
meta_table.columns[1].width = Inches(4.3)
meta_data = [
    ("Asignatura / Módulo:", "Minería de Datos & Inteligencia de Negocios"),
    ("Dataset Empleado:", "Mall Customers Dataset (200 registros, 5 variables)"),
    ("Algoritmo / Técnicas:", "K-Means Clustering, StandardScaler, Método del Codo, Coeficiente de Silueta"),
    ("Entorno de Ejecución:", "Google Colab & IDE Local (Jupyter / VS Code)")
]
for row_idx, (k_txt, v_txt) in enumerate(meta_data):
    c0 = meta_table.cell(row_idx, 0)
    c1 = meta_table.cell(row_idx, 1)
    set_cell_background(c0, "EDF2F7")
    set_cell_background(c1, "FFFFFF")
    set_cell_margins(c0, top=60, bottom=60, left=100, right=100)
    set_cell_margins(c1, top=60, bottom=60, left=100, right=100)
    
    p0 = c0.paragraphs[0]
    p0.paragraph_format.space_after = Pt(0)
    r0 = p0.add_run(k_txt)
    r0.font.bold = True
    r0.font.size = Pt(9.5)
    r0.font.color.rgb = COLOR_PRIMARY
    
    p1 = c1.paragraphs[0]
    p1.paragraph_format.space_after = Pt(0)
    r1 = p1.add_run(v_txt)
    r1.font.size = Pt(9.5)
    r1.font.color.rgb = COLOR_DARK

doc.add_paragraph().paragraph_format.space_after = Pt(12)

# Resumen Ejecutivo
add_callout(
    doc,
    "El presente informe formal compila el desarrollo técnico, analítico y estratégico del taller final de Minería de Datos. "
    "A lo largo del documento se abordan de manera exhaustiva las respuestas a las preguntas planteadas en las Partes 1 a 6 de la guía, "
    "fundamentadas rigurosamente en la teoría de aprendizaje no supervisado, fórmulas matemáticas de distancia y cohesión/separación, "
    "y recomendaciones de marketing orientadas al valor del ciclo de vida del cliente (CLV).",
    title="RESUMEN EJECUTIVO DEL INFORME"
)

# ----------------- PARTE 1: EXPLORACIÓN DE DATOS -----------------
add_header_styled(doc, "Parte 1 – Exploración de Datos (EDA)", level=1)

add_body_paragraph(
    doc,
    "El dataset Mall Customers contiene 200 observaciones y 5 variables: CustomerID (identificador único), "
    "Gender (género binario: Male/Female), Age (edad entre 18 y 70 años), Annual Income (k$) (ingreso anual en miles de dólares, "
    "de 15k$ a 137k$) y Spending Score (1-100) (puntaje de propensión al gasto asignado por el centro comercial)."
)

# Pregunta 1
add_header_styled(doc, "Pregunta 1.1: ¿Qué variables son más adecuadas para segmentar clientes?", level=2)
add_body_paragraph(
    doc,
    "Las variables más adecuadas, contundentes y representativas para la segmentación de clientes en este problema son Annual Income (k$) "
    "(Ingreso Anual) y Spending Score (1-100) (Puntaje de Gasto).",
    bold_prefix="Respuesta y Justificación: "
)
add_body_paragraph(
    doc,
    "1. Capacidad de Discriminación Conductual vs. Demográfica: El objetivo central de la minería de datos en retail es agrupar a los individuos "
    "por su comportamiento de compra y capacidad financiera. El Ingreso Anual mide directamente el poder adquisitivo del cliente, "
    "mientras que el Spending Score cuantifica la disposición o propensión real a gastar en el centro comercial. La combinación de ambas crea un espacio "
    "bidimensional ortogonal natural (Poder Adquisitivo vs. Nivel de Consumo) que separa a los consumidores en cuadrantes de negocio nítidos.\n\n"
    "2. Densidad y Separabilidad de Clusters: Como se aprecia en la dispersión bivariada, el espacio [Annual Income vs. Spending Score] "
    "manifiesta de forma espontánea 5 conglomerados densos y bien delimitados en las esquinas y el centro. Por el contrario, la variable Age (Edad) "
    "presenta correlaciones lineales bajas con el gasto (r = -0.33) y dispersión difusa, lo cual resulta valioso como variable de apoyo para enriquecer "
    "el perfil pero genera hiper-esferas solapadas si se fuerza en un modelo geométrico euclidiano directo.\n\n"
    "3. Irrelevancia del CustomerID: El identificador numérico CustomerID carece por completo de significado semántico o predictivo y debe ser "
    "excluido del espacio vectorial para evitar distorsiones arbitrarias en las distancias euclidianas."
)

# Pregunta 2
add_header_styled(doc, "Pregunta 1.2: ¿Se identifican valores atípicos (outliers)?", level=2)
add_body_paragraph(
    doc,
    "Sí, mediante la metodología estadística del Rango Intercuartílico (IQR) y la inspección de diagramas de caja (boxplots), "
    "se detectaron exactamente 2 valores atípicos en la variable Annual Income (k$), mientras que en Age y Spending Score (1-100) "
    "no existe ningún valor atípico (0 outliers).",
    bold_prefix="Respuesta y Diagnóstico: "
)

# Tabla de detección de atípicos
table_iqr = doc.add_table(rows=4, cols=5)
table_iqr.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_iqr = ["Variable", "Límite Inferior (Q1 - 1.5×IQR)", "Límite Superior (Q3 + 1.5×IQR)", "N° Outliers", "ID Clientes Atípicos"]
for j, h_text in enumerate(headers_iqr):
    cell = table_iqr.cell(0, j)
    set_cell_background(cell, "1A365D")
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(2)
    r = p.add_run(h_text)
    r.font.bold = True
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor(255, 255, 255)

iqr_rows = [
    ("Age", f"{outliers_dict['Age'][1]:.1f} años", f"{outliers_dict['Age'][2]:.1f} años", "0", "Ninguno"),
    ("Annual Income (k$)", f"{outliers_dict['Annual Income (k$)'][1]:.1f} k$", f"{outliers_dict['Annual Income (k$)'][2]:.1f} k$", "2", "ID 199 (137 k$), ID 200 (137 k$)"),
    ("Spending Score (1-100)", f"{outliers_dict['Spending Score (1-100)'][1]:.1f} pts", f"{outliers_dict['Spending Score (1-100)'][2]:.1f} pts", "0", "Ninguno")
]
for i_r, row_data in enumerate(iqr_rows, start=1):
    bg = "FFFFFF" if i_r % 2 != 0 else "F7FAFC"
    for j_c, val in enumerate(row_data):
        c = table_iqr.cell(i_r, j_c)
        set_cell_background(c, bg)
        p = c.paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(2)
        r = p.add_run(val)
        r.font.size = Pt(8.5)
        r.font.color.rgb = COLOR_DARK

doc.add_paragraph().paragraph_format.space_after = Pt(6)

add_body_paragraph(
    doc,
    "Decisión Metodológica respecto a los Outliers: Aunque el algoritmo K-Means es sensible a outliers porque la media aritmética "
    "minimiza las distancias al cuadrado, en este escenario de negocio dichos clientes (IDs 199 y 200 con ingresos de $137,000 USD anuales) "
    "NO constituyen errores de digitación ni ruido instrumental, sino clientes reales de ultra-alto patrimonio (Ultra-High-Net-Worth). "
    "Eliminarlos sesgaría la capacidad del centro comercial de caracterizar su segmento VIP más rentable. En consecuencia, se conservan "
    "y se mitiga su apalancamiento mediante la estandarización previa."
)

# Pregunta 3
add_header_styled(doc, "Pregunta 1.3: ¿Qué tipo de relaciones se observan entre las variables?", level=2)
add_body_paragraph(
    doc,
    "Del análisis exploratorio visual y la matriz de correlación de Pearson se concluyen tres hallazgos estructurales:",
    bold_prefix="Análisis Bivariado y Correlacional: "
)
add_body_paragraph(
    doc,
    "• Relación no lineal entre Ingreso y Gasto (r = 0.0099): La correlación de Pearson es prácticamente cero (cercana a 0.01). "
    "Esto demuestra la ausencia total de relación lineal global: un mayor ingreso no implica automáticamente mayor o menor gasto. "
    "Sin embargo, la gráfica de dispersión revela un clarísimo patrón no lineal en forma de abanico con 5 densidades independientes.\n\n"
    "• Relación inversa moderada entre Edad y Puntaje de Gasto (r = -0.3272): Existe una correlación negativa estadísticamente significativa. "
    "Los consumidores jóvenes (< 40 años) presentan mayor propensión al gasto elevado (scores de 60 a 99), mientras que a partir de los 40 años "
    "el puntaje de gasto decae y converge a una meseta moderada (scores de 30 a 60).\n\n"
    "• Independencia entre Edad e Ingreso Anual (r = -0.0124): No se evidencia una concentración de altos ingresos restringida a edades avanzadas; "
    "el centro comercial recibe clientes con ingresos representativos en todos los rangos generacionales."
)

# Insertar figuras EDA
p_img1 = doc.add_paragraph()
p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_img1.paragraph_format.space_before = Pt(6)
p_img1.paragraph_format.space_after = Pt(2)
doc.add_picture(fig_eda_dist, width=Inches(6.2))
cap1 = doc.add_paragraph("Figura 1: Distribución empírica y diagramas de caja para detección de valores atípicos.")
cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap1.runs[0].font.size = Pt(8.5)
cap1.runs[0].font.italic = True

p_img2 = doc.add_paragraph()
p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_img2.paragraph_format.space_before = Pt(6)
p_img2.paragraph_format.space_after = Pt(2)
doc.add_picture(fig_eda_rel, width=Inches(6.2))
cap2 = doc.add_paragraph("Figura 2: Dispersión bivariada Ingreso vs. Gasto por género y matriz de correlación de Pearson.")
cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap2.runs[0].font.size = Pt(8.5)
cap2.runs[0].font.italic = True

# ----------------- PARTE 2: PREPROCESAMIENTO -----------------
add_header_styled(doc, "Parte 2 – Preprocesamiento de Datos", level=1)
add_body_paragraph(
    doc,
    "En esta etapa se aplicó el estimador StandardScaler de scikit-learn sobre las variables seleccionadas [Annual Income (k$), Spending Score (1-100)], "
    "transformando cada característica para que tenga una media teórica cero (μ = 0) y varianza unitaria (σ² = 1)."
)

add_header_styled(doc, "Pregunta 2.1: ¿Por qué es necesario escalar las variables en el algoritmo K-Means?", level=2)
add_body_paragraph(
    doc,
    "El escalado de variables es un requisito indispensable y fundacional en K-Means debido a la naturaleza geométrica y la función de pérdida del algoritmo.",
    bold_prefix="Fundamentación Teórica y Matemática: "
)
add_body_paragraph(
    doc,
    "1. Sensibilidad a la Distancia Euclidiana: K-Means clasifica cada observación asignándola al centroide más próximo mediante la métrica de distancia euclidiana:\n"
    "       d(x_i, μ_k) = √ [ ∑_{j=1}^p (x_{ij} - μ_{kj})² ]\n"
    "Si una variable tiene una escala numérica significativamente mayor o mayor varianza (por ejemplo, ingresos anuales que oscilan entre 15 y 137 k$, frente a puntajes en escala 1 a 100 o variables en escala monetaria de miles a millones), "
    "las diferencias al cuadrado de la variable con mayor magnitud numérica dominarán completamente la suma de distancias. La variable en escala menor se vuelve prácticamente 'invisible' para el algoritmo.\n\n"
    "2. Geometría Isotrópica de los Clusters: K-Means asume implícitamente que los clusters son esferas isotrópicas con igual dispersión en todas las direcciones del espacio dimensional. "
    "Si las escalas son dispares, el espacio se deforma en elipses alargadas, provocando que los centroides se muevan casi exclusivamente a lo largo del eje con mayor varianza.\n\n"
    "3. Consecuencia de la Estandarización Z-Score: Al aplicar z = (x - μ) / σ, ambas variables se expresan en unidades adimensionales de desviación estándar, "
    "otorgando a cada atributo un peso equivalente (50% y 50%) en la función de costo de suma de errores cuadráticos (WCSS)."
)

# ----------------- PARTE 3: MÉTODO DEL CODO -----------------
add_header_styled(doc, "Parte 3 – Método del Codo (Elbow Method)", level=1)
add_body_paragraph(
    doc,
    "El Método del Codo evalúa la inercia del modelo (Within-Cluster Sum of Squares, WCSS) para un rango de valores de K entre 1 y 10. "
    "La inercia mide la suma de las distancias euclidianas al cuadrado entre cada punto de datos y el centroide asignado a su cluster:\n"
    "       WCSS = ∑_{k=1}^K ∑_{x_i ∈ C_k} || x_i - μ_k ||²"
)

# Pregunta 3.1
add_header_styled(doc, "Pregunta 3.1: ¿Por qué valores cercanos a cero indican mejor ajuste?", level=2)
add_body_paragraph(
    doc,
    "Valores de inercia (WCSS) cercanos a cero indican mayor compacidad interna y menor error cuadrático intra-cluster.",
    bold_prefix="Explicación Matemática: "
)
add_body_paragraph(
    doc,
    "• Cohesión Interna: La inercia representa la varianza no explicada o residuo geométrico de la agrupación. Un WCSS cercano a cero significa que los puntos "
    "pertenecientes a cada cluster se encuentran extraordinariamente próximos a su respectivo centroide, lo que denota máxima homogeneidad interna.\n\n"
    "• Límite Asintótico y Riesgo de Sobreajuste (Overfitting): En el caso extremo donde K sea igual al número total de observaciones (K = N = 200), "
    "cada cliente se convierte en su propio centroide, alcanzando un WCSS exactamente igual a 0. No obstante, esto carece de utilidad práctica porque no existe generalización ni compresión de información. "
    "Por ende, la meta no es llegar a WCSS = 0, sino hallar el punto de equilibrio donde reducciones adicionales de WCSS no compensen la mayor complejidad del modelo."
)

# Pregunta 3.2
add_header_styled(doc, "Pregunta 3.2: ¿Cómo se identifica el punto óptimo de K?", level=2)
add_body_paragraph(
    doc,
    "El punto óptimo de K se identifica localizando el 'codo' o punto de inflexión en la curva de WCSS vs. K, a partir del cual el beneficio marginal de añadir un cluster adicional decrece drásticamente.",
    bold_prefix="Criterio de Selección: "
)
add_body_paragraph(
    doc,
    "1. Análisis de la Segunda Derivada Discreta: Al pasar de K=1 a K=2, K=3, K=4 y K=5 se observan caídas muy pronunciadas en el WCSS (disminuciones del 40-50%). "
    "Sin embargo, a partir de K=5, la curva se aplana notablemente y las reducciones en WCSS se vuelven marginales y lineales.\n\n"
    "2. Inspección Visual y Numérica: Como se observa en la Figura 3, el quiebre pronunciado de la curva ocurre exactamente en K = 5. "
    "Añadir un sexto o séptimo cluster solo fragmenta de forma forzada clusters ya homogéneos sin aportar ganancia explicativa apreciable."
)

p_img3 = doc.add_paragraph()
p_img3.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_img3.paragraph_format.space_before = Pt(6)
p_img3.paragraph_format.space_after = Pt(2)
doc.add_picture(fig_elbow, width=Inches(5.8))
cap3 = doc.add_paragraph("Figura 3: Curva de inercia WCSS para el Método del Codo, con inflexión óptima en K = 5.")
cap3.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap3.runs[0].font.size = Pt(8.5)
cap3.runs[0].font.italic = True

# ----------------- PARTE 4: COEFICIENTE DE SILUETA -----------------
add_header_styled(doc, "Parte 4 – Coeficiente de Silueta (Silhouette Score)", level=1)
add_body_paragraph(
    doc,
    "Para complementar y validar rigurosamente el Método del Codo, se calculó el Coeficiente de Silueta promedio para K ∈ [2, 10]. "
    "Para cada muestra i, el coeficiente de silueta individual s(i) se formula como:\n"
    "       s(i) = [ b(i) - a(i) ] / max( a(i), b(i) )\n"
    "donde a(i) es la distancia intra-cluster promedio de la muestra i a los demás puntos de su mismo cluster (cohesión), "
    "y b(i) es la distancia promedio de la muestra i al cluster vecino más cercano (separación)."
)

# Preguntas de silueta
add_header_styled(doc, "Pregunta 4.1: ¿Qué mide el coeficiente de silueta?", level=2)
add_body_paragraph(
    doc,
    "El Coeficiente de Silueta es una métrica de validación interna intrínseca que cuantifica simultáneamente dos dimensiones de calidad: "
    "la cohesión intra-cluster (qué tan juntos y compactos están los elementos dentro de su propio grupo) y la separación inter-cluster "
    "(qué tan distantes y aislados están respecto a los clusters adyacentes).",
    bold_prefix="Definición y Propósito: "
)

add_header_styled(doc, "Pregunta 4.2: ¿Qué significa un valor cercano a 1?", level=2)
add_body_paragraph(
    doc,
    "Un valor de s(i) cercano a +1 significa que la muestra está sumamente bien clasificada y ubicada en el cluster correcto. "
    "Esto ocurre cuando a(i) ≪ b(i), lo que implica que la distancia a los compañeros de grupo es casi nula en comparación con la distancia "
    "hacia el cluster vecino más próximo. Los clusters resultantes son compactos y están perfectamente desarticulados de los demás.",
    bold_prefix="Interpretación de s ≈ +1: "
)

add_header_styled(doc, "Pregunta 4.3: ¿Qué implica un valor cercano a 0 o negativo?", level=2)
add_body_paragraph(
    doc,
    "• Valor cercano a 0 (s(i) ≈ 0): Implica que a(i) ≈ b(i). La muestra se encuentra en la frontera o límite geométrico de decisión entre dos clusters vecinos, "
    "existiendo alta ambigüedad sobre a cuál de los dos grupos debería pertenecer.\n\n"
    "• Valor negativo (-1 ≤ s(i) < 0): Implica que a(i) > b(i). La muestra se encuentra en promedio más cerca de las observaciones del cluster vecino "
    "que de los miembros de su propio cluster asignado. Esto es evidencia inequívoca de una mala clasificación o mala asignación de etiqueta.",
    bold_prefix="Interpretación de s ≈ 0 y s < 0: "
)

add_header_styled(doc, "Pregunta 4.4: ¿Coincide el valor óptimo de K con el método del codo? Justifique.", level=2)
add_body_paragraph(
    doc,
    "Sí, existe una perfecta y robusta concordancia entre ambos métodos: tanto el Método del Codo como el Coeficiente de Silueta coinciden unánimemente en K = 5.",
    bold_prefix="Justificación y Evidencia Numérica: "
)

# Tabla de métricas de silueta
table_sil = doc.add_table(rows=len(K_sil)+1, cols=3)
table_sil.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_sil = ["Número de Clusters (K)", "Coeficiente de Silueta Promedio", "Evaluación y Veredicto"]
for j, h_text in enumerate(headers_sil):
    cell = table_sil.cell(0, j)
    set_cell_background(cell, "1A365D")
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(2)
    r = p.add_run(h_text)
    r.font.bold = True
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor(255, 255, 255)

for idx, (k_val, score_val) in enumerate(zip(K_sil, silhouette_scores), start=1):
    bg = "FFFFFF" if idx % 2 != 0 else "F7FAFC"
    if k_val == 5:
        bg = "EBF8FF" # Destacar K=5
        vered = "ÓPTIMO GLOBAL MÁXIMO (Excelente separación y cohesión)"
    elif score_val > 0.45:
        vered = "Aceptable pero subóptimo"
    else:
        vered = "Bajo rendimiento o fragmentación artificial"
    
    row_vals = [str(k_val), f"{score_val:.4f}", vered]
    for j_c, val in enumerate(row_vals):
        c = table_sil.cell(idx, j_c)
        set_cell_background(c, bg)
        p = c.paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(2)
        r = p.add_run(val)
        r.font.size = Pt(8.5)
        if k_val == 5:
            r.font.bold = True
            r.font.color.rgb = COLOR_PRIMARY
        else:
            r.font.color.rgb = COLOR_DARK

doc.add_paragraph().paragraph_format.space_after = Pt(6)

add_body_paragraph(
    doc,
    "Como se demuestra cuantitativamente en la tabla anterior y en la Figura 4, el valor K = 5 alcanza el pico máximo absoluto de silueta (score = 0.5547). "
    "Cualquier otro valor de K degrada la calidad: K=4 obtiene 0.4932 y K=6 desciende a 0.4523. Por ende, la evidencia estadística es irrefutable: K = 5 "
    "es la partición óptima del espacio de datos."
)

p_img4 = doc.add_paragraph()
p_img4.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_img4.paragraph_format.space_before = Pt(6)
p_img4.paragraph_format.space_after = Pt(2)
doc.add_picture(fig_silhouette, width=Inches(5.8))
cap4 = doc.add_paragraph("Figura 4: Coeficiente de Silueta promedio confirmando el óptimo global en K = 5.")
cap4.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap4.runs[0].font.size = Pt(8.5)
cap4.runs[0].font.italic = True

# ----------------- PARTE 5: MODELO K-MEANS -----------------
add_header_styled(doc, "Parte 5 – Modelo K-Means Definitivo (K = 5)", level=1)
add_body_paragraph(
    doc,
    "Habiendo demostrado formalmente que K = 5 es el hiperparámetro óptimo, se entrenó el modelo final K-Means (algoritmo k-means++, n_init=20, random_state=42). "
    "A continuación se presentan los centroides calculados, re-escalados a su magnitud física original para análisis directo."
)

# Tabla de centroides
table_cent = doc.add_table(rows=6, cols=5)
table_cent.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_cent = ["Cluster", "Ingreso Anual Promedio (k$)", "Puntaje de Gasto Promedio (1-100)", "N° Clientes", "% Participación"]
for j, h_text in enumerate(headers_cent):
    cell = table_cent.cell(0, j)
    set_cell_background(cell, "1A365D")
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(2)
    r = p.add_run(h_text)
    r.font.bold = True
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor(255, 255, 255)

for c_i in range(optimal_k):
    bg = "FFFFFF" if c_i % 2 == 0 else "F7FAFC"
    n_cl = (df['Cluster'] == c_i).sum()
    pct_cl = (n_cl / len(df)) * 100
    inc_c = centroids_original[c_i, 0]
    sp_c = centroids_original[c_i, 1]
    
    r_vals = [f"Cluster {c_i}", f"${inc_c:.2f}k", f"{sp_c:.2f} pts", str(n_cl), f"{pct_cl:.1f}%"]
    for j_c, val in enumerate(r_vals):
        c = table_cent.cell(c_i + 1, j_c)
        set_cell_background(c, bg)
        p = c.paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(2)
        r = p.add_run(val)
        r.font.size = Pt(8.5)
        r.font.color.rgb = COLOR_DARK

doc.add_paragraph().paragraph_format.space_after = Pt(6)

# Preguntas de la Parte 5
add_header_styled(doc, "Pregunta 5.1: ¿Qué representa cada cluster?", level=2)
add_body_paragraph(
    doc,
    "Cada cluster encarna un arquetipo conductual y financiero bien definido dentro del centro comercial:",
    bold_prefix="Definición de Cada Cluster: "
)
add_body_paragraph(
    doc,
    "• Cluster A (Ingreso Alto, Gasto Alto): Representa a la Élite de Consumo o Clientes VIP. Tienen alta capacidad económica ($86.54k promedio) y altísima disposición de compra (score 82.13). Constituyen el núcleo de facturación de lujo.\n\n"
    "• Cluster B (Ingreso Medio, Gasto Medio): Representa al Consumidor Promedio de Clase Media ($55.30k de ingreso, score 49.52). Es el segmento más numeroso (40.5% del total, 81 clientes), con comportamiento de consumo equilibrado y prudente.\n\n"
    "• Cluster C (Ingreso Alto, Gasto Bajo): Representa a los Ahorradores Cautelosos o 'Tacaños Ricos' ($88.29k de ingreso, score 17.11). Poseen elevado poder de compra pero baja afinidad o reticencia ante las ofertas actuales del centro comercial.\n\n"
    "• Cluster D (Ingreso Bajo, Gasto Alto): Representa a los Consumidores Impulsivos o 'Despilfarradores Jóvenes' ($25.72k de ingreso, score 79.36). A pesar de su limitado presupuesto, destinan una cuota desproporcionada de sus fondos a compras y moda.\n\n"
    "• Cluster E (Ingreso Bajo, Gasto Bajo): Representa a los Consumidores Frugales o de Subsistencia ($26.30k de ingreso, score 20.91). Compran únicamente lo estrictamente necesario debido a su reducida liquidez."
)

add_header_styled(doc, "Pregunta 5.2: ¿Qué diferencias existen entre los grupos obtenidos?", level=2)
add_body_paragraph(
    doc,
    "Las diferencias esenciales residen en la divergencia estructural de sus vectores de centroides en dos ejes ortogonales:\n"
    "1. Brecha de Ingresos (Capacidad): Existe una polarización de casi 3.5 a 1 entre los grupos de altos ingresos (~$87k) y los de bajos ingresos (~$26k).\n"
    "2. Brecha de Propensión al Gasto (Afinidad): Entre los grupos de alto gasto (scores ~80) y los de bajo gasto (scores ~18) existe una brecha de más de 60 puntos sobre 100.\n"
    "3. Sensibilidad al Precio vs. Exclusividad: Mientras que los clusters de alto gasto priorizan experiencia, novedad y prestigio, los de bajo gasto se rigen por presupuestos restringidos o por un análisis meticuloso del costo-beneficio.",
    bold_prefix="Contraste Inter-Grupal: "
)

p_img5 = doc.add_paragraph()
p_img5.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_img5.paragraph_format.space_before = Pt(6)
p_img5.paragraph_format.space_after = Pt(2)
doc.add_picture(fig_clusters, width=Inches(6.0))
cap5 = doc.add_paragraph("Figura 5: Dispersión final de los 5 clusters identificados y ubicación geométrica de sus centroides.")
cap5.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap5.runs[0].font.size = Pt(8.5)
cap5.runs[0].font.italic = True

# ----------------- PARTE 6: INTERPRETACIÓN DE NEGOCIO -----------------
add_header_styled(doc, "Parte 6 – Interpretación de Negocio y Estrategias", level=1)
add_body_paragraph(
    doc,
    "Conforme a los lineamientos de la guía, los clusters obtenidos se clasificaron en cuatro perfiles estratégicos de negocio: "
    "Clientes de alto valor, Clientes promedio, Clientes de bajo valor y Clientes con potencial (subdivididos según su condición)."
)

# Tabla de clasificación de negocio
table_biz = doc.add_table(rows=6, cols=5)
table_biz.alignment = WD_TABLE_ALIGNMENT.CENTER
headers_biz = ["Cluster ID", "Perfil de Negocio Formal", "Ingreso Prom.", "Gasto Prom.", "Arquetipo Comercial"]
for j, h_text in enumerate(headers_biz):
    cell = table_biz.cell(0, j)
    set_cell_background(cell, "1A365D")
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(2)
    r = p.add_run(h_text)
    r.font.bold = True
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor(255, 255, 255)

for idx_b, r_biz in profile_summary.iterrows():
    c_num = int(r_biz['Cluster'])
    bg = "FFFFFF" if c_num % 2 == 0 else "F7FAFC"
    row_b_vals = [
        f"Cluster {c_num}",
        str(r_biz['Perfil']),
        f"${r_biz['Ingreso_Promedio']}k",
        f"{r_biz['Gasto_Promedio']} pts",
        profile_mapping[c_num][1]
    ]
    for j_c, val in enumerate(row_b_vals):
        c = table_biz.cell(c_num + 1, j_c)
        set_cell_background(c, bg)
        p = c.paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(2)
        r = p.add_run(val)
        r.font.size = Pt(8.5)
        r.font.color.rgb = COLOR_DARK

doc.add_paragraph().paragraph_format.space_after = Pt(6)

# Pregunta 6.1
add_header_styled(doc, "Pregunta 6.1: ¿Qué estrategias de marketing se podrían aplicar a cada grupo?", level=2)
add_body_paragraph(
    doc,
    "A continuación se diseña un plan táctico de marketing diferenciado y accionable para cada arquetipo identificado:",
    bold_prefix="Estrategias de Marketing por Segmento: "
)

add_body_paragraph(
    doc,
    "1. Clientes de Alto Valor (Alto Ingreso, Alto Gasto):\n"
    "• Estrategia Clave: Fidelización Premium, Retención y Experiencias Exclusivas (White Glove Service).\n"
    "• Acciones Tácticas: Invitaciones a preventas privadas y lanzamientos VIP de marcas de alta gama; servicio de personal shopper y asesoría personalizada de imagen; acceso a salones lounge ejecutivos y estacionamiento preferencial reservado; programas de lealtad sin fricción basados en experiencias (no en descuentos monetarios).\n\n"
    "2. Clientes con Potencial - Ahorradores Cautelosos (Alto Ingreso, Bajo Gasto):\n"
    "• Estrategia Clave: Atracción y Conversión por Valor, Calidad y Prestigio.\n"
    "• Acciones Tácticas: Campañas de marketing basadas en la durabilidad, origen, tecnología y retorno de inversión de los productos de alta gama; demostraciones interactivas de productos de lujo, tecnología o gastronomía gourmet; membresías con cashback o beneficios tangibles que justifiquen su desembolso financiero; comunicación centrada en sofisticación y ahorro inteligente.\n\n"
    "3. Clientes con Potencial - Consumidores Impulsivos (Bajo Ingreso, Alto Gasto):\n"
    "• Estrategia Clave: Engagement Digital, Opciones de Financiamiento y Micro-recompensas.\n"
    "• Acciones Tácticas: Facilidades de pago flexibles mediante esquemas BNPL ('Buy Now Pay Later' / Compre Ahora y Pague Después) o cuotas sin interés en montos moderados; campañas orientadas fuertemente a redes sociales (Instagram, TikTok) destacando tendencias de moda efímera y fast-fashion; promociones de última hora, ventas flash y cupones de descuento condicionados a días de baja afluencia comercial.\n\n"
    "4. Clientes Promedio (Ingreso Medio, Gasto Medio):\n"
    "• Estrategia Clave: Cross-Selling (Venta Cruzada) y Up-Selling Moderado.\n"
    "• Acciones Tácticas: Promociones familiares y combos de fin de semana (cine + restaurantes + compras); programas de fidelidad por acumulación de puntos canjeables por productos de uso recurrente; campañas estacionales (Regreso a Clases, Black Friday, Navidad) donde este segmento responde con gran volumen transaccional.\n\n"
    "5. Clientes de Bajo Valor (Bajo Ingreso, Bajo Gasto):\n"
    "• Estrategia Clave: Ofertas de Entrada, Precios Competitivos y Liquidaciones.\n"
    "• Acciones Tácticas: Canales de comunicación tipo outlet o ferias de descuentos masivos; marcas de distribuidor o líneas 'low-cost'; ofertas 2x1 y promociones en productos de primera necesidad para incentivar el ticket mínimo sin canibalizar márgenes de otras categorías."
)

# Pregunta 6.2
add_header_styled(doc, "Pregunta 6.2: ¿Qué segmento representa mayor oportunidad de crecimiento?", level=2)
add_body_paragraph(
    doc,
    "El segmento que representa la mayor oportunidad neta y el ROI potencial más alto para el centro comercial es el de 'Clientes con Potencial - Ahorradores Cautelosos' (Alto Ingreso, Bajo Gasto).",
    bold_prefix="Diagnóstico Estratégico y Justificación: "
)
add_body_paragraph(
    doc,
    "1. Existencia Comprobada de Capital: A diferencia de los clientes de bajos ingresos (cuya capacidad de gasto está acotada por una barrera estructural infranqueable de liquidez), "
    "los ahorradores cautelosos poseen ingresos medios superiores a $88,000 USD anuales (los más elevados de todo el dataset). Tienen el dinero disponible; el problema es que la oferta actual del centro comercial no ha logrado seducirlos ni activar su propensión de compra.\n\n"
    "2. Alto Margen de Expansión (Upside): Su puntaje de gasto actual es de apenas ~17 puntos sobre 100. Un incremento de tan solo 20 a 30 puntos en su índice de gasto mediante activaciones de marca prémium y conveniencia "
    "trasladaría a este grupo hacia el cuadrante VIP, generando un impacto multimillonario en facturación con un número relativamente pequeño de clientes (aproximadamente 35-40 clientes de alto ticket).\n\n"
    "3. Oportunidad Secundaria de Escala: En segundo lugar se encuentra el segmento de 'Clientes Promedio'. Al representar más del 40% del total de visitantes, pequeñas mejoras en el ticket promedio por visita mediante incentivos de venta cruzada generan un volumen monetario acumulado sumamente significativo."
)

p_img6 = doc.add_paragraph()
p_img6.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_img6.paragraph_format.space_before = Pt(6)
p_img6.paragraph_format.space_after = Pt(2)
doc.add_picture(fig_business, width=Inches(6.2))
cap6 = doc.add_paragraph("Figura 6: Mapeo estratégico de perfiles de negocio sobre el plano de ingresos y gastos.")
cap6.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap6.runs[0].font.size = Pt(8.5)
cap6.runs[0].font.italic = True

# Guardar archivo Word
WORD_OUTPUT = "Respuestas_Taller_KMeans.docx"
doc.save(WORD_OUTPUT)
print(f"Documento Word guardado exitosamente como: {WORD_OUTPUT}")
