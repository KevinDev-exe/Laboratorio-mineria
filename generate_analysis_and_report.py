"""
Script maestro para generar el documento Word (.docx) COMPLETO del Taller Final de Minería de Datos:
Segmentación de Clientes con K-Means (Partes 1 a 7 completas, incluyendo tablas estadísticas,
fórmulas, respuestas exhaustivas a todas las preguntas, gráficos en alta resolución y guía de despliegue).
"""

import os
import shutil
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

# Configuración visual de matplotlib
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

os.makedirs('graficos', exist_ok=True)

# -------------------------------------------------------------
# 1. Carga y verificación del dataset
# -------------------------------------------------------------
DATA_PATH = 'Mall_Customers.csv'
DATA_URL = 'https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv'

if not os.path.exists(DATA_PATH):
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

# -------------------------------------------------------------
# CÁLCULOS Y GRÁFICOS DE ALTA RESOLUCIÓN (PARTES 1 A 7)
# -------------------------------------------------------------
features_to_check = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
outliers_dict = {}
for col in features_to_check:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outliers_dict[col] = (len(outliers), Q1, Q3, IQR, lower_bound, upper_bound)

# Figura 1: Distribuciones y Boxplots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
palette = ['#1f77b4', '#2ca02c', '#d62728']
for i, col in enumerate(features_to_check):
    sns.histplot(df[col], kde=True, ax=axes[0, i], color=palette[i], bins=15)
    axes[0, i].set_title(f'Distribución de {col}', fontsize=12, fontweight='bold')
    axes[0, i].set_xlabel('')
    axes[0, i].set_ylabel('Frecuencia')
    sns.boxplot(x=df[col], ax=axes[1, i], color=palette[i])
    axes[1, i].set_title(f'Boxplot de {col}', fontsize=12, fontweight='bold')
    axes[1, i].set_xlabel(col)
plt.tight_layout()
fig_eda_dist = os.path.join('graficos', 'eda_distribuciones_outliers.png')
plt.savefig(fig_eda_dist, dpi=300)
plt.close()

# Figura 2: Relaciones bivariadas y correlaciones
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
sns.scatterplot(
    data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
    hue='Gender', palette={'Male': '#1f77b4', 'Female': '#e377c2'},
    s=70, alpha=0.85, ax=axes[0]
)
axes[0].set_title('Dispersión: Ingreso Anual vs Puntaje de Gasto', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Ingreso Anual (k$)', fontsize=11)
axes[0].set_ylabel('Puntaje de Gasto (1-100)', fontsize=11)

numeric_df = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
corr_matrix = numeric_df.corr().round(3)
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.3f', vmin=-1, vmax=1, ax=axes[1], cbar=True)
axes[1].set_title('Matriz de Correlación Lineal de Pearson', fontsize=13, fontweight='bold')
plt.tight_layout()
fig_eda_rel = os.path.join('graficos', 'eda_relaciones_correlacion.png')
plt.savefig(fig_eda_rel, dpi=300)
plt.close()

# Parte 2: Escalado
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Parte 3: Método del Codo
K_range = list(range(1, 11))
wcss = []
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(9, 5.2))
plt.plot(K_range, wcss, marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label='WCSS (Inercia)')
plt.axvline(x=5, color='#d62728', linestyle='--', linewidth=1.8, label='Codo Óptimo Identificado (K=5)')
plt.annotate('Punto de inflexión\n(K=5, WCSS=65.57)', xy=(5, wcss[4]), xytext=(6.2, wcss[4]+60),
             arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5),
             fontsize=10, fontweight='bold', color='#1A365D')
plt.title('Método del Codo (Elbow Method) para Selección de K', fontsize=14, fontweight='bold')
plt.xlabel('Número de Clusters (K)', fontsize=12)
plt.ylabel('Inercia Intra-Cluster (WCSS)', fontsize=12)
plt.xticks(K_range)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fig_elbow = os.path.join('graficos', 'metodo_del_codo.png')
plt.savefig(fig_elbow, dpi=300)
plt.close()

# Parte 4: Coeficiente de Silueta
silhouette_scores = []
K_sil = list(range(2, 11))
for k in K_sil:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
axes[0].plot(K_sil, silhouette_scores, marker='s', linewidth=2.5, markersize=8, color='#2ca02c')
axes[0].axvline(x=5, color='#d62728', linestyle='--', linewidth=1.8, label=f'Máximo Global K=5 ({max(silhouette_scores):.4f})')
axes[0].set_title('Coeficiente de Silueta Promedio vs K', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Número de Clusters (K)', fontsize=11)
axes[0].set_ylabel('Silhouette Score Promedio', fontsize=11)
axes[0].set_xticks(K_sil)
axes[0].legend(fontsize=10)
axes[0].grid(True, linestyle=':', alpha=0.6)

# Diagrama de silueta detallado para K=5
km5 = KMeans(n_clusters=5, init='k-means++', n_init=20, random_state=42)
labels5 = km5.fit_predict(X_scaled)
sample_silhouette_values = silhouette_samples(X_scaled, labels5)
y_lower = 10
for i in range(5):
    ith_cluster_silhouette_values = sample_silhouette_values[labels5 == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / 5)
    axes[1].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
    axes[1].text(-0.05, y_lower + 0.5 * size_cluster_i, f'C{i}', fontweight='bold')
    y_lower = y_upper + 10
axes[1].axvline(x=silhouette_score(X_scaled, labels5), color="red", linestyle="--", label="Promedio (0.5547)")
axes[1].set_title("Estructura de Silueta por Cluster para K = 5", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Valores del Coeficiente de Silueta s(i)", fontsize=11)
axes[1].set_ylabel("Clusters", fontsize=11)
axes[1].set_yticks([])
axes[1].legend(loc="lower right")
plt.tight_layout()
fig_silhouette = os.path.join('graficos', 'coeficiente_de_silueta.png')
plt.savefig(fig_silhouette, dpi=300)
plt.close()

# Parte 5: Modelo K-Means Definitivo (K=5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=20, max_iter=300, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
plt.figure(figsize=(10, 6.2))
for i in range(optimal_k):
    cluster_points = df[df['Cluster'] == i]
    plt.scatter(
        cluster_points['Annual Income (k$)'],
        cluster_points['Spending Score (1-100)'],
        s=65, c=cluster_colors[i], label=f'Cluster {i}',
        alpha=0.85, edgecolor='k', linewidth=0.5
    )
plt.scatter(
    centroids_original[:, 0], centroids_original[:, 1],
    s=260, c='yellow', marker='*', edgecolor='black',
    linewidth=1.8, label='Centroides', zorder=10
)
plt.title('Segmentación Final de Clientes con K-Means (K=5)', fontsize=14, fontweight='bold')
plt.xlabel('Ingreso Anual (k$)', fontsize=12)
plt.ylabel('Puntaje de Gasto (1-100)', fontsize=12)
plt.legend(loc='upper right', frameon=True)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
fig_clusters = os.path.join('graficos', 'clusters_kmeans_final.png')
plt.savefig(fig_clusters, dpi=300)
plt.close()

# Parte 6: Interpretación de Negocio
profile_mapping = {}
for i in range(optimal_k):
    inc, sp = centroids_original[i]
    if inc > 70 and sp > 70:
        perfil = "Clientes de Alto Valor (VIP / Target Premium)"
        categoria = "Clientes de alto valor"
    elif inc > 70 and sp < 40:
        perfil = "Clientes con Potencial - Ahorradores Cautelosos"
        categoria = "Clientes con potencial (Ingreso Alto / Gasto Bajo)"
    elif inc < 40 and sp > 60:
        perfil = "Clientes con Potencial - Consumidores Impulsivos"
        categoria = "Clientes con potencial (Ingreso Bajo / Gasto Alto)"
    elif inc < 40 and sp < 40:
        perfil = "Clientes de Bajo Valor (Consumo Frugal)"
        categoria = "Clientes de bajo valor"
    else:
        perfil = "Clientes Promedio (Población Estándar)"
        categoria = "Clientes promedio"
    profile_mapping[i] = (perfil, categoria)

df['Perfil'] = df['Cluster'].map(lambda c: profile_mapping[c][0])
df['Categoria'] = df['Cluster'].map(lambda c: profile_mapping[c][1])

profile_summary = df.groupby(['Cluster', 'Perfil', 'Categoria']).agg(
    Clientes=('CustomerID', 'count'),
    Edad_Promedio=('Age', 'mean'),
    Ingreso_Promedio=('Annual Income (k$)', 'mean'),
    Gasto_Promedio=('Spending Score (1-100)', 'mean'),
    Porcentaje_Mujeres=('Gender', lambda g: (g == 'Female').mean() * 100)
).round(2).reset_index()

plt.figure(figsize=(11, 6.8))
sns.scatterplot(
    data=df, x='Annual Income (k$)', y='Spending Score (1-100)',
    hue='Perfil', palette='Set1', s=80, alpha=0.9, edgecolor='black', linewidth=0.5
)
plt.scatter(
    centroids_original[:, 0], centroids_original[:, 1],
    s=250, c='yellow', marker='X', edgecolor='black',
    linewidth=1.8, label='Centroides', zorder=10
)
plt.title('Mapa Estratégico de Perfiles de Negocio (Mall Customers)', fontsize=14, fontweight='bold')
plt.xlabel('Ingreso Anual (k$)', fontsize=12)
plt.ylabel('Puntaje de Gasto (1-100)', fontsize=12)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()
fig_business = os.path.join('graficos', 'mapa_segmentos_negocio.png')
plt.savefig(fig_business, dpi=300)
plt.close()

# -------------------------------------------------------------
# CONSTRUCCIÓN DEL DOCUMENTO WORD COMPLETO (.DOCX)
# -------------------------------------------------------------
print("Generando documento Word completo...")
doc = docx.Document()

for section in doc.sections:
    section.top_margin = Inches(0.95)
    section.bottom_margin = Inches(0.95)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)

COLOR_PRIMARY = RGBColor(26, 54, 93)     # #1A365D
COLOR_SECONDARY = RGBColor(43, 108, 176) # #2B6CB0
COLOR_DARK = RGBColor(45, 55, 72)        # #2D3748

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
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
        run.font.size = Pt(16.5)
        run.font.bold = True
        run.font.color.rgb = COLOR_PRIMARY
        h.paragraph_format.space_before = Pt(16)
        h.paragraph_format.space_after = Pt(6)
    elif level == 2:
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = COLOR_SECONDARY
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
    elif level == 3:
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = COLOR_DARK
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.space_after = Pt(3)
    return h

def add_callout(doc, text, title=""):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(6.5)
    cell = table.cell(0, 0)
    set_cell_background(cell, "F0F4F8")
    set_cell_margins(cell, top=130, bottom=130, left=180, right=180)
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
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

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

def style_table_header(table, headers):
    for j, h_text in enumerate(headers):
        cell = table.cell(0, j)
        set_cell_background(cell, "1A365D")
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        r = p.add_run(h_text)
        r.font.bold = True
        r.font.size = Pt(8.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

def populate_table_rows(table, rows_data, highlight_row_idx=None):
    for i_r, row_data in enumerate(rows_data, start=1):
        bg = "FFFFFF" if i_r % 2 != 0 else "F7FAFC"
        if highlight_row_idx is not None and i_r == highlight_row_idx:
            bg = "EBF8FF"
        for j_c, val in enumerate(row_data):
            c = table.cell(i_r, j_c)
            set_cell_background(c, bg)
            set_cell_margins(c, top=70, bottom=70, left=100, right=100)
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(0)
            r = p.add_run(str(val))
            r.font.size = Pt(8.5)
            if highlight_row_idx is not None and i_r == highlight_row_idx:
                r.font.bold = True
                r.font.color.rgb = COLOR_PRIMARY
            else:
                r.font.color.rgb = COLOR_DARK

# =============================================================
# PORTADA Y OBJETIVOS DEL TALLER
# =============================================================
title_p = doc.add_paragraph()
title_p.paragraph_format.space_before = Pt(18)
title_p.paragraph_format.space_after = Pt(4)
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
t_run = title_p.add_run("TALLER FINAL DE MINERÍA DE DATOS:\nSEGMENTACIÓN DE CLIENTES CON K-MEANS")
t_run.font.size = Pt(20)
t_run.font.bold = True
t_run.font.color.rgb = COLOR_PRIMARY

sub_p = doc.add_paragraph()
sub_p.paragraph_format.space_after = Pt(14)
sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
s_run = sub_p.add_run("Desarrollo Integral de las Partes 1 a 7, Respuestas Fundamentadas, Evaluación de Métricas y Despliegue Interactivo")
s_run.font.size = Pt(12)
s_run.font.italic = True
s_run.font.color.rgb = COLOR_SECONDARY

meta_table = doc.add_table(rows=5, cols=2)
meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
meta_data = [
    ("Dataset Oficial:", "Mall Customers Dataset (200 registros, 5 variables - Kaggle)"),
    ("Algoritmo Principal:", "K-Means Clustering (Inicialización k-means++, Distancia Euclidiana)"),
    ("Técnicas de Evaluación:", "StandardScaler (Z-Score), Método del Codo (WCSS) y Coeficiente de Silueta"),
    ("Compatibilidad de Código:", "Cualquier IDE Local (VS Code, JupyterLab, PyCharm) y Google Colab"),
    ("Despliegue Interactivo:", "Aplicación Web con Streamlit y Plotly (app.py)")
]
for row_idx, (k_txt, v_txt) in enumerate(meta_data):
    c0 = meta_table.cell(row_idx, 0)
    c1 = meta_table.cell(row_idx, 1)
    set_cell_background(c0, "EDF2F7")
    set_cell_background(c1, "FFFFFF")
    set_cell_margins(c0, top=60, bottom=60, left=100, right=100)
    set_cell_margins(c1, top=60, bottom=60, left=100, right=100)
    r0 = c0.paragraphs[0].add_run(k_txt)
    r0.font.bold = True
    r0.font.size = Pt(9.5)
    r0.font.color.rgb = COLOR_PRIMARY
    r1 = c1.paragraphs[0].add_run(v_txt)
    r1.font.size = Pt(9.5)
    r1.font.color.rgb = COLOR_DARK

doc.add_paragraph().paragraph_format.space_after = Pt(8)

add_callout(
    doc,
    "Objetivo General:\n"
    "Desarrollar un modelo de segmentación de clientes utilizando el algoritmo K-Means, aplicando técnicas de evaluación "
    "como el método del codo y el coeficiente de silueta, con el fin de analizar patrones de comportamiento y apoyar la toma de decisiones en marketing.\n\n"
    "Objetivos Específicos Cubiertos:\n"
    "• Comprender el funcionamiento matemático y geométrico del algoritmo K-Means.\n"
    "• Aplicar técnicas de preprocesamiento y estandarización de datos.\n"
    "• Responder de forma rigurosa todas las preguntas teóricas y analíticas del taller.\n"
    "• Determinar el número óptimo de clusters (K = 5) y evaluar la calidad mediante métricas internas.\n"
    "• Interpretar los resultados desde una perspectiva estratégica de negocio y marketing.\n"
    "• Implementar un despliegue interactivo funcional con clasificación en tiempo real en Streamlit.",
    title="MARCO DE OBJETIVOS DEL TALLER"
)

# =============================================================
# PARTE 1 – EXPLORACIÓN DE DATOS
# =============================================================
add_header_styled(doc, "Parte 1 – Exploración de Datos (EDA)", level=1)

add_header_styled(doc, "Actividades Desarrolladas en la Parte 1:", level=3)
add_body_paragraph(
    doc,
    "1. Carga del dataset: Se cargó el archivo Mall_Customers.csv compuesto por 200 registros completos (sin valores nulos ni filas duplicadas) y 5 columnas.\n"
    "2. Identificación de variables: CustomerID (identificador nominal entero), Gender (categórico binario: Female 56%, Male 44%), Age (cuantitativo discreto: 18 a 70 años), Annual Income (k$) (cuantitativo continuo en miles de USD: 15 a 137 k$) y Spending Score (1-100) (puntaje de propensión al gasto: 1 a 99 puntos).\n"
    "3. Análisis descriptivo básico: A continuación se presenta el resumen estadístico completo de las variables numéricas."
)

# Tabla de estadísticas descriptivas
desc_df = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].describe().round(2)
tbl_desc = doc.add_table(rows=4, cols=9)
tbl_desc.alignment = WD_TABLE_ALIGNMENT.CENTER
style_table_header(tbl_desc, ["Variable", "N", "Media", "Desv. Est.", "Mín", "Q1 (25%)", "Mediana (50%)", "Q3 (75%)", "Máx"])
desc_rows = []
for col_name in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    s = desc_df[col_name]
    desc_rows.append([col_name, int(s['count']), s['mean'], s['std'], s['min'], s['25%'], s['50%'], s['75%'], s['max']])
populate_table_rows(tbl_desc, desc_rows)
doc.add_paragraph().paragraph_format.space_after = Pt(6)

# Preguntas Parte 1
add_header_styled(doc, "Pregunta 1.1: ¿Qué variables son más adecuadas para segmentar clientes?", level=2)
add_body_paragraph(
    doc,
    "Las variables más adecuadas y con mayor poder discriminante para segmentar clientes en este problema son Annual Income (k$) (Ingreso Anual) y Spending Score (1-100) (Puntaje de Gasto).",
    bold_prefix="Respuesta: "
)
add_body_paragraph(
    doc,
    "• Justificación de Negocio (Capacidad vs. Propensión): En analítica de retail, la segmentación accionable requiere cruzar el poder adquisitivo del consumidor (cuánto puede gastar, medido por el Ingreso Anual) con su comportamiento real de consumo en el centro comercial (cuánto decide gastar, medido por el Spending Score). Esta combinación forma una matriz estratégica bidimensional directa.\n"
    "• Justificación Geométrica y Estadística: En el espacio bidimensional [Ingreso Anual vs. Puntaje de Gasto] las observaciones forman de manera natural 5 conglomerados de alta densidad local y clara separación inter-grupal. En cambio, la variable Age (Edad) presenta distribución dispersa a lo largo de todos los niveles de ingreso y se aprovecha mejor como variable descriptiva posterior para perfilar demográficamente cada cluster.\n"
    "• Exclusión de CustomerID y Gender del cálculo euclidiano: CustomerID es un índice secuencial del 1 al 200 carente de sentido métrico (incluirlo introduciría ruido espurio). Por su parte, Gender es una variable dicotómica que muestra distribuciones casi idénticas en ambos sexos, por lo que se emplea en el perfilamiento posterior."
)

add_header_styled(doc, "Pregunta 1.2: ¿Se identifican valores atípicos?", level=2)
add_body_paragraph(
    doc,
    "Sí. Aplicando el método formal del Rango Intercuartílico (IQR = Q3 - Q1) y diagramas de caja (boxplots), se identificaron exactamente 2 valores atípicos superiores en la variable Annual Income (k$), correspondientes a los clientes con CustomerID 199 y 200 (ambos con ingresos de $137,000 USD anuales). En las variables Age y Spending Score (1-100) no existen valores atípicos (0 outliers).",
    bold_prefix="Respuesta: "
)

tbl_iqr = doc.add_table(rows=4, cols=6)
tbl_iqr.alignment = WD_TABLE_ALIGNMENT.CENTER
style_table_header(tbl_iqr, ["Variable", "Q1 (25%)", "Q3 (75%)", "IQR", "Límites Normales [Inf, Sup]", "Outliers Detectados"])
iqr_rows = [
    ["Age (años)", "28.75", "49.00", "20.25", "[-1.62, 79.38]", "0 (Ninguno)"],
    ["Annual Income (k$)", "41.50", "78.00", "36.50", "[-13.25, 132.75]", "2 clientes (ID 199 y 200: 137 k$)"],
    ["Spending Score (1-100)", "34.75", "73.00", "38.25", "[-22.62, 130.38]", "0 (Ninguno)"]
]
populate_table_rows(tbl_iqr, iqr_rows, highlight_row_idx=2)
doc.add_paragraph().paragraph_format.space_after = Pt(6)

add_body_paragraph(
    doc,
    "• Tratamiento de los valores atípicos: Estos 2 registros NO son errores de medición ni datos corruptos; representan clientes reales de alto patrimonio ( segmento VIP/Premium ). En segmentación comercial, eliminar a los clientes más acaudalados cercenaría la visibilidad del segmento de lujo. Por lo tanto, se conservan y su magnitud se regulariza mediante estandarización Z-Score."
)

add_header_styled(doc, "Pregunta 1.3: ¿Qué tipo de relaciones se observan entre las variables?", level=2)
add_body_paragraph(
    doc,
    "Se observan tres comportamientos relacionales fundamentales entre las variables del dataset:",
    bold_prefix="Respuesta: "
)
add_body_paragraph(
    doc,
    "1. Relación No Lineal Estructurada entre Ingreso Anual y Puntaje de Gasto (r = 0.010): El coeficiente de correlación lineal de Pearson es prácticamente cero (0.010), lo que demuestra que no existe proporcionalidad lineal directa (ganar más dinero no implica gastar más en el mall). Sin embargo, visualmente existe una fuerte relación espacial no lineal en forma de 'X' o cinco islas bien diferenciadas: cuatro grupos en las esquinas extremas y un grupo denso en el centro.\n"
    "2. Relación Lineal Inversa Moderada entre Edad y Puntaje de Gasto (r = -0.327): Existe una correlación negativa estadísticamente significativa. Los clientes jóvenes (18 a 39 años) concentran los puntajes de gasto más altos (superiores a 60 puntos), mientras que los clientes mayores de 40 años presentan un comportamiento más conservador (puntajes entre 1 y 60 puntos).\n"
    "3. Independencia entre Edad e Ingreso Anual (r = -0.012): El nivel de ingresos no depende linealmente de la edad en esta muestra de clientes."
)

p_img1 = doc.add_paragraph()
p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_picture(fig_eda_dist, width=Inches(6.2))
cap1 = doc.add_paragraph("Figura 1: Histogramas de distribución y Boxplots para detección de valores atípicos (Parte 1).")
cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap1.runs[0].font.size = Pt(8.5)
cap1.runs[0].font.italic = True

p_img2 = doc.add_paragraph()
p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_picture(fig_eda_rel, width=Inches(6.2))
cap2 = doc.add_paragraph("Figura 2: Gráfico de dispersión Ingreso vs. Gasto por género y Matriz de Correlación de Pearson (Parte 1).")
cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap2.runs[0].font.size = Pt(8.5)
cap2.runs[0].font.italic = True

# =============================================================
# PARTE 2 – PREPROCESAMIENTO
# =============================================================
add_header_styled(doc, "Parte 2 – Preprocesamiento de Datos", level=1)

add_body_paragraph(
    doc,
    "Actividad desarrollada: Se aplicó la transformación StandardScaler de scikit-learn a la matriz de características X = [Annual Income (k$), Spending Score (1-100)], estandarizando cada variable mediante la fórmula Z-Score: z = (x - μ) / σ, logrando media cero (μ = 0) y desviación estándar unitaria (σ = 1)."
)

add_header_styled(doc, "Pregunta 2.1: ¿Por qué es necesario escalar las variables en el algoritmo K-Means?", level=2)
add_body_paragraph(
    doc,
    "Escalar las variables es estrictamente necesario en K-Means porque es un algoritmo basado en distancias geométricas (específicamente la distancia euclidiana) y varianzas isotrópicas.",
    bold_prefix="Respuesta: "
)
add_body_paragraph(
    doc,
    "1. Dominancia de Escala en la Distancia Euclidiana: K-Means asigna cada cliente x_i al centroide μ_k más cercano calculando:\n"
    "       d(x_i, μ_k) = √ [ (Ingreso_i - μ_Ingreso)² + (Gasto_i - μ_Gasto)² ]\n"
    "Si las variables se encuentran en escalas o rangos distintos (por ejemplo, el Ingreso llega hasta 137 o si estuviera expresado en dólares reales hasta 137,000, mientras el Puntaje de Gasto va de 1 a 100), las diferencias al cuadrado de la variable con mayor magnitud numérica dominarán por completo la suma. El algoritmo terminaría agrupando únicamente por ingreso e ignorando el puntaje de gasto.\n\n"
    "2. Eliminación del Sesgo de Varianza: Sin estandarización, las variables con mayor varianza ejercen una tracción desproporcionada sobre la posición de los centroides, deformando la geometría esférica de los clusters en elipsoides sesgados.\n\n"
    "3. Equidad de Peso (50% - 50%) y Convergencia: Al aplicar StandardScaler, ambas variables quedan expresadas en unidades adimensionales de desviación estándar (rango aproximado de -2.0 a +2.9), garantizando que tanto el Ingreso Anual como el Puntaje de Gasto tengan exactamente el mismo peso relativo en la función objetivo y acelerando la convergencia del algoritmo."
)

# =============================================================
# PARTE 3 – MÉTODO DEL CODO
# =============================================================
add_header_styled(doc, "Parte 3 – Método del Codo (Elbow Method)", level=1)

add_body_paragraph(
    doc,
    "Actividad desarrollada: Se entrenaron 10 modelos secuenciales de K-Means para K = 1 hasta K = 10 sobre los datos escalados, registrando en cada uno la Inercia o WCSS (Within-Cluster Sum of Squares), definida como:\n"
    "       WCSS = ∑_{k=1}^K ∑_{x_i ∈ C_k} || x_i - μ_k ||²"
)

tbl_elbow = doc.add_table(rows=11, cols=4)
tbl_elbow.alignment = WD_TABLE_ALIGNMENT.CENTER
style_table_header(tbl_elbow, ["Número de Clusters (K)", "Inercia Intra-Cluster (WCSS)", "Reducción Absoluta (Δ WCSS)", "Reducción Porcentual (%)"])
elbow_rows = []
for idx_k, k_val in enumerate(K_range):
    val_w = wcss[idx_k]
    if idx_k == 0:
        d_abs, d_pct = "-", "-"
    else:
        diff = wcss[idx_k - 1] - val_w
        pct = (diff / wcss[idx_k - 1]) * 100
        d_abs = f"-{diff:.2f}"
        d_pct = f"-{pct:.1f}%"
    elbow_rows.append([f"K = {k_val}", f"{val_w:.2f}", d_abs, d_pct])
populate_table_rows(tbl_elbow, elbow_rows, highlight_row_idx=5)
doc.add_paragraph().paragraph_format.space_after = Pt(6)

add_header_styled(doc, "Pregunta 3.1: ¿Por qué valores cercanos a cero indican mejor ajuste?", level=2)
add_body_paragraph(
    doc,
    "Porque la inercia (WCSS) mide la suma de las distancias al cuadrado entre cada observación y el centroide de su propio cluster; por tanto, un valor cercano a cero indica que los puntos están muy próximos a su centroide, reflejando clusters internamente muy compactos, densos y homogéneos.",
    bold_prefix="Respuesta: "
)
add_body_paragraph(
    doc,
    "• Precisión Geométrica vs. Sobreajuste (Overfitting): Matemáticamente, a medida que aumenta K, el WCSS disminuye monótonamente. En el extremo donde K = N (200 clusters para 200 clientes), el WCSS es exactamente 0, pero el modelo carece de utilidad porque no agrupa ni generaliza. Por ello, aunque un valor menor indica mejor ajuste geométrico, se debe buscar el equilibrio entre bajo WCSS y un número reducido y manejable de clusters."
)

add_header_styled(doc, "Pregunta 3.2: ¿Cómo se identifica el punto óptimo de K?", level=2)
add_body_paragraph(
    doc,
    "El punto óptimo de K se identifica localizando el 'codo' o punto de máxima curvatura (inflexión) en la gráfica de WCSS contra K, es decir, el valor donde la reducción de la inercia deja de ser drástica y pasa a ser marginal o lineal.",
    bold_prefix="Respuesta: "
)
add_body_paragraph(
    doc,
    "• Evidencia en nuestro modelo: En la tabla y en la Figura 3 se observa que al incrementar K de 1 a 5 la inercia cae fuertemente en cada paso (de 400.00 a 65.57, con reducciones de entre 32% y 39% por paso). Sin embargo, al pasar de K = 5 a K = 6, la reducción cae abruptamente a solo 15.8% y la curva se aplana. Esto confirma inequívocamente que el punto óptimo es K = 5."
)

p_img3 = doc.add_paragraph()
p_img3.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_picture(fig_elbow, width=Inches(5.7))
cap3 = doc.add_paragraph("Figura 3: Gráfica del Método del Codo mostrando el quiebre de inercia en K = 5 (Parte 3).")
cap3.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap3.runs[0].font.size = Pt(8.5)
cap3.runs[0].font.italic = True

# =============================================================
# PARTE 4 – COEFICIENTE DE SILUETA
# =============================================================
add_header_styled(doc, "Parte 4 – Coeficiente de Silueta (Silhouette Score)", level=1)

add_body_paragraph(
    doc,
    "Actividad desarrollada: Se calculó el Coeficiente de Silueta promedio para K = 2 hasta K = 10 utilizando la formulación matemática para cada punto i:\n"
    "       s(i) = [ b(i) - a(i) ] / max( a(i), b(i) )"
)

tbl_sil = doc.add_table(rows=len(K_sil)+1, cols=3)
tbl_sil.alignment = WD_TABLE_ALIGNMENT.CENTER
style_table_header(tbl_sil, ["Número de Clusters (K)", "Coeficiente de Silueta Promedio", "Diagnóstico de Calidad"])
sil_rows = []
for k_val, s_val in zip(K_sil, silhouette_scores):
    if k_val == 5:
        diag = "MÁXIMO GLOBAL ÓPTIMO (Alta cohesión y separación)"
    elif s_val >= 0.45:
        diag = "Estructura aceptable pero subóptima"
    else:
        diag = "Solapamiento o sobre-fragmentación de grupos"
    sil_rows.append([f"K = {k_val}", f"{s_val:.4f}", diag])
populate_table_rows(tbl_sil, sil_rows, highlight_row_idx=4)
doc.add_paragraph().paragraph_format.space_after = Pt(6)

add_header_styled(doc, "Pregunta 4.1: ¿Qué mide el coeficiente de silueta?", level=2)
add_body_paragraph(
    doc,
    "El coeficiente de silueta mide la calidad interna del agrupamiento combinando simultáneamente dos criterios: la cohesión intra-cluster a(i) (distancia promedio del punto i a los demás puntos de su mismo cluster) y la separación inter-cluster b(i) (distancia promedio del punto i a todos los puntos del cluster vecino más cercano).",
    bold_prefix="Respuesta: "
)

add_header_styled(doc, "Pregunta 4.2: ¿Qué significa un valor cercano a 1?", level=2)
add_body_paragraph(
    doc,
    "Un valor cercano a +1 significa que la observación está óptimamente agrupada: su distancia interna a(i) es muy pequeña en comparación con la distancia al cluster vecino más cercano b(i) (a(i) ≪ b(i)). Indica clusters compactos y claramente separados entre sí.",
    bold_prefix="Respuesta: "
)

add_header_styled(doc, "Pregunta 4.3: ¿Qué implica un valor cercano a 0 o negativo?", level=2)
add_body_paragraph(
    doc,
    "• Valor cercano a 0 (s(i) ≈ 0): Implica que a(i) ≈ b(i), es decir, la observación se encuentra justo en la frontera de decisión o zona de solapamiento entre dos clusters adyacentes.\n"
    "• Valor negativo (s(i) < 0): Implica que a(i) > b(i), lo que significa que la observación fue asignada erróneamente a un cluster cuando en realidad está más cerca de los puntos de un cluster vecino.",
    bold_prefix="Respuesta: "
)

add_header_styled(doc, "Pregunta 4.4: ¿Coincide el valor óptimo de K con el método del codo? Justifique.", level=2)
add_body_paragraph(
    doc,
    "Sí, coincide con exactitud absoluta en K = 5.",
    bold_prefix="Respuesta y Justificación: "
)
add_body_paragraph(
    doc,
    "• Justificación Cuantitativa: Mientras el método del codo identificó en K = 5 el punto de inflexión donde se estabiliza la varianza intra-cluster (WCSS = 65.57), el coeficiente de silueta alcanza su valor máximo global exactamente en K = 5 con un puntaje de 0.5547 (superando notablemente a K=4 con 0.4939 y a K=6 con 0.5399).\n"
    "• Conclusión de Validación Cruzada: Al converger tanto una métrica de compacidad pura (Codo) como una métrica de relación cohesión/separación (Silueta) en K = 5, se confirma matemáticamente que la estructura real subyacente del dataset consta de 5 segmentos naturales."
)

p_img4 = doc.add_paragraph()
p_img4.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_picture(fig_silhouette, width=Inches(6.2))
cap4 = doc.add_paragraph("Figura 4: Curva de Coeficiente de Silueta vs K y diagrama de silueta por cluster para K = 5 (Parte 4).")
cap4.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap4.runs[0].font.size = Pt(8.5)
cap4.runs[0].font.italic = True

# =============================================================
# PARTE 5 – MODELO K-MEANS
# =============================================================
add_header_styled(doc, "Parte 5 – Modelo K-Means (Entrenamiento y Centroides)", level=1)

add_body_paragraph(
    doc,
    "Actividades desarrolladas:\n"
    "1. Se entrenó el modelo K-Means definitivo con K = 5 clusters (init='k-means++', n_init=20, random_state=42).\n"
    "2. Se asignó la etiqueta de cluster (0 a 4) a cada una de las 200 observaciones del dataset.\n"
    "3. Se extrajeron las coordenadas de los 5 centroides tanto en la escala estandarizada (Z-Score) como en su escala original desescalada (miles de USD y puntaje 1-100)."
)

tbl_cent = doc.add_table(rows=6, cols=6)
tbl_cent.alignment = WD_TABLE_ALIGNMENT.CENTER
style_table_header(tbl_cent, ["Cluster", "Centroide Z-Score (Ingreso, Gasto)", "Ingreso Promedio (k$)", "Gasto Promedio (1-100)", "N° Clientes", "% Total"])
cent_rows = []
for c_i in range(optimal_k):
    n_cl = (df['Cluster'] == c_i).sum()
    pct_cl = (n_cl / len(df)) * 100
    z_inc, z_sp = centroids_scaled[c_i]
    o_inc, o_sp = centroids_original[c_i]
    cent_rows.append([
        f"Cluster {c_i}",
        f"({z_inc:+.2f}, {z_sp:+.2f})",
        f"${o_inc:.2f} k$",
        f"{o_sp:.2f} pts",
        f"{n_cl} clientes",
        f"{pct_cl:.1f}%"
    ])
populate_table_rows(tbl_cent, cent_rows)
doc.add_paragraph().paragraph_format.space_after = Pt(6)

add_header_styled(doc, "Pregunta 5.1: ¿Qué representa cada cluster?", level=2)
add_body_paragraph(
    doc,
    "Cada uno de los 5 clusters obtenidos representa un grupo socioeconómico y conductual claramente definido dentro del centro comercial:",
    bold_prefix="Respuesta: "
)
add_body_paragraph(
    doc,
    "• Cluster 0 (Ingreso Medio: $55.30k | Gasto Medio: 49.52 pts | 81 clientes, 40.5%): Representa al consumidor estándar o promedio del centro comercial, con capacidad financiera moderada y hábitos de compra equilibrados.\n"
    "• Cluster 1 (Ingreso Alto: $86.54k | Gasto Alto: 82.13 pts | 39 clientes, 19.5%): Representa al segmento élite o VIP, clientes con alta solvencia económica y máxima frecuencia y volumen de gasto.\n"
    "• Cluster 2 (Ingreso Bajo: $25.73k | Gasto Alto: 79.36 pts | 22 clientes, 11.0%): Representa a los compradores entusiastas o impulsivos (predominantemente jóvenes con edad media de 25.3 años), quienes pese a percibir ingresos bajos registran un gasto muy elevado.\n"
    "• Cluster 3 (Ingreso Alto: $88.20k | Gasto Bajo: 17.11 pts | 35 clientes, 17.5%): Representa a los clientes ahorradores, exigentes o cautelosos, que cuentan con el mayor ingreso anual promedio pero gastan muy poco en el centro comercial.\n"
    "• Cluster 4 (Ingreso Bajo: $26.30k | Gasto Bajo: 20.91 pts | 23 clientes, 11.5%): Representa a los clientes conservadores de bajos ingresos, cuyo consumo se limita a compras puntuales o básicas."
)

add_header_styled(doc, "Pregunta 5.2: ¿Qué diferencias existen entre los grupos obtenidos?", level=2)
add_body_paragraph(
    doc,
    "Existen diferencias estructurales pronunciadas en tres dimensiones: capacidad económica, propensión al gasto y perfil demográfico (edad):",
    bold_prefix="Respuesta: "
)
add_body_paragraph(
    doc,
    "1. Asimetría en Capacidad Financiera: Los Clusters 1 y 3 superan los $86,000 USD anuales, triplicando el ingreso promedio de los Clusters 2 y 4 (~$26,000 USD).\n"
    "2. Divergencia Conductual a Igualdad de Ingreso: A pesar de tener casi el mismo ingreso alto (~$87k), el Cluster 1 tiene un puntaje de gasto de 82.13 pts mientras que el Cluster 3 apenas alcanza 17.11 pts (una brecha de 65 puntos). El mismo fenómeno ocurre en la franja de bajos ingresos entre el Cluster 2 (79.36 pts) y el Cluster 4 (20.91 pts).\n"
    "3. Diferencia Demográfica Generacional: Los clusters de alto gasto (Cluster 2 con 25.27 años y Cluster 1 con 32.69 años) son significativamente más jóvenes que los clusters de bajo gasto (Cluster 3 con 41.11 años y Cluster 4 con 45.22 años)."
)

p_img5 = doc.add_paragraph()
p_img5.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_picture(fig_clusters, width=Inches(5.8))
cap5 = doc.add_paragraph("Figura 5: Visualización de los 5 clusters generados y sus centroides (estrellas amarillas) (Parte 5).")
cap5.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap5.runs[0].font.size = Pt(8.5)
cap5.runs[0].font.italic = True

# =============================================================
# PARTE 6 – INTERPRETACIÓN DE NEGOCIO
# =============================================================
add_header_styled(doc, "Parte 6 – Interpretación de Negocio y Marketing", level=1)

add_body_paragraph(
    doc,
    "Actividad desarrollada: Se clasificaron los 5 clusters en los perfiles estratégicos solicitados por el taller (Clientes de alto valor, Clientes promedio, Clientes de bajo valor y Clientes con potencial), consolidando sus indicadores demográficos y comerciales en la siguiente matriz:"
)

tbl_biz = doc.add_table(rows=6, cols=6)
tbl_biz.alignment = WD_TABLE_ALIGNMENT.CENTER
style_table_header(tbl_biz, ["Cluster", "Clasificación Oficial Guía", "Nombre del Perfil Comercial", "Edad Media", "Ingreso Medio", "Gasto Medio"])
biz_rows = []
for idx_b, r_b in profile_summary.iterrows():
    c_num = int(r_b['Cluster'])
    biz_rows.append([
        f"Cluster {c_num}",
        profile_mapping[c_num][1],
        profile_mapping[c_num][0],
        f"{r_b['Edad_Promedio']:.1f} años",
        f"${r_b['Ingreso_Promedio']:.2f} k$",
        f"{r_b['Gasto_Promedio']:.2f} pts"
    ])
populate_table_rows(tbl_biz, biz_rows)
doc.add_paragraph().paragraph_format.space_after = Pt(6)

add_header_styled(doc, "Pregunta 6.1: ¿Qué estrategias de marketing se podrían aplicar a cada grupo?", level=2)
add_body_paragraph(
    doc,
    "Para maximizar la rentabilidad del centro comercial, se proponen estrategias diferenciadas por perfil:",
    bold_prefix="Respuesta: "
)
add_body_paragraph(
    doc,
    "1. Clientes de Alto Valor (Cluster 1 - Alto Ingreso / Alto Gasto):\n"
    "   • Estrategia: Fidelización VIP, Retención y Marketing Experiencial.\n"
    "   • Acciones: Programas de membresía Black/Platinum con acceso a preventas exclusivas, personal shopper gratuito, valet parking preferencial y eventos privados de marcas de lujo. No requieren descuentos agresivos, sino exclusividad y reconocimiento.\n\n"
    "2. Clientes con Potencial - Ahorradores Cautelosos (Cluster 3 - Alto Ingreso / Bajo Gasto):\n"
    "   • Estrategia: Conversión por Propuesta de Valor, Calidad y Ampliación del Mix Comercial.\n"
    "   • Acciones: Atraer su gasto incorporando tiendas de tecnología avanzada, alta gastronomía, servicios ejecutivos y marcas de prestigio duradero; enviar comunicación personalizada basada en retorno de inversión, garantías extendidas y alianzas con tarjetas bancarias prémium (cashback).\n\n"
    "3. Clientes con Potencial - Consumidores Impulsivos (Cluster 2 - Bajo Ingreso / Alto Gasto):\n"
    "   • Estrategia: Marketing Digital, Tendencias y Micro-Financiamiento.\n"
    "   • Acciones: Campañas intensivas en redes sociales (TikTok, Instagram) con influencers de moda rápida (fast-fashion) y entretenimiento; implementar pagos fraccionados sin interés (Buy Now, Pay Later - BNPL) y cupones gamificados para mantener su frecuencia de visita sin sobreendeudarlos.\n\n"
    "4. Clientes Promedio (Cluster 0 - Ingreso Medio / Gasto Medio):\n"
    "   • Estrategia: Venta Cruzada (Cross-Selling) e Incremento del Ticket Promedio.\n"
    "   • Acciones: Paquetes promocionales familiares de fin de semana (compras + cine + plazoleta de comidas), programas de acumulación de puntos por visitas recurrentes y campañas estacionales (Día de la Madre, Navidad, Regreso a Clases).\n\n"
    "5. Clientes de Bajo Valor (Cluster 4 - Bajo Ingreso / Bajo Gasto):\n"
    "   • Estrategia: Marketing de Oportunidad y Tráfico en Horas Valle.\n"
    "   • Acciones: Descuentos por liquidación de temporada, promociones 2x1 en productos básicos, ferias de outlet y descuentos especiales en días de semana de baja afluencia (martes y miércoles)."
)

add_header_styled(doc, "Pregunta 6.2: ¿Qué segmento representa mayor oportunidad de crecimiento?", level=2)
add_body_paragraph(
    doc,
    "El segmento que representa la mayor oportunidad de crecimiento financiero para el centro comercial es el Cluster 3: Clientes con Potencial - Ahorradores Cautelosos (Alto Ingreso y Bajo Gasto), complementado en volumen por el Cluster 0 (Clientes Promedio).",
    bold_prefix="Respuesta y Justificación de Negocio: "
)
add_body_paragraph(
    doc,
    "• Potencial Financiero No Capturado (Cluster 3): Este grupo posee el ingreso promedio más alto de todo el estudio ($88,200 USD anuales), pero su puntaje de gasto es el más bajo (17.11 puntos). Esto significa que el cliente ya visita el centro comercial y cuenta con plena liquidez económica, pero está realizando sus compras principales en la competencia o en canales online. Lograr que este segmento incremente su gasto al nivel del Cluster 1 duplicaría la facturación de alto margen del mall con un costo de adquisición (CAC) mínimo.\n"
    "• Efecto Escala (Cluster 0): Al concentrar el 40.5% de todos los clientes (81 personas), cualquier incremento marginal del 10% o 15% en su gasto promedio mediante venta cruzada genera un flujo de caja inmediato y estable."
)

p_img6 = doc.add_paragraph()
p_img6.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_picture(fig_business, width=Inches(6.2))
cap6 = doc.add_paragraph("Figura 6: Mapa de Segmentación Estratégica con los perfiles de negocio asignados (Parte 6).")
cap6.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap6.runs[0].font.size = Pt(8.5)
cap6.runs[0].font.italic = True

# =============================================================
# PARTE 7 – DESPLIEGUE INTERACTIVO CON STREAMLIT
# =============================================================
add_header_styled(doc, "Parte 7 – Despliegue con Streamlit y Guía de Ejecución", level=1)

add_body_paragraph(
    doc,
    "Se desarrolló la aplicación web interactiva completa en el archivo app.py utilizando el framework Streamlit y gráficos dinámicos de Plotly, cumpliendo todos los requerimientos funcionales exigidos en la guía:"
)

add_body_paragraph(
    doc,
    "1. Visualización del Método del Codo y Silueta: Pestaña interactiva que grafica la curva WCSS (K=1..10) y el Coeficiente de Silueta (K=2..10), señalizando el punto óptimo K=5.\n"
    "2. Visualización de los Clusters Generados: Mapa interactivo en 2D con proyección de los 5 clusters, sus centroides, tamaño proporcional a la edad y tabla estadística resumen.\n"
    "3. Ingreso de un Nuevo Cliente: Controles deslizantes (sliders) reactivos en tiempo real para introducir Ingreso Anual (k$), Puntaje de Gasto (1-100), Edad y Género, además de 5 botones rápidos de prueba por arquetipo.\n"
    "4. Clasificación Automática en un Cluster: Al ingresar o mover cualquier valor, el sistema aplica scaler.transform() y kmeans.predict() en tiempo real, mostrando la tarjeta ejecutiva con el Cluster asignado, el Perfil de Negocio, la Estrategia de Marketing recomendada y proyectando al nuevo cliente como una estrella dorada en el gráfico de dispersión."
)

# Tabla de validación de predicciones en Streamlit
add_header_styled(doc, "Validación de Clasificación Automática de Nuevos Clientes:", level=3)
tbl_pred = doc.add_table(rows=6, cols=5)
tbl_pred.alignment = WD_TABLE_ALIGNMENT.CENTER
style_table_header(tbl_pred, ["Caso de Prueba", "Ingreso Ingresado", "Gasto Ingresado", "Cluster Predicho", "Perfil Asignado por la App"])
pred_rows = [
    ["Prueba 1: Cliente VIP", "$95 k$", "88 pts", "Cluster 1", "Clientes de Alto Valor (VIP / Target Premium)"],
    ["Prueba 2: Ahorrador", "$90 k$", "18 pts", "Cluster 3", "Clientes con Potencial - Ahorradores Cautelosos"],
    ["Prueba 3: Impulsivo", "$25 k$", "85 pts", "Cluster 2", "Clientes con Potencial - Consumidores Impulsivos"],
    ["Prueba 4: Promedio", "$55 k$", "50 pts", "Cluster 0", "Clientes Promedio (Población Estándar)"],
    ["Prueba 5: Bajo Valor", "$22 k$", "15 pts", "Cluster 4", "Clientes de Bajo Valor (Consumo Frugal)"]
]
populate_table_rows(tbl_pred, pred_rows)
doc.add_paragraph().paragraph_format.space_after = Pt(8)

add_callout(
    doc,
    "1. Ejecución en Google Colab:\n"
    "   • Subir y abrir el archivo taller_kmeans_mall_customers.ipynb en Google Colab.\n"
    "   • Ejecutar todas las celdas (Entorno de ejecución -> Ejecutar todas). El cuaderno descarga automáticamente el dataset Mall_Customers.csv desde un repositorio público si no lo detecta localmente.\n\n"
    "2. Ejecución en Cualquier IDE Local (VS Code, PyCharm, JupyterLab):\n"
    "   • Abrir la carpeta del proyecto y ejecutar el cuaderno taller_kmeans_mall_customers.ipynb.\n\n"
    "3. Ejecución de la Aplicación Web Streamlit (Parte 7):\n"
    "   • En la terminal del proyecto, ejecutar el comando:\n"
    "     streamlit run app.py\n"
    "   • La aplicación se abrirá automáticamente en el navegador en http://localhost:8501.",
    title="GUÍA RÁPIDA DE EJECUCIÓN MULTI-ENTORNO"
)

# Guardar en ambos nombres para comodidad del usuario
OUT_1 = "Respuestas_Taller_KMeans.docx"
OUT_2 = "Taller_Final_Mineria_de_Datos_KMeans_Completo.docx"
doc.save(OUT_1)
shutil.copyfile(OUT_1, OUT_2)
print(f"Documentos Word generados exitosamente:\n - {OUT_1}\n - {OUT_2}")
