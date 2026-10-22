"""
Script para generar el cuaderno Jupyter interactivo 'taller_kmeans_mall_customers.ipynb'
con respuestas completas, visualizaciones y compatibilidad dual (Local / Google Colab).
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# 0. ENCABEZADO Y BADGE COLAB
cells.append(nbf.v4.new_markdown_cell("""# 🛍️ TALLER FINAL DE MINERÍA DE DATOS: SEGMENTACIÓN DE CLIENTES CON K-MEANS

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KevinDev-exe/Laboratorio-mineria/blob/main/taller_kmeans_mall_customers.ipynb)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=KevinDev-exe/Laboratorio-mineria&branch=main&mainModule=app.py)

**Asignatura:** Minería de Datos & Inteligencia de Negocios  
**Dataset:** Mall Customers Dataset ([Kaggle Source](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python))  
**Objetivo:** Desarrollar un modelo de segmentación de clientes utilizando el algoritmo K-Means, aplicando técnicas de evaluación como el método del codo y el coeficiente de silueta, con el fin de analizar patrones de comportamiento y apoyar la toma de decisiones estratégicas de marketing.

---
### 📌 Instrucciones de Ejecución:
- **En Google Colab:** Puedes ejecutar este cuaderno directamente celda por celda (`Shift + Enter`). El código incluye una rutina de descarga automática que obtiene el dataset oficial desde un repositorio GitHub público en caso de que no esté cargado en la sesión efímera.
- **En cualquier IDE Local (VS Code, JupyterLab, PyCharm):** Funciona de forma inmediata si tienes el entorno Python con `pandas`, `numpy`, `matplotlib`, `seaborn` y `scikit-learn`. Si el archivo `Mall_Customers.csv` está en la misma carpeta lo utilizará directamente.
"""))

# 1. IMPORTS Y CONFIGURACIÓN
cells.append(nbf.v4.new_code_cell("""# Configuración de librerías esenciales
import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Algoritmos y métricas de evaluación
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import joblib

# Configuración gráfica para visualizaciones de alta calidad
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 110
%matplotlib inline

print("✅ Librerías importadas correctamente.")
"""))

# 2. CARGA HÍBRIDA DEL DATASET
cells.append(nbf.v4.new_code_cell("""# Carga Inteligente Híbrida (Local y Google Colab)
DATA_FILE = 'Mall_Customers.csv'
DATA_URL = 'https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv'

if not os.path.exists(DATA_FILE):
    print(f"📥 Archivo local no encontrado. Descargando automáticamente desde URL pública...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    print("✅ Descarga completada con éxito.")
else:
    print(f"📁 Cargando dataset existente en el entorno local: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
# Limpieza de nombres de columnas
df.columns = [col.strip() for col in df.columns]

print(f"Dimensiones del dataset: {df.shape[0]} filas y {df.shape[1]} columnas.")
df.head()
"""))

# 3. PARTE 1 - EXPLORACIÓN DE DATOS (EDA)
cells.append(nbf.v4.new_markdown_cell("""---
## PARTE 1 – Exploración de Datos (EDA)

En esta fase inicial realizamos:
1. Inspección estructural del dataset y verificación de tipos de datos y valores nulos.
2. Análisis descriptivo de las variables cuantitativas (tendencia central y dispersión).
3. Detección formal de valores atípicos (outliers) mediante el criterio del Rango Intercuartílico (IQR).
4. Análisis de relaciones bivariadas y correlaciones lineales.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Inspección general de tipos de datos y valores nulos
print("=== INFORMACIÓN GENERAL DEL DATASET ===")
df.info()

print("\\n=== VALORES NULOS POR VARIABLE ===")
print(df.isnull().sum())

print("\\n=== ESTADÍSTICAS DESCRIPTIVAS BÁSICAS ===")
df.describe().round(2)
"""))

cells.append(nbf.v4.new_markdown_cell("""### ❓ Pregunta 1.1: ¿Qué variables son más adecuadas para segmentar clientes?

**Respuesta y Fundamentación:**

1. **Variables Óptimas Seleccionadas:** Las variables más adecuadas, contundentes y representativas para la segmentación en este problema son **`Annual Income (k$)` (Ingreso Anual)** y **`Spending Score (1-100)` (Puntaje de Gasto)**.
2. **Capacidad de Discriminación Conductual vs. Demográfica:**
   - El objetivo primario de la segmentación en retail comercial es agrupar a los individuos por su capacidad económica de compra y su disposición psicológica real al consumo.
   - La combinación `[Annual Income vs. Spending Score]` crea un espacio bidimensional ortogonal que separa a los clientes de forma natural en cuadrantes estratégicos de negocio: alto ingreso con alto gasto, alto ingreso con bajo gasto, bajo ingreso con alto gasto, etc.
3. **Separabilidad Geométrica:**
   - Como se evidenciará en la gráfica de dispersión, en el plano Ingreso vs. Gasto los datos se distribuyen espontáneamente en 5 agrupaciones densas y bien aisladas entre sí.
   - En contraste, la variable `Age` (Edad) muestra correlaciones lineales débiles con el gasto ($r = -0.33$) y una dispersión continua difusa que no delimita fronteras nítidas sin recurrir a hiperplanos complejos.
4. **Exclusión de `CustomerID`:**
   - La variable `CustomerID` es un identificador consecutivo artificial sin valor predictivo o semántico; debe ser excluida para no distorsionar las métricas de distancia euclidiana.
"""))

cells.append(nbf.v4.new_code_cell("""# 2. Detección formal de Outliers mediante Criterio IQR
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
outliers_summary = {}

print("=== REPORTE DE VALORES ATÍPICOS (IQR) ===")
for col in features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_lim = Q1 - 1.5 * IQR
    upper_lim = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_lim) | (df[col] > upper_lim)]
    outliers_summary[col] = outliers
    print(f"- {col}:")
    print(f"  Límites IQR: [{lower_lim:.2f}, {upper_lim:.2f}] | Outliers detectados: {len(outliers)}")
    if len(outliers) > 0:
        print(f"  IDs de clientes atípicos: {outliers['CustomerID'].tolist()}")
        print(f"  Valores: {outliers[col].tolist()}")
"""))

cells.append(nbf.v4.new_code_cell("""# Visualización de distribuciones y diagramas de caja (Boxplots)
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
colors = ['#1f77b4', '#2ca02c', '#d62728']

for i, col in enumerate(features):
    # Histograma con KDE
    sns.histplot(df[col], kde=True, ax=axes[0, i], color=colors[i], bins=15)
    axes[0, i].set_title(f'Distribución de {col}', fontsize=12, fontweight='bold')
    axes[0, i].set_xlabel('')
    axes[0, i].set_ylabel('Frecuencia')
    
    # Boxplot
    sns.boxplot(x=df[col], ax=axes[1, i], color=colors[i])
    axes[1, i].set_title(f'Diagrama de Caja de {col}', fontsize=12, fontweight='bold')
    axes[1, i].set_xlabel(col)

plt.suptitle('Análisis Univariado y Detección de Valores Atípicos', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### ❓ Pregunta 1.2: ¿Se identifican valores atípicos?

**Respuesta y Diagnóstico Metodológico:**

- **Resultado Estadístico:**
  - En la variable **`Age`**: **0 valores atípicos** (todas las edades se ubican entre 18 y 70 años, dentro del rango normal $[-1.63, 79.38]$).
  - En la variable **`Spending Score (1-100)`**: **0 valores atípicos** (los puntajes oscilan entre 1 y 99 puntos, dentro del rango $[-22.62, 130.38]$).
  - En la variable **`Annual Income (k$)`**: Se identificaron exactamente **2 valores atípicos** (Límite superior: $132.75$ k$):
    - **Cliente ID 199:** Ingreso = $137$ k$, Gasto = $18$ pts.
    - **Cliente ID 200:** Ingreso = $137$ k$, Gasto = $83$ pts.
- **Decisión de Negocio y Modelado:**
  - Estos dos registros **NO** corresponden a errores de captura o ruido instrumental, sino a clientes reales de ultra-alto poder adquisitivo (*Ultra-High-Net-Worth*).
  - En el contexto de un centro comercial, el segmento de máxima capacidad financiera es precisamente uno de los más valiosos para el diseño de estrategias comerciales prémium. Eliminarlos sesgaría la representatividad del segmento VIP. Por tanto, se conservan en el dataset y su impacto en las distancias se equilibra mediante la normalización con `StandardScaler`.
"""))

cells.append(nbf.v4.new_code_cell("""# 3. Relaciones bivariadas y Matriz de Correlación
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Dispersión Ingreso Anual vs Puntaje de Gasto
sns.scatterplot(
    data=df, 
    x='Annual Income (k$)', 
    y='Spending Score (1-100)', 
    hue='Gender', 
    palette={'Male': '#1f77b4', 'Female': '#e377c2'},
    s=75, 
    alpha=0.85, 
    ax=axes[0]
)
axes[0].set_title('Ingreso Anual vs Puntaje de Gasto (por Género)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Ingreso Anual (k$)', fontsize=11)
axes[0].set_ylabel('Puntaje de Gasto (1-100)', fontsize=11)

# Heatmap de correlación lineal de Pearson
corr = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f', vmin=-1, vmax=1, ax=axes[1], cbar=True)
axes[1].set_title('Matriz de Correlación Lineal (Pearson)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### ❓ Pregunta 1.3: ¿Qué tipo de relaciones se observan entre las variables?

**Hallazgos Clave:**

1. **Relación No Lineal entre Ingreso Anual y Puntaje de Gasto ($r = 0.0099 \approx 0$):**
   - La correlación lineal global es prácticamente nula. Esto indica que un mayor ingreso no implica automáticamente mayor o menor gasto.
   - No obstante, la gráfica de dispersión revela un patrón no lineal muy claro en forma de abanico con 5 conglomerados densos bien definidos:
     - Ingreso bajo con gasto bajo.
     - Ingreso bajo con gasto alto.
     - Ingreso medio con gasto medio (el grupo más denso en el centro).
     - Ingreso alto con gasto bajo.
     - Ingreso alto con gasto alto.
2. **Relación Inversa Moderada entre Edad y Puntaje de Gasto ($r = -0.327$):**
   - Los clientes jóvenes (< 40 años) tienden a registrar puntajes de gasto superiores (frecuentemente entre 60 y 99 puntos).
   - Los clientes mayores de 40 años muestran una disminución gradual en su nivel de gasto hacia rangos moderados (30 a 60 puntos).
3. **Independencia entre Edad e Ingreso ($r = -0.012$):**
   - El nivel de ingresos está uniformemente repartido a lo largo de las distintas edades de los clientes del mall.
"""))

# 4. PARTE 2 - PREPROCESAMIENTO
cells.append(nbf.v4.new_markdown_cell("""---
## PARTE 2 – Preprocesamiento de Datos

En esta etapa preparamos la matriz de características $X$ y aplicamos normalización con `StandardScaler`.
"""))

cells.append(nbf.v4.new_code_cell("""# Selección de características para el modelo
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Normalización mediante StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Verificación estadística del escalado
scaled_df = pd.DataFrame(X_scaled, columns=['Annual_Income_Scaled', 'Spending_Score_Scaled'])
print("=== VERIFICACIÓN DEL ESCALADO Z-SCORE ===")
print("Medias (deben ser cercanas a 0):")
print(scaled_df.mean().round(4))
print("\\nDesviaciones estándar (deben ser iguales a 1):")
print(scaled_df.std().round(4))
scaled_df.head()
"""))

cells.append(nbf.v4.new_markdown_cell("""### ❓ Pregunta 2.1: ¿Por qué es necesario escalar las variables en el algoritmo K-Means?

**Fundamentación Teórica y Matemática:**

1. **Sensibilidad a la Distancia Euclidiana:**
   - K-Means es un algoritmo de agrupamiento no supervisado basado en distancias. En cada iteración asigna una observación $x_i$ al cluster cuyo centroide $\\mu_k$ minimice la distancia euclidiana:
   $$d(x_i, \\mu_k) = \\sqrt{\\sum_{j=1}^p (x_{ij} - \\mu_{kj})^2}$$
   - Si una variable presenta un rango de valores o varianza mayor (ej. ingresos en escala de decenas de miles vs puntajes de 1 a 100), la magnitud cuadrática de esa variable domina casi en su totalidad el cálculo de la distancia, anulando el peso de las demás variables.
2. **Supuesto de Isotropía de los Clusters:**
   - K-Means asume implícitamente que los clusters son hiper-esferas isotrópicas (con la misma dispersión en todos los ejes). Sin escalado, el espacio euclidiano se deforma en elipsoides elongados y los centroides se mueven casi exclusivamente sobre la dimensión de mayor escala.
3. **Equidad Ponderal Mediante Estandarización Z-Score:**
   - Al estandarizar mediante $z = \\frac{x - \\mu}{\\sigma}$, ambas dimensiones se expresan en unidades adimensionales de desviación estándar, garantizando que el Ingreso Anual y el Puntaje de Gasto contribuyan equitativamente (50% y 50%) a la función de pérdida del algoritmo.
"""))

# 5. PARTE 3 - MÉTODO DEL CODO
cells.append(nbf.v4.new_markdown_cell("""---
## PARTE 3 – Método del Codo (Elbow Method)

El método del codo evalúa la inercia del modelo o **WCSS** (*Within-Cluster Sum of Squares*) para distintos valores de $K$:
$$WCSS = \\sum_{k=1}^K \\sum_{x_i \\in C_k} ||x_i - \\mu_k||^2$$
"""))

cells.append(nbf.v4.new_code_cell("""# Cálculo de WCSS / Inercia para K entre 1 y 10
K_range = range(1, 11)
wcss = []

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    kmeans_temp.fit(X_scaled)
    wcss.append(kmeans_temp.inertia_)

# Tabla de resultados de Inercia
elbow_table = pd.DataFrame({'K (Número de Clusters)': list(K_range), 'Inercia (WCSS)': np.round(wcss, 3)})
elbow_table['Reducción Porcentual (%)'] = (-elbow_table['Inercia (WCSS)'].diff() / elbow_table['Inercia (WCSS)'].shift(1) * 100).round(2)
display(elbow_table)
"""))

cells.append(nbf.v4.new_code_cell("""# Gráfica del Método del Codo
plt.figure(figsize=(9, 5.5))
plt.plot(K_range, wcss, marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label='Inercia (WCSS)')
plt.axvline(x=5, color='#d62728', linestyle='--', linewidth=1.8, label='Punto de Inflexión / Codo Óptimo (K=5)')
plt.title('Método del Codo para Seleccionar K Óptimo', fontsize=14, fontweight='bold')
plt.xlabel('Número de Clusters (K)', fontsize=12)
plt.ylabel('Inercia Intra-Cluster (WCSS)', fontsize=12)
plt.xticks(K_range)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### ❓ Preguntas de la Parte 3:

#### 1. ¿Por qué valores cercanos a cero indican mejor ajuste?
- **Cohesión Interna Máxima:** La inercia representa el error cuadrático residual de la partición. Un valor de WCSS cercano a cero significa que cada dato individual se encuentra muy cercano geométricamente al centroide de su cluster, lo que refleja clusters altamente compactos y homogéneos.
- **Riesgo de Sobreajuste (Overfitting):** Si $K = N$ (donde cada cliente es su propio cluster), el WCSS sería exactamente $0$. Sin embargo, esto es un modelo trivial inútil para la toma de decisiones. Por ello, se busca el punto de balance óptimo entre compacidad interna y simplicidad del modelo.

#### 2. ¿Cómo se identifica el punto óptimo de K?
- Se identifica mediante el **punto de inflexión o 'codo'** en la curva decreciente de WCSS vs. $K$.
- Antes del codo ($K < 5$), añadir un cluster genera descensos bruscos y sustanciales en la inercia (reducciones superiores al 30-50%).
- A partir de $K = 5$, la pendiente de la curva se vuelve plana y suave, indicando rendimientos decrecientes: añadir más clusters sólo fragmenta grupos naturales sin aportar ganancia explicativa apreciable. Por tanto, el codo óptimo se sitúa claramente en **$K = 5$**.
"""))

# 6. PARTE 4 - COEFICIENTE DE SILUETA
cells.append(nbf.v4.new_markdown_cell("""---
## PARTE 4 – Coeficiente de Silueta (Silhouette Score)

El coeficiente de silueta evalúa simultáneamente la **cohesión intra-cluster** $a(i)$ y la **separación inter-cluster** $b(i)$:
$$s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}, \\quad s(i) \\in [-1, 1]$$
"""))

cells.append(nbf.v4.new_code_cell("""# Cálculo del Silhouette Score para K entre 2 y 10
K_sil = range(2, 11)
silhouette_scores = []

for k in K_sil:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

sil_df = pd.DataFrame({'K': list(K_sil), 'Silhouette Score': np.round(silhouette_scores, 4)})
sil_df['Evaluación'] = sil_df['Silhouette Score'].apply(
    lambda s: 'Óptimo Global Máximo' if s == max(silhouette_scores) else ('Bueno' if s > 0.45 else 'Regular')
)
display(sil_df)
"""))

cells.append(nbf.v4.new_code_cell("""# Gráfica del Coeficiente de Silueta
plt.figure(figsize=(9, 5.5))
plt.plot(K_sil, silhouette_scores, marker='s', linewidth=2.5, markersize=8, color='#2ca02c', label='Silhouette Score')
max_score = max(silhouette_scores)
best_k = K_sil[silhouette_scores.index(max_score)]
plt.axvline(x=best_k, color='#d62728', linestyle='--', linewidth=1.8, label=f'Máximo Global en K={best_k} ({max_score:.4f})')
plt.title('Coeficiente de Silueta Promedio vs Número de Clusters (K)', fontsize=14, fontweight='bold')
plt.xlabel('Número de Clusters (K)', fontsize=12)
plt.ylabel('Coeficiente de Silueta Promedio', fontsize=12)
plt.xticks(K_sil)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### ❓ Preguntas de la Parte 4:

#### 1. ¿Qué mide el coeficiente de silueta?
- Cuantifica la calidad intrínseca del agrupamiento evaluando dos propiedades esenciales:
  1. **Cohesión ($a(i)$):** Qué tan cerca está cada punto de los miembros de su propio cluster.
  2. **Separación ($b(i)$):** Qué tan lejos está cada punto de los miembros del cluster vecino más cercano.

#### 2. ¿Qué significa un valor cercano a 1?
- Significa una **excelente calidad de agrupamiento**: la muestra está muy cerca de los compañeros de su cluster ($a(i) \\to 0$) y muy lejos del cluster adyacente ($a(i) \\ll b(i)$). Los clusters resultantes son compactos y se encuentran claramente aislados en el espacio.

#### 3. ¿Qué implica un valor cercano a 0 o negativo?
- **Cercano a 0:** La muestra se ubica en el límite o frontera de decisión entre dos clusters vecinos ($a(i) \\approx b(i)$), denotando ambigüedad en la asignación.
- **Negativo ($< 0$):** La muestra está en promedio más cerca del cluster vecino que de su propio cluster asignado ($a(i) > b(i)$), lo cual representa un error de asignación o mala clasificación.

#### 4. ¿Coincide el valor óptimo de K con el método del codo? Justifique.
- **Sí, coinciden con total exactitud y rigor:**
  - El método del codo detectó su punto de quiebre en **$K = 5$**.
  - La curva del coeficiente de silueta alcanza su **pico máximo absoluto precisamente en $K = 5$ con un score de $0.5547$**.
  - Ambos criterios convergen independientemente en la misma solución matemática: particionar la población en 5 clusters produce la mejor combinación de cohesión intra-cluster y separación inter-cluster.
"""))

# 7. PARTE 5 - MODELO K-MEANS
cells.append(nbf.v4.new_markdown_cell("""---
## PARTE 5 – Modelo K-Means Definitivo (K = 5)

Entrenamos el modelo final con $K = 5$, asignamos las etiquetas a cada cliente y recuperamos los centroides tanto en escala estandarizada como en la escala original.
"""))

cells.append(nbf.v4.new_code_cell("""# Entrenamiento del modelo definitivo
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=20, max_iter=300, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Recuperación de centroides
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)

centroids_df = pd.DataFrame(centroids_original, columns=['Ingreso Anual Promedio (k$)', 'Puntaje de Gasto Promedio (1-100)'])
centroids_df.index = [f'Cluster {i}' for i in range(optimal_k)]
centroids_df['Total Clientes'] = [ (df['Cluster'] == i).sum() for i in range(optimal_k) ]
centroids_df['Porcentaje (%)'] = (centroids_df['Total Clientes'] / len(df) * 100).round(2)
display(centroids_df.round(2))
"""))

cells.append(nbf.v4.new_code_cell("""# Visualización gráfica de los Clusters y sus Centroides
plt.figure(figsize=(10, 6.5))
cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(
        cluster_data['Annual Income (k$)'], 
        cluster_data['Spending Score (1-100)'],
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
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### ❓ Preguntas de la Parte 5:

#### 1. ¿Qué representa cada cluster?
- **Cluster 0 (Ingreso Medio, Gasto Medio):** Representa al cliente promedio estándar (40.5% del total). Tienen un ingreso anual moderado (~$55.30k) y un nivel de gasto moderado (~49.52 pts).
- **Cluster 1 (Ingreso Alto, Gasto Alto):** Representa a los clientes VIP / Élite comercial (~$86.54k de ingreso y ~82.13 pts de gasto). Alta capacidad económica y máxima disposición a consumir.
- **Cluster 2 (Ingreso Bajo, Gasto Alto):** Representa a los consumidores impulsivos o jóvenes (~$25.73k de ingreso y ~79.36 pts de gasto). A pesar de sus bajos ingresos, tienen un gasto muy elevado.
- **Cluster 3 (Ingreso Alto, Gasto Bajo):** Representa a los ahorradores cautelosos (~$88.20k de ingreso y ~17.11 pts de gasto). Alto poder adquisitivo pero reticentes al consumo en el centro comercial.
- **Cluster 4 (Ingreso Bajo, Gasto Bajo):** Representa a los consumidores de subsistencia o frugales (~$26.30k de ingreso y ~20.91 pts de gasto). Capacidad limitada y gasto estrictamente necesario.

#### 2. ¿Qué diferencias existen entre los grupos obtenidos?
- **Diferencia de Capacidad Económica (Ingresos):** Se observa una marcada polarización entre los grupos de altos ingresos (~$87k) y los de bajos ingresos (~$26k).
- **Diferencia de Propensión al Gasto:** Entre los grupos de alto gasto (scores ~80) y los de bajo gasto (scores ~18-20) existe una brecha de más de 60 puntos, reflejando patrones de estilo de vida diametralmente opuestos.
"""))

# 8. PARTE 6 - INTERPRETACIÓN DE NEGOCIO
cells.append(nbf.v4.new_markdown_cell("""---
## PARTE 6 – Interpretación de Negocio y Estrategias de Marketing

Clasificamos los clusters en perfiles formales de negocio:
- **Clientes de alto valor**
- **Clientes promedio**
- **Clientes de bajo valor**
- **Clientes con potencial**
"""))

cells.append(nbf.v4.new_code_cell("""# Mapeo de Clusters a Perfiles de Negocio
def clasificar_perfil(cluster_id, income, spending):
    if income > 70 and spending > 70:
        return "Clientes de Alto Valor (VIP / Target Premium)", "Alto Valor"
    elif income > 70 and spending < 40:
        return "Clientes con Potencial - Ahorradores Cautelosos", "Con Potencial"
    elif income < 40 and spending > 60:
        return "Clientes con Potencial - Consumidores Impulsivos", "Con Potencial"
    elif income < 40 and spending < 40:
        return "Clientes de Bajo Valor (Consumo Frugal)", "Bajo Valor"
    else:
        return "Clientes Promedio (Población Central)", "Promedio"

# Generación del mapeo
mapping_info = {}
for i in range(optimal_k):
    inc = centroids_original[i, 0]
    sp = centroids_original[i, 1]
    perfil, cat = clasificar_perfil(i, inc, sp)
    mapping_info[i] = (perfil, cat)

df['Perfil_Negocio'] = df['Cluster'].map(lambda c: mapping_info[c][0])
df['Categoria'] = df['Cluster'].map(lambda c: mapping_info[c][1])

# Resumen ejecutivo por perfil
resumen_negocio = df.groupby(['Cluster', 'Perfil_Negocio', 'Categoria']).agg(
    Clientes=('CustomerID', 'count'),
    Edad_Media=('Age', 'mean'),
    Ingreso_Medio_k=('Annual Income (k$)', 'mean'),
    Gasto_Medio=('Spending Score (1-100)', 'mean'),
    Porcentaje_Mujeres=('Gender', lambda g: (g == 'Female').mean() * 100)
).round(2).reset_index()

display(resumen_negocio)
"""))

cells.append(nbf.v4.new_code_cell("""# Visualización del Mapa Estratégico de Negocio
plt.figure(figsize=(11, 7))
sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Perfil_Negocio',
    palette='Set1',
    s=85,
    alpha=0.9,
    edgecolor='k',
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
plt.title('Mapa Estratégico de Segmentación de Clientes', fontsize=14, fontweight='bold')
plt.xlabel('Ingreso Anual (k$)', fontsize=12)
plt.ylabel('Puntaje de Gasto (1-100)', fontsize=12)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### ❓ Preguntas de la Parte 6:

#### 1. ¿Qué estrategias de marketing se podrían aplicar a cada grupo?

1. **Clientes de Alto Valor (Alto Ingreso, Alto Gasto):**
   - **Estrategia:** Retención, fidelización VIP y maximización de experiencia (*White Glove Service*).
   - **Tácticas:** Eventos y preventas privadas de colecciones de lujo, servicio de *personal shopper*, acceso a salas lounge exclusivas y programas de lealtad basados en privilegios vivenciales (no en descuentos).
2. **Clientes con Potencial - Ahorradores Cautelosos (Alto Ingreso, Bajo Gasto):**
   - **Estrategia:** Activación y seducción basada en valor intrínseco, calidad y prestigio.
   - **Tácticas:** Demostraciones de productos de alta gama y durabilidad garantizada, gastronomía gourmet, membresías con *cashback* financiero y comunicación que enfatice la sofisticación del ahorro inteligente.
3. **Clientes con Potencial - Consumidores Impulsivos (Bajo Ingreso, Alto Gasto):**
   - **Estrategia:** *Engagement* digital y facilidades de pago responsables.
   - **Tácticas:** Modelos de financiamiento *Buy Now, Pay Later* (BNPL) con cuotas sin interés en montos moderados; campañas orientadas a redes sociales (Instagram/TikTok) en tendencias de moda joven y ventas *flash*.
4. **Clientes Promedio (Ingreso Medio, Gasto Medio):**
   - **Estrategia:** Venta cruzada (*Cross-selling*) y promociones de volumen.
   - **Tácticas:** Combos familiares de fin de semana (compras + cine + restaurantes), programas de fidelización basados en acumulación de puntos canjeables y promociones estacionales masivas (Navidad, Black Friday).
5. **Clientes de Bajo Valor (Bajo Ingreso, Bajo Gasto):**
   - **Estrategia:** Estimulación del ticket mínimo con precios accesibles.
   - **Tácticas:** Ferias de saldos y marcas outlet, promociones 2x1 y líneas de productos esenciales accesibles.

---

#### 2. ¿Qué segmento representa mayor oportunidad de crecimiento?
- **El segmento de mayor oportunidad de crecimiento es el de 'Clientes con Potencial - Ahorradores Cautelosos' (Alto Ingreso, Bajo Gasto).**
- **Justificación Estratégica:**
  1. **Disponibilidad Real de Capital:** Poseen el ingreso promedio más elevado de toda la población (~$88.20k anuales). A diferencia de los grupos de bajos ingresos (cuya capacidad de compra está restringida por una barrera de liquidez), estos clientes tienen el dinero en sus cuentas bancarias.
  2. **Alto Margen de Expansión (*Upside*):** Su índice de gasto actual es de tan solo 17.11 puntos. Si la gerencia de marketing logra aumentar su puntaje en 30 puntos mediante marcas de alta gama y experiencias atractivas, este grupo se transformaría en clientes de Alto Valor, generando el mayor retorno sobre la inversión (ROI) para el centro comercial con solo 35 clientes de ticket alto.
"""))

# 9. PARTE 7 - PREPARACIÓN PARA STREAMLIT
cells.append(nbf.v4.new_markdown_cell("""---
## PARTE 7 – Preparación para Despliegue con Streamlit

Exportamos el modelo K-Means entrenado y el estimador `StandardScaler` en formato `.joblib`/`.pkl` para que puedan ser consumidos en la aplicación interactiva de Streamlit (`app.py`).
"""))

cells.append(nbf.v4.new_code_cell("""# Guardado de artefactos para el despliegue
joblib.dump(kmeans, 'kmeans_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("✅ Modelo K-Means y StandardScaler exportados exitosamente para Streamlit.")
"""))

nb['cells'] = cells
nb['metadata'] = {
    'language_info': {
        'name': 'python',
        'version': '3.12'
    },
    'colab': {
        'name': 'Taller_Final_KMeans_Mall_Customers.ipynb',
        'provenance': []
    }
}

OUTPUT_NB = 'taller_kmeans_mall_customers.ipynb'
with open(OUTPUT_NB, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Cuaderno Jupyter generado exitosamente: {OUTPUT_NB}")
