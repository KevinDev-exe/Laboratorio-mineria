# 🛍️ Taller de Minería de Datos: Segmentación de Clientes con K-Means

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=KevinDev-exe/Laboratorio-mineria&branch=main&mainModule=app.py)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KevinDev-exe/Laboratorio-mineria/blob/main/taller_kmeans_mall_customers.ipynb)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App%20Interactiva-FF4B4B?logo=streamlit)
![Status](https://img.shields.io/badge/Status-24%2F7%20Online-brightgreen)

Proyecto integral de análisis exploratorio, modelado de aprendizaje no supervisado, evaluación de clusters y despliegue interactivo para la segmentación estratégica de clientes de un centro comercial (*Mall Customers*), utilizando el algoritmo **K-Means Clustering**, validación mediante **Método del Codo** y **Coeficiente de Silueta**, y una aplicación web en **Streamlit**.

---

## 🎯 Objetivo del Proyecto

Identificar patrones de comportamiento en los visitantes de un centro comercial en función de su **Ingreso Anual (k$)** y su **Puntuación de Gasto (1-100)** para diseñar estrategias de marketing segmentadas, optimizar el retorno de inversión publicitaria y clasificar nuevos clientes en tiempo real.

---

## 📁 Estructura del Repositorio

```text
Laboratorio-mineria/
│
├── app.py                                             # Aplicación Web Interactiva en Streamlit (Parte 7)
├── Mall_Customers.csv                                 # Dataset oficial con 200 registros de clientes
├── taller_kmeans_mall_customers.ipynb                 # Cuaderno Jupyter interactivo (Local y Google Colab)
├── Respuestas_Taller_KMeans.docx                      # Informe formal en Word con todas las respuestas (Partes 1-7)
├── Taller_Final_Mineria_de_Datos_KMeans_Completo.docx # Copia maestra del documento Word completo
├── generate_analysis_and_report.py                    # Script integral: EDA, K-Means, gráficos e informe Word
├── create_notebook.py                                 # Generador programático del cuaderno Jupyter
├── requirements.txt                                   # Dependencias del proyecto para entorno local y Streamlit Cloud
│
├── graficos/                                          # Visualizaciones en alta resolución (300 DPI)
│   ├── eda_distribuciones_outliers.png
│   ├── eda_relaciones_correlacion.png
│   ├── metodo_del_codo.png
│   ├── coeficiente_de_silueta.png
│   ├── clusters_kmeans_final.png
│   └── mapa_segmentos_negocio.png
│
└── README.md                                          # Documentación del proyecto
```

---

## 🔍 Metodología Aplicada (Partes 1 a 7)

1. **Parte 1 – Exploración de Datos (EDA):**
   - Estadísticas descriptivas completas de edad, ingresos y puntajes de gasto.
   - Detección de valores atípicos mediante el Rango Intercuartílico ($IQR$): identificación de 2 outliers superiores en ingreso anual ($137\text{k}\$$).
   - Análisis de relaciones bivariadas y matriz de correlación de Pearson.

2. **Parte 2 – Preprocesamiento:**
   - Estandarización Z-Score con `StandardScaler` ($\mu=0, \sigma=1$) para garantizar equidad geométrica en el cálculo de distancias euclidianas.

3. **Parte 3 y 4 – Determinación del Número Óptimo de Clusters ($K$):**
   - **Método del Codo:** Evaluación de la inercia intra-cluster ($WCSS$) para $K \in [1, 10]$, detectando el punto de inflexión en **$K = 5$** ($WCSS = 65.57$).
   - **Coeficiente de Silueta:** Máxima cohesión y separación validada exactamente en **$K = 5$** ($\text{Silhouette Score} = 0.5547$).

4. **Parte 5 y 6 – Modelo K-Means e Interpretación de Negocio ($K = 5$ Grupos):**
   - **👑 Clientes de Alto Valor / VIP (Alto Ingreso, Alto Gasto):** El grupo de mayor rentabilidad. Estrategia: Fidelización exclusiva, preventas VIP y atención prémium.
   - **💎 Clientes con Potencial - Ahorradores Cautelosos (Alto Ingreso, Bajo Gasto):** Mayor oportunidad de crecimiento (ROI). Estrategia: Propuesta de valor basada en calidad, durabilidad y prestigio.
   - **⚡ Clientes con Potencial - Consumidores Impulsivos (Bajo Ingreso, Alto Gasto):** Alta receptividad a tendencias jóvenes. Estrategia: Pagos flexibles (BNPL) y marketing digital en redes sociales.
   - **⚖️ Clientes Promedio (Ingreso Medio, Gasto Medio):** Núcleo del tráfico del centro comercial (40.5% de clientes). Estrategia: Venta cruzada (*cross-selling*) y paquetes familiares.
   - **🏷️ Clientes de Bajo Valor (Bajo Ingreso, Bajo Gasto):** Sensibles al precio. Estrategia: Descuentos, promociones 2x1 y zonas *outlet*.

5. **Parte 7 – Despliegue Interactivo con Streamlit (`app.py`):**
   - Visualización interactiva del método del codo y silueta con Plotly.
   - Exploración de los 5 clusters generados y sus centroides.
   - Clasificador de nuevos clientes en tiempo real mediante controles deslizantes (*sliders*) y botones rápidos de prueba por arquetipo.

---

## 🚀 Instalación y Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/KevinDev-exe/Laboratorio-mineria.git
cd Laboratorio-mineria
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar la Aplicación Interactiva en Streamlit (Parte 7)
```bash
streamlit run app.py
```
La aplicación se abrirá en tu navegador en `http://localhost:8501`.

### 4. Abrir el Cuaderno Jupyter (Local o Google Colab)
```bash
jupyter notebook taller_kmeans_mall_customers.ipynb
```
*(El cuaderno cuenta con carga híbrida automática del dataset para funcionar directamente en Google Colab sin configuraciones manuales)*.

### 5. Regenerar el Informe Word y Gráficos
```bash
python generate_analysis_and_report.py
```

---

## 👤 Autor
* **Desarrollador / Estudiante:** KevinDev-exe
* **Asignatura:** Minería de Datos & Inteligencia de Negocios
