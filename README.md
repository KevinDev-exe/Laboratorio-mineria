# 🛍️ Taller de Minería de Datos: Segmentación de Clientes con K-Means

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)
![Status](https://img.shields.io/badge/Status-Completado-brightgreen)

Proyecto de análisis exploratorio, modelado de aprendizaje no supervisado y segmentación estratégica de clientes para un centro comercial (*Mall Customers*), utilizando el algoritmo **K-Means Clustering** y técnicas de validación como el **Método del Codo** y el **Coeficiente de Silueta**.

---

## 🎯 Objetivo del Proyecto

Identificar patrones de comportamiento en los visitantes de un centro comercial en función de su **Ingreso Anual (k$)** y su **Puntuación de Gasto (1-100)** para diseñar estrategias de marketing segmentadas, optimizar el retorno de inversión publicitaria y maximizar la fidelización de clientes.

---

## 📁 Estructura del Repositorio

```text
Laboratorio-mineria/
│
├── Mall_Customers.csv                 # Dataset con 200 registros de clientes
├── generate_analysis_and_report.py    # Script integral: EDA, K-Means, gráficos e informe Word
├── create_notebook.py                 # Generador programático del cuaderno Jupyter
├── taller_kmeans_mall_customers.ipynb # Cuaderno interactivo (Local y Google Colab)
├── Respuestas_Taller_KMeans.docx      # Informe formal en Word con respuestas del taller
│
├── graficos/                          # Visualizaciones en alta resolución (300 DPI)
│   ├── eda_distribuciones_outliers.png
│   ├── eda_relaciones_correlacion.png
│   ├── metodo_del_codo.png
│   ├── coeficiente_de_silueta.png
│   ├── clusters_kmeans_final.png
│   └── mapa_segmentos_negocio.png
│
└── README.md                          # Documentación del proyecto
```

---

## 🔍 Metodología Aplicada

1. **Análisis Exploratorio de Datos (EDA):**
   - Estadísticas descriptivas de edad, ingresos y puntajes de gasto.
   - Detección de valores atípicos mediante el Rango Intercuartílico ($IQR$).
   - Análisis de correlación y distribuciones por género.

2. **Preprocesamiento:**
   - Estandarización de variables con `StandardScaler` para balancear la escala de ingresos y puntajes de gasto.

3. **Determinación del Número Óptimo de Clusters ($k$):**
   - **Método del Codo:** Evaluación de la inercia intra-cluster ($WCSS$) para $k \in [2, 10]$, detectando el punto de inflexión en $k = 5$.
   - **Coeficiente de Silueta:** Máxima cohesión y separación validada con $k = 5$ (Score $\approx 0.55$).

4. **Segmentación Estratégica ($k = 5$ Grupos):**
   - **👑 Clientes VIP (Alto Ingreso, Alto Gasto):** El grupo de mayor rentabilidad. Estrategia: Clubes de fidelización exclusivos y atención prémium.
   - **💎 Ahorradores Cautelosos (Alto Ingreso, Bajo Gasto):** Mayor potencial de crecimiento. Estrategia: Ofertas de alto valor percibido, calidad y durabilidad.
   - **🛍️ Consumidores Impulsivos (Bajo Ingreso, Alto Gasto):** Alta receptividad a tendencias. Estrategia: Micro-financiamiento, cuotas y campañas en redes sociales.
   - **⚖️ Clientes Promedio (Ingreso Medio, Gasto Medio):** El núcleo del volumen transaccional (~40% de los clientes). Estrategia: Venta cruzada (*cross-selling*) y promociones familiares.
   - **🏷️ Clientes de Bajo Valor (Bajo Ingreso, Bajo Gasto):** Sensibles al precio. Estrategia: Descuentos masivos, ferias tipo *outlet* y líneas accesibles.

---

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/KevinDev-exe/Laboratorio-mineria.git
cd Laboratorio-mineria
```

### 2. Instalar dependencias
Se recomienda un entorno virtual con Python 3.9 o superior:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn python-docx notebook
```

### 3. Ejecutar el análisis y regenerar el informe
Para correr todo el pipeline analítico y generar el archivo Word junto con los gráficos:
```bash
python generate_analysis_and_report.py
```

### 4. Abrir el Cuaderno Interactivo
```bash
jupyter notebook taller_kmeans_mall_customers.ipynb
```
*(También puedes abrirlo directamente en Google Colab o en VS Code)*.

---

## 📊 Resultados Visuales

El análisis produce las figuras disponibles en la carpeta [`graficos/`](graficos/):
- **Clusters Finales:** Mapeo de centroides y asignación de cada cliente en el plano dimensional.
- **Validación Matemática:** Gráficas del Codo y Silueta justificando $k=5$.
- **Mapa de Oportunidades de Negocio:** Clasificación estratégica con directrices de marketing.

---

## 👤 Autor
* **Desarrollador / Estudiante:** KevinDev-exe
* **Asignatura:** Minería de Datos & Inteligencia de Negocios
