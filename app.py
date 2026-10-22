"""
Aplicación Web Interactiva de Despliegue con Streamlit:
Segmentación Estratégica de Clientes con K-Means (Mall Customers)
Parte 7 del Taller Final de Minería de Datos.
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------------------------------------------------
# Configuración inicial de la página
# -------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation AI | K-Means",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1A365D;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4A5568;
        margin-bottom: 20px;
    }
    .profile-card {
        background: linear-gradient(135deg, #1A365D 0%, #2B6CB0 100%);
        color: white;
        border-radius: 12px;
        padding: 22px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    }
    .preset-btn {
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Carga de datos y modelo K-Means
# -------------------------------------------------------------
DATA_PATH = "Mall_Customers.csv"
DATA_URL = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_resource
def compute_clustering(df, n_clusters=5):
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20, max_iter=300, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Centroides en escala original
    centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Mapeo descriptivo según la posición de centroides
    profile_dict = {}
    for i, (inc, sp) in enumerate(centroids_orig):
        if inc > 70 and sp > 70:
            nombre = "Clientes de Alto Valor (VIP / Target Premium)"
            cat = "Alto Valor"
            desc = "Ingresos elevados y altísima disposición de compra. Constituyen el núcleo de facturación de lujo."
            estrategia = "Fidelización VIP, eventos privados, personal shopper y salones lounge exclusivos."
            color = "#2ca02c"
        elif inc > 70 and sp < 40:
            nombre = "Clientes con Potencial - Ahorradores Cautelosos"
            cat = "Con Potencial (Mayor ROI)"
            desc = "Alto poder adquisitivo pero baja afinidad o reticencia a la oferta actual del centro comercial."
            estrategia = "Campañas de calidad, origen y durabilidad, catas gourmet y membresías con beneficios financieros."
            color = "#ff7f0e"
        elif inc < 40 and sp > 60:
            nombre = "Clientes con Potencial - Consumidores Impulsivos"
            cat = "Con Potencial"
            desc = "Bajo ingreso pero puntaje de gasto muy elevado. Comportamiento guiado por tendencias y moda."
            estrategia = "Facilidades de pago BNPL, promociones dinámicas en redes sociales (TikTok/IG) y ventas flash."
            color = "#d62728"
        elif inc < 40 and sp < 40:
            nombre = "Clientes de Bajo Valor (Consumo Frugal)"
            cat = "Bajo Valor"
            desc = "Bajo ingreso y gasto mínimo. Consumen solo lo estrictamente necesario."
            estrategia = "Ofertas 2x1, zonas outlet, marcas de distribuidor y promociones de canasta básica."
            color = "#7f7f7f"
        else:
            nombre = "Clientes Promedio (Población Central)"
            cat = "Promedio"
            desc = "Ingreso medio y gasto moderado. Es el segmento más abundante y constante del mall."
            estrategia = "Venta cruzada (cine + restaurantes + compras), planes familiares y cupones de fidelización."
            color = "#1f77b4"
        
        profile_dict[i] = {
            "nombre": nombre,
            "categoria": cat,
            "descripcion": desc,
            "estrategia": estrategia,
            "color": color,
            "inc_center": inc,
            "sp_center": sp
        }
        
    return scaler, kmeans, X_scaled, clusters, centroids_orig, profile_dict

@st.cache_data
def compute_evaluation_curves(X_scaled):
    K_range = list(range(1, 11))
    wcss = []
    for k in K_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        
    sil_scores = []
    sil_k = list(range(2, 11))
    for k in sil_k:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, labels))
        
    return K_range, wcss, sil_k, sil_scores

# -------------------------------------------------------------
# Carga inicial
# -------------------------------------------------------------
df_raw = load_data()
df = df_raw.copy()

# Sidebar: Configuración
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=85)
st.sidebar.title("Configuración")

k_selected = st.sidebar.slider(
    "Número de Clusters (K):",
    min_value=2,
    max_value=8,
    value=5,
    help="El valor óptimo demostrado mediante el método del codo y el coeficiente de silueta es K=5."
)

if k_selected != 5:
    st.sidebar.warning(f"⚠️ K={k_selected}. Recuerda que K=5 es el óptimo demostrado teóricamente.")

scaler, kmeans, X_scaled, clusters, centroids_orig, profile_dict = compute_clustering(df, n_clusters=k_selected)
df['Cluster'] = clusters
df['Perfil'] = df['Cluster'].map(lambda c: profile_dict[c]['nombre'])
df['Categoria'] = df['Cluster'].map(lambda c: profile_dict[c]['categoria'])

# -------------------------------------------------------------
# Encabezado
# -------------------------------------------------------------
st.markdown('<p class="main-title">🛍️ Segmentación de Clientes con K-Means</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Taller Final de Minería de Datos | Despliegue Interactivo con Streamlit (Parte 7)</p>', unsafe_allow_html=True)

# -------------------------------------------------------------
# Tabs de navegación
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Clasificador de Nuevos Clientes (En Vivo)",
    "📊 Visualización de Clusters",
    "📈 Método del Codo y Silueta",
    "💼 Perfiles y Estrategias de Negocio"
])

# -------------------------------------------------------------
# TAB 1: CLASIFICADOR REACTIVO EN TIEMPO REAL
# -------------------------------------------------------------
with tab1:
    st.header("🎯 Clasificación de Nuevos Clientes en Tiempo Real")
    st.write("Modifica los valores del cliente o utiliza los **botones rápidos de prueba**. La predicción y la visualización se recalculan **instantáneamente**:")

    # Inicializar session_state si no existe
    if 'input_income' not in st.session_state:
        st.session_state.input_income = 85
    if 'input_spending' not in st.session_state:
        st.session_state.input_spending = 80
    if 'input_age' not in st.session_state:
        st.session_state.input_age = 32
    if 'input_gender' not in st.session_state:
        st.session_state.input_gender = "Female"

    # Botones de presets rápidos para testear cada cluster
    st.write("**⚡ Pruebas Rápidas de Arquetipos:**")
    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    
    with p_col1:
        if st.button("👑 Probar VIP", use_container_width=True):
            st.session_state.input_income = 95
            st.session_state.input_spending = 88
            st.rerun()
    with p_col2:
        if st.button("💰 Ahorrador Cauteloso", use_container_width=True):
            st.session_state.input_income = 90
            st.session_state.input_spending = 18
            st.rerun()
    with p_col3:
        if st.button("⚡ Impulsivo Joven", use_container_width=True):
            st.session_state.input_income = 25
            st.session_state.input_spending = 85
            st.rerun()
    with p_col4:
        if st.button("⚖️ Cliente Promedio", use_container_width=True):
            st.session_state.input_income = 55
            st.session_state.input_spending = 50
            st.rerun()
    with p_col5:
        if st.button("🏷️ Bajo Valor", use_container_width=True):
            st.session_state.input_income = 22
            st.session_state.input_spending = 15
            st.rerun()

    st.write("")
    
    col_input1, col_input2 = st.columns([1, 1.25])
    
    with col_input1:
        st.subheader("📝 Parámetros del Cliente")
        
        # Sliders reactivos en tiempo real (sin form que bloquee)
        in_income = st.slider(
            "💵 Ingreso Anual (miles de USD, k$):", 
            min_value=10, 
            max_value=145, 
            value=st.session_state.input_income, 
            step=1,
            key="slider_income",
            help="Modifica este valor y observa la predicción actualizarse al instante."
        )
        
        in_spending = st.slider(
            "🛍️ Puntaje de Gasto (Escala 1 a 100):", 
            min_value=1, 
            max_value=100, 
            value=st.session_state.input_spending, 
            step=1,
            key="slider_spending",
            help="Modifica este valor y observa la predicción actualizarse al instante."
        )
        
        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            in_age = st.number_input("🎂 Edad (años):", min_value=18, max_value=85, value=st.session_state.input_age, step=1)
        with col_sub2:
            in_gender = st.selectbox("🚻 Género:", ["Female", "Male"], index=0 if st.session_state.input_gender == "Female" else 1)

    # Cálculo inmediato de la predicción K-Means
    new_point = np.array([[in_income, in_spending]])
    new_point_scaled = scaler.transform(new_point)
    pred_cluster = int(kmeans.predict(new_point_scaled)[0])
    pred_info = profile_dict[pred_cluster]
    
    with col_input2:
        st.subheader("💡 Resultado de la Clasificación Automática")
        st.markdown(f"""
        <div class="profile-card">
            <span style='background-color:#E2E8F0; color:#1A202C; padding: 4px 10px; border-radius: 6px; font-weight:bold; font-size:0.85rem;'>
                PREDICCIÓN ACTIVA
            </span>
            <h2 style='color: #F6E05E; margin-top:8px; margin-bottom:4px;'>Cluster Asignado: Cluster {pred_cluster}</h2>
            <h3 style='margin-top:0; font-size:1.3rem;'>{pred_info['nombre']}</h3>
            <p style='font-size: 1.05rem;'><b>Arquetipo:</b> {pred_info['categoria']}</p>
            <p style='font-size: 0.95rem; opacity: 0.95;'><b>Comportamiento:</b> {pred_info['descripcion']}</p>
            <hr style='border: 0.5px solid rgba(255,255,255,0.3);'>
            <h4 style='color: #90CDF4; margin-bottom:5px;'>🎯 Estrategia Táctica de Marketing:</h4>
            <p style='font-size: 0.98rem; font-weight: 500;'>{pred_info['estrategia']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Métricas de distancia respecto al centroide
        st.write("")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            diff_inc = in_income - pred_info['inc_center']
            st.metric("Ingreso Ingresado", f"${in_income}k", f"{diff_inc:+.1f}k del centro")
        with col_m2:
            diff_sp = in_spending - pred_info['sp_center']
            st.metric("Gasto Ingresado", f"{in_spending} pts", f"{diff_sp:+.1f} pts del centro")

    st.divider()
    st.subheader("📍 Proyección en Vivo: Ubicación del Cliente en el Plano de Clusters")
    
    # Gráfico interactivo con el cliente nuevo destacado
    fig_new = px.scatter(
        df,
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        color='Perfil',
        hover_data=['CustomerID', 'Age', 'Gender'],
        title=f"Posición del Cliente (Ingreso=${in_income}k, Gasto={in_spending} pts) ➜ Cluster {pred_cluster}",
        labels={'Annual Income (k$)': 'Ingreso Anual (k$)', 'Spending Score (1-100)': 'Puntaje de Gasto (1-100)'},
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.65
    )
    
    # Añadir centroides
    fig_new.add_trace(go.Scatter(
        x=centroids_orig[:, 0],
        y=centroids_orig[:, 1],
        mode='markers+text',
        name='Centroides',
        text=[f"C{i}" for i in range(k_selected)],
        textposition="top right",
        marker=dict(symbol='x', size=15, color='black', line=dict(width=2, color='white'))
    ))
    
    # Añadir nuevo cliente proyectado
    fig_new.add_trace(go.Scatter(
        x=[in_income],
        y=[in_spending],
        mode='markers+text',
        name='⭐ NUEVO CLIENTE EVALUADO',
        text=[f"📍 CLIENTE ({in_income}k$, {in_spending}pts)"],
        textposition="top center",
        marker=dict(symbol='star', size=26, color='#ECC94B', line=dict(width=2.5, color='#744210'))
    ))
    
    fig_new.update_layout(
        height=540,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_new, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: VISUALIZACIÓN DE CLUSTERS
# -------------------------------------------------------------
with tab2:
    st.header("📊 Exploración y Clusters Generados por K-Means")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Total Clientes", f"{len(df)}")
    with col_stat2:
        st.metric("Clusters Activos", f"{k_selected}")
    with col_stat3:
        st.metric("Ingreso Promedio", f"${df['Annual Income (k$)'].mean():.1f}k")
    with col_stat4:
        st.metric("Gasto Promedio", f"{df['Spending Score (1-100)'].mean():.1f} pts")
        
    st.write("")
    
    fig_clusters = px.scatter(
        df,
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        color='Perfil',
        size='Age',
        hover_name='CustomerID',
        hover_data={'Annual Income (k$)': True, 'Spending Score (1-100)': True, 'Age': True, 'Gender': True, 'Perfil': False},
        title=f"Segmentación de Clientes con K = {k_selected} (Tamaño de burbuja = Edad)",
        labels={'Annual Income (k$)': 'Ingreso Anual (k$)', 'Spending Score (1-100)': 'Puntaje de Gasto (1-100)'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    fig_clusters.add_trace(go.Scatter(
        x=centroids_orig[:, 0],
        y=centroids_orig[:, 1],
        mode='markers+text',
        name='Centroides',
        text=[f"C{i}" for i in range(k_selected)],
        textposition="bottom center",
        marker=dict(symbol='star-diamond', size=16, color='yellow', line=dict(width=2, color='black'))
    ))
    
    fig_clusters.update_layout(
        height=580,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_clusters, use_container_width=True)
    
    st.subheader("📋 Resumen Estadístico por Cluster y Centroides Físicos")
    resumen_tbl = df.groupby(['Cluster', 'Perfil', 'Categoria']).agg(
        Total_Clientes=('CustomerID', 'count'),
        Edad_Media=('Age', 'mean'),
        Ingreso_Medio=('Annual Income (k$)', 'mean'),
        Gasto_Medio=('Spending Score (1-100)', 'mean'),
        Porcentaje_Mujeres=('Gender', lambda g: (g == 'Female').mean() * 100)
    ).round(2).reset_index()
    resumen_tbl['% Participación'] = (resumen_tbl['Total_Clientes'] / len(df) * 100).round(2)
    
    st.dataframe(resumen_tbl, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: MÉTODO DEL CODO Y COEFICIENTE DE SILUETA
# -------------------------------------------------------------
with tab3:
    st.header("📈 Evaluación y Selección del K Óptimo")
    st.write("Análisis cuantitativo de la inercia (WCSS) y la calidad de silueta para la justificación matemática de **$K = 5$**:")
    
    K_range, wcss, sil_k, sil_scores = compute_evaluation_curves(X_scaled)
    
    col_codo, col_sil = st.columns(2)
    
    with col_codo:
        st.subheader("1. Método del Codo (Inercia WCSS)")
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=K_range,
            y=wcss,
            mode='lines+markers',
            name='WCSS',
            line=dict(color='#2B6CB0', width=3),
            marker=dict(size=9, color='#1A365D')
        ))
        fig_elbow.add_vline(x=5, line_width=2, line_dash="dash", line_color="#E53E3E", annotation_text="Codo Óptimo (K=5)")
        fig_elbow.update_layout(
            title="Curva de Inercia Intra-Cluster (Elbow Curve)",
            xaxis_title="Número de Clusters (K)",
            yaxis_title="Inercia (WCSS)",
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            height=420
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.info("💡 **Interpretación del Codo:** La curva muestra caídas pronunciadas de K=1 a K=5. A partir de K=5, la disminución en la suma de distancias al cuadrado se estabiliza, evidenciando el punto de máxima eficiencia.")
        
    with col_sil:
        st.subheader("2. Coeficiente de Silueta Promedio")
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=sil_k,
            y=sil_scores,
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='#2E8540', width=3),
            marker=dict(size=9, color='#22543D')
        ))
        fig_sil.add_vline(x=5, line_width=2, line_dash="dash", line_color="#E53E3E", annotation_text="Máximo Global (0.5547)")
        fig_sil.update_layout(
            title="Coeficiente de Silueta vs K",
            xaxis_title="Número de Clusters (K)",
            yaxis_title="Silhouette Score Promedio",
            xaxis=dict(tickmode='linear', tick0=2, dtick=1),
            height=420
        )
        st.plotly_chart(fig_sil, use_container_width=True)
        st.success("✅ **Concordancia Exacta:** El Coeficiente de Silueta alcanza su valor máximo absoluto precisamente en **K = 5 (0.5547)**, validando matemáticamente la cohesión y separación óptima de los grupos.")

# -------------------------------------------------------------
# TAB 4: PERFILES Y ESTRATEGIAS DE NEGOCIO
# -------------------------------------------------------------
with tab4:
    st.header("💼 Perfiles de Negocio y Estrategias Tácticas de Marketing")
    st.write("Caracterización de los 4 grandes arquetipos comerciales identificados en la población de Mall Customers:")
    
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.markdown("""
        ### 👑 1. Clientes de Alto Valor (VIP / Target Premium)
        - **Perfil:** Ingreso Anual Alto (~$86.5k), Puntaje de Gasto Alto (~82.1 pts).
        - **Objetivo Estratégico:** Fidelización, retención y maximización del valor del ciclo de vida (CLV).
        - **Acciones Tácticas:**
          - Invitaciones a preventas y lanzamientos privados de colecciones de lujo.
          - Servicio de *Personal Shopper* y estacionamiento preferencial reservado.
          - Acceso a salones lounge ejecutivos con atenciones de alta gama.
        
        ---
        
        ### 🎯 2. Clientes con Potencial - Ahorradores Cautelosos (Mayor Oportunidad de ROI)
        - **Perfil:** Ingreso Anual Muy Alto (~$88.2k), Puntaje de Gasto Bajo (~17.1 pts).
        - **Diagnóstico:** Tienen el mayor ingreso del mall pero la oferta actual no los seduce.
        - **Acciones Tácticas:**
          - Campañas enfocadas en calidad, durabilidad, exclusividad y tecnología.
          - Experiencias gastronómicas gourmet y demostraciones interactivas de alta gama.
          - Programas de *cashback* financiero en lugar de descuentos directos.
        """)
        
    with col_p2:
        st.markdown("""
        ### ⚡ 3. Clientes con Potencial - Consumidores Impulsivos
        - **Perfil:** Ingreso Anual Bajo (~$25.7k), Puntaje de Gasto Alto (~79.4 pts).
        - **Objetivo Estratégico:** Engagement digital y financiamiento responsable.
        - **Acciones Tácticas:**
          - Esquemas de pago flexible tipo *Buy Now, Pay Later* (BNPL) sin intereses.
          - Campañas en Instagram/TikTok para lanzamientos de tendencias de moda joven.
          - Ventas *flash* y eventos de fin de semana con dinámicas interactivas.
        
        ---
        
        ### ⚖️ 4. Clientes Promedio (Columna Vertebral de Tráfico)
        - **Perfil:** Ingreso Anual Medio (~$55.3k), Puntaje de Gasto Medio (~49.5 pts). Representan más del 40% de los visitantes.
        - **Acciones Tácticas:**
          - Paquetes familiares integrados (compras + cine + restaurantes).
          - Programas de acumulación de puntos canjeables por artículos cotidianos.
          - Campañas estacionales masivas (Black Friday, Navidad, Regreso a Clases).
        """)

st.sidebar.divider()
st.sidebar.caption("Taller Final Minería de Datos | Segmentación K-Means | Streamlit v" + st.__version__)
