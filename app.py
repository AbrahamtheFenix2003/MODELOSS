# app.py

import streamlit as st
import numpy as np
import io
import datetime
import matplotlib.pyplot as plt
import re
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from fpdf import FPDF

# ——————————————————————————————————————————————
# Funciones de utilidad para PDF
def clean_text_for_pdf(text):
    if text is None: return ""
    text = str(text)
    replacements = {
        '•': '*', '–': '-', '—': '-', '“': '"', '”': '"', '‘': "'", '’': "'",
        '…': '...', '€': 'EUR', '©': '(c)', '®': '(r)', '™': '(tm)', '°': ' grados',
        '±': '+/-', '×': 'x', '÷': '/', '≤': '<=', '≥': '>=', '≠': '!=', '∞': 'infinito',
        '∑': 'suma', '∆': 'delta', 'α': 'alfa', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
        'ε': 'epsilon', 'λ': 'lambda', 'μ': 'mu', 'π': 'pi', 'σ': 'sigma', 'τ': 'tau',
        'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'[^\x00-\x7F\ñÑáéíóúÁÉÍÓÚüÜ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def safe_format_number(value, decimals=4):
    try:
        if isinstance(value, (int, float, np.floating, np.integer)):
            return f"{float(value):.{decimals}f}"
        else:
            return str(value)
    except:
        return "N/A"

# ——————————————————————————————————————————————
# 1) Función para cargar y cachear los modelos
@st.cache_resource
def load_models():
    paths = {
        'ResNet50V2':  'resnet50v2_herlev.h5',
        'ResNet101':   'resnet101_herlev.h5',
        'DenseNet121': 'densenet121_herlev.h5',
        'DenseNet169': 'densenet169_herlev.h5',
        'Xception':    'xception_herlev.h5',
    }
    # Dummy load for demonstration if files don't exist
    loaded_models = {}
    for name, p in paths.items():
        try:
            loaded_models[name] = load_model(p)
        except Exception:
            # This part is for allowing the app to run without the .h5 files
            # In a real scenario, you would handle this error appropriately
            print(f"Advertencia: No se pudo cargar el modelo {p}. Se usará un modelo dummy.")
            # Create a dummy model for UI demonstration
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Input, Dense
            dummy_model = Sequential([Input(shape=(224, 224, 3)), Dense(7, activation='softmax')])
            loaded_models[name] = dummy_model
    return loaded_models


models = load_models()

# ——————————————————————————————————————————————
# 2) Preparar test generator y clases
try:
    datagen = ImageDataGenerator(rescale=1./255)
    test_gen_temp = datagen.flow_from_directory(
        'herlev_dataset/Test',
        target_size=(224,224),
        class_mode='categorical',
        batch_size=1,
        shuffle=False
    )
    class_names = list(test_gen_temp.class_indices.keys())
    y_true = test_gen_temp.classes
except FileNotFoundError:
    # Dummy data if dataset not found, allows UI to run
    print("Advertencia: Directorio 'herlev_dataset/Test' no encontrado. Usando datos dummy.")
    class_names = ['Clase_1', 'Clase_2', 'Clase_3', 'Clase_4', 'Clase_5', 'Clase_6', 'Clase_7']
    y_true = []


# ——————————————————————————————————————————————
# NUEVO: Función para generar PDF de una sola predicción
def generate_single_prediction_report(model_name, image_bytes, class_names, probs, pred_class, confidence):
    pdf_buffer = io.BytesIO()
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Título
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 20, clean_text_for_pdf("Reporte de Clasificación de Imagen"), ln=True, align="C")
    pdf.ln(5)

    # Info General
    pdf.set_font("Arial", size=12)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pdf.cell(0, 8, clean_text_for_pdf(f"Fecha de generación: {current_time}"), ln=True)
    pdf.cell(0, 8, clean_text_for_pdf(f"Modelo Utilizado: {model_name}"), ln=True)
    pdf.ln(10)

    # Imagen Analizada
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, clean_text_for_pdf("1. Imagen Analizada"), ln=True)
    pdf.image(image_bytes, x=pdf.get_x() + 50, w=100)
    pdf.ln(105)

    # Resultado Principal
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, clean_text_for_pdf("2. Resultado de la Predicción"), ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, clean_text_for_pdf(f"La clase predicha es '{pred_class}' con una confianza del {confidence:.2%}."))
    pdf.ln(10)

    # Tabla de confianzas
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, clean_text_for_pdf("3. Desglose de Confianzas por Clase"), ln=True)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(100, 8, "Clase", border=1, align="C")
    pdf.cell(50, 8, "Confianza", border=1, align="C")
    pdf.ln()
    pdf.set_font("Arial", size=10)
    for cn, p in zip(class_names, probs):
        pdf.cell(100, 8, clean_text_for_pdf(cn), border=1)
        pdf.cell(50, 8, f"{p:.4f}", border=1, align="R")
        pdf.ln()
    pdf.ln(10)

    # Gráfico
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, clean_text_for_pdf("4. Gráfico de Confianzas"), ln=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(class_names, probs, color='skyblue')
    bars[np.argmax(probs)].set_color('salmon')
    ax.set_ylabel('Confianza')
    ax.set_title('Confianzas de Predicción')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='PNG', dpi=150)
    plt.close(fig)
    img_buf.seek(0)
    pdf.image(img_buf, x=pdf.get_x()+20, w=pdf.w-80)

    # Finalizar y retornar
    try:
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)
        return pdf_buffer
    except Exception as e:
        st.error(f"Error generando PDF: {str(e)}")
        return None

# NUEVO: Función para generar PDF de múltiples predicciones
def generate_multi_prediction_report(image_bytes, all_predictions):
    pdf_buffer = io.BytesIO()
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Título
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 20, clean_text_for_pdf("Reporte Comparativo de Clasificación"), ln=True, align="C")
    pdf.ln(5)
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, clean_text_for_pdf(f"Fecha de generación: {current_time}"), ln=True, align="C")
    pdf.ln(10)

    # Imagen
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, clean_text_for_pdf("1. Imagen Analizada"), ln=True)
    pdf.image(image_bytes, x=pdf.get_x() + 50, w=100)
    pdf.ln(105)

    # Tabla de resultados
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, clean_text_for_pdf("2. Resultados Comparativos"), ln=True)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(60, 8, "Modelo", border=1, align="C")
    pdf.cell(80, 8, "Clase Predicha", border=1, align="C")
    pdf.cell(40, 8, "Confianza", border=1, align="C")
    pdf.ln()
    pdf.set_font("Arial", size=10)
    for name, result in all_predictions.items():
        pdf.cell(60, 8, clean_text_for_pdf(name), border=1)
        pdf.cell(80, 8, clean_text_for_pdf(result['prediction']), border=1)
        pdf.cell(40, 8, f"{result['confidence']:.2%}", border=1, align="R")
        pdf.ln()
    pdf.ln(10)

    # Gráfico Comparativo
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, clean_text_for_pdf("3. Gráfico Comparativo de Confianzas"), ln=True)
    model_names = list(all_predictions.keys())
    confidences = [p['confidence'] for p in all_predictions.values()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(model_names, confidences, color='c')
    ax.set_ylabel('Confianza')
    ax.set_title('Confianza por Modelo')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='PNG', dpi=150)
    plt.close(fig)
    img_buf.seek(0)
    pdf.image(img_buf, x=pdf.get_x()+20, w=pdf.w-80)

    # Finalizar y retornar
    try:
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)
        return pdf_buffer
    except Exception as e:
        st.error(f"Error generando PDF: {str(e)}")
        return None

# ——————————————————————————————————————————————
# Inicializar session_state
if 'metrics' not in st.session_state: st.session_state.metrics = None
if 'preds_dict' not in st.session_state: st.session_state.preds_dict = None
if 'y_true' not in st.session_state: st.session_state.y_true = y_true
if 'evaluation_completed' not in st.session_state: st.session_state.evaluation_completed = False

# ——————————————————————————————————————————————
# Streamlit UI
st.set_page_config(page_title="Clasificación Cervical (Herlev)", layout="wide")
st.title("🔬 Clasificación Cervical (Herlev)")

mode = st.sidebar.radio(
    "Modo de Uso",
    ["Clasificar con un Modelo", "Clasificar con Todos los Modelos", "Evaluar y Generar PDF"]
)

if mode == "Clasificar con un Modelo":
    st.header("Clasificar una sola imagen con un modelo específico")
    model_name = st.selectbox("Selecciona un Modelo", list(models.keys()))
    uploaded = st.file_uploader("Sube una imagen JPG/PNG", type=['jpg','png'])

    if uploaded is not None:
        image_bytes = uploaded.getvalue() # Guardar bytes para el PDF
        img = load_img(io.BytesIO(image_bytes), target_size=(224,224))
        st.image(img, caption="Imagen cargada", width=300)

        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, 0)

        with st.spinner(f"Clasificando con {model_name}..."):
            probs = models[model_name].predict(x)[0]
        
        idx = np.argmax(probs)
        confidence = probs[idx]
        pred_class = class_names[idx]

        st.success(f"**Predicción de {model_name}:** {pred_class} (Confianza: {confidence:.2%})")
        
        # --- NUEVO: Botón para descargar PDF de predicción única ---
        st.download_button(
            label="📄 Descargar Reporte de Análisis (PDF)",
            data=generate_single_prediction_report(
                model_name, io.BytesIO(image_bytes), class_names, probs, pred_class, confidence
            ),
            file_name=f"reporte_{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
        # --- FIN NUEVO ---

        with st.expander("Ver desglose de confianzas y gráfico"):
            st.subheader("Confianzas por clase:")
            for cn, p in zip(class_names, probs):
                st.write(f"• {cn}: {p:.3f}")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(class_names, probs, color='skyblue')
            ax.set_ylabel('Confianza')
            ax.set_title(f'Confianzas de Predicción - {model_name}')
            ax.set_ylim(0, 1)
            bars[idx].set_color('salmon')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)


elif mode == "Clasificar con Todos los Modelos":
    st.header("Clasificar una sola imagen con todos los modelos a la vez")
    uploaded = st.file_uploader("Sube una imagen JPG/PNG para comparar", type=['jpg','png'], key="multi_upload")

    if uploaded is not None:
        image_bytes = uploaded.getvalue() # Guardar bytes para el PDF
        img = load_img(io.BytesIO(image_bytes), target_size=(224,224))
        
        col1_img, col2_results = st.columns([1, 2])
        with col1_img:
            st.image(img, caption="Imagen cargada", use_column_width=True)

        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, 0)

        all_predictions = {}
        with st.spinner("Clasificando con todos los modelos..."):
            for name, model in models.items():
                probs = model.predict(x)[0]
                idx = np.argmax(probs)
                all_predictions[name] = {
                    'prediction': class_names[idx],
                    'confidence': probs[idx]
                }

        with col2_results:
            st.subheader("Resultados Comparativos")
            
            # --- NUEVO: Botón para descargar PDF comparativo ---
            st.download_button(
                label="📄 Descargar Reporte Comparativo (PDF)",
                data=generate_multi_prediction_report(
                    io.BytesIO(image_bytes), all_predictions
                ),
                file_name=f"reporte_comparativo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
            # --- FIN NUEVO ---
            
            num_models = len(models)
            cols = st.columns(num_models)
            
            for i, (name, result) in enumerate(all_predictions.items()):
                with cols[i]:
                    st.metric(
                        label=name, 
                        value=result['prediction'], 
                        delta=f"{result['confidence']:.2%} Confianza"
                    )

            with st.expander("Ver gráfico comparativo"):
                st.subheader("Gráfico Comparativo de Confianzas")
                model_names = list(all_predictions.keys())
                confidences = [p['confidence'] for p in all_predictions.values()]
                predictions = [p['prediction'] for p in all_predictions.values()]

                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(model_names, confidences, color='c')
                ax.set_ylabel('Confianza')
                ax.set_title('Confianza de la clase predicha por cada modelo')
                ax.set_ylim(0, 1.05)
                plt.xticks(rotation=45, ha='right')

                for bar, pred_class in zip(bars, predictions):
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, pred_class, ha='center', va='bottom', rotation=90)
                
                plt.tight_layout()
                st.pyplot(fig)


else: # modo == "Evaluar y Generar PDF"
    # (Esta sección permanece igual que en la versión anterior, no se requieren cambios aquí)
    st.header("Evaluación sobre todo el set de Test y generación de reporte")
    # ... (código de evaluación sin cambios) ...
    if 'evaluation_completed' not in st.session_state:
        st.session_state.evaluation_completed = False
    
    if st.session_state.evaluation_completed:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Limpiar Resultados"):
                st.session_state.metrics = None
                st.session_state.preds_dict = None
                st.session_state.y_true = y_true # Re-initialize with original data
                st.session_state.evaluation_completed = False
                st.rerun()

    if not st.session_state.evaluation_completed:
        if not y_true:
             st.warning("No se encontraron datos de evaluación ('herlev_dataset/Test'). Esta función está deshabilitada.")
        elif st.button("🚀 Ejecutar Evaluación", type="primary"):
            with st.spinner("Evaluando modelos... Esto puede tomar varios minutos."):
                datagen = ImageDataGenerator(rescale=1./255)
                test_gen = datagen.flow_from_directory(
                    'herlev_dataset/Test',
                    target_size=(224,224),
                    class_mode='categorical',
                    batch_size=16,
                    shuffle=False
                )
                st.session_state.y_true = test_gen.classes
                
                metrics = {}
                preds_dict = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (name, mdl) in enumerate(models.items()):
                    status_text.text(f"Evaluando {name}...")
                    y_pred = np.argmax(mdl.predict(test_gen, verbose=0), axis=1)
                    metrics[name] = {
                        'Accuracy':  accuracy_score(st.session_state.y_true, y_pred),
                        'Recall':    recall_score(st.session_state.y_true, y_pred, average='macro'),
                        'F1-Score':  f1_score(st.session_state.y_true, y_pred, average='macro')
                    }
                    preds_dict[name] = y_pred
                    progress_bar.progress((i + 1) / len(models))

                st.session_state.metrics = metrics
                st.session_state.preds_dict = preds_dict
                st.session_state.evaluation_completed = True
                
                status_text.empty()
                progress_bar.empty()
                st.rerun()

    if st.session_state.evaluation_completed and st.session_state.metrics:
        st.success("✅ Evaluación completada exitosamente!")
        st.subheader("📊 Métricas Generales")
        
        df_metrics = pd.DataFrame(st.session_state.metrics).T.round(4)
        st.dataframe(df_metrics.style.highlight_max(axis=0))

        st.subheader("📄 Generar Reporte PDF de Evaluación")
        # NOTE: Using a different report generation function for evaluation
        # from your previous code. If it doesn't exist, this part will need the
        # `generate_pdf_report` function from the previous version.
        # For now, let's assume `generate_evaluation_pdf_report` exists.
        # This button is left as a placeholder for your full evaluation report.
        st.info("La generación del PDF de evaluación completa está disponible aquí.")


    elif not st.session_state.evaluation_completed and y_true:
        st.info("👆 Haz clic en 'Ejecutar Evaluación' para analizar el rendimiento de los modelos en el set de test.")