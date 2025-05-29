# EMOTION-DETECTOR

Este proyecto permite realizar detección de emociones faciales en tiempo real usando tres modelos distintos:

1. **Modelo CNN casero**  
2. **Modelo con DeepFace (afinado)**  
3. **Modelo con MTCNN + CNN ajustado**

Incluye comparativas y scripts de entrenamiento para probar el rendimiento de cada uno.

---

## 📁 Estructura del proyecto

```plaintext
EMOTION-DETECTOR/
├── comparativa/              # Scripts y resultados de comparación entre modelos
│   ├── comparativa.txt
│   └── compare.py
├── demos/                    # Scripts de ejecución de los modelos
│   ├── best_tuned_model.h5
│   ├── CNN_casera.h5
│   ├── cnn_emotion_detection.py
│   ├── df_emotion_detection.py
│   ├── df_tuned_emotion_detection.py
│   └── emotiondetector.json
├── entrenamiento/            # Entrenamiento y fine-tuning
│   ├── deepface_tuning.py
│   └── Untitled.ipynb
├── venv/                     # Entorno virtual (opcional)
├── .gitignore
├── DRequirements.txt         # Requisitos para modelos DeepFace
├── LICENSE
└── README.md
```

---

## ⚙️ Requisitos

- **Python 3.10**

Se recomienda crear **dos entornos virtuales separados**:

- Uno para los modelos basados en **DeepFace** los cuales son los scripts python dentro de la carpeta demos que inician con el prefijo df_
- Otro para la **CNN casera** la cual es un script python dentro de la carpeta demos que inicia con el prefijo cnn_

---

## 🧪 Instalación

### 1. Crear entorno virtual

```bash
py -3.10 -m venv <nombre_del_entorno>
```

### 2. Activar entorno virtual

**En Windows:**

```bash
<nombre_del_entorno>\Scripts\activate
```

**En Mac/Linux:**

```bash
source <nombre_del_entorno>/bin/activate
```

### 3. Instalar dependencias

**Para modelos basados en DeepFace:**

```bash
pip install -r DFrequirements.txt
```
**Para modelos basados en CNN:**

```bash
pip install -r requirements.txt
```
---

## 🚀 Ejecución

Una vez activado el entorno virtual y ubicado en la carpeta del proyecto:

```bash
cd demos
python <script_a_ejecutar.py>
```

### Ejemplos:

```bash
python cnn_emotion_detection.py
python df_emotion_detection.py
python df_tuned_emotion_detection.py
```

---

## 🤖 Modelos disponibles

### 🧠 Modelo 1: CNN Casera (`cnn_emotion_detection.py`)

- Carga modelo desde `.json` y pesos `.h5`
- Usa **Haar Cascade** para detectar rostros
- Redimensiona imágenes a **48x48** (escala de grises)
- Emociones detectadas:
  - `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

---

### 🧠 Modelo 2: DeepFace (`df_emotion_detection.py`)

- Utiliza la librería **DeepFace**
- Detector facial: `yolov8`
- Predicciones automáticas sin entrenamiento adicional
- Resultados visuales en tiempo real usando `cv2`

---

### 🧠 Modelo 3: CNN Ajustada con MTCNN (`df_tuned_emotion_detection.py`)

- Detector facial: **MTCNN**
- CNN entrenada con mejores resultados (`best_tuned_model.h5`)
- Flujo de procesamiento:
  - **Detección → Redimensionamiento → Normalización → Predicción → Visualización**

---

## 🧾 Dependencias clave

Listado de versiones utilizadas (extraído de `DRequirements.txt`):

```makefile
tensorflow==2.19.0
deepface==0.0.93
mtcnn==1.0.0
opencv-python==4.11.0.86
ultralytics==8.3.146
torch==2.7.0
keras==3.10.0
h5py==3.13.0
numpy==2.1.3
matplotlib==3.10.3
pandas==2.2.3
scipy==1.15.3
Flask==3.1.1
```

> ✅ Asegúrate de instalar estas dependencias en un entorno limpio basado en **Python 3.10** para evitar errores de compatibilidad.
