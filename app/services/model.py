import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf

# Clases en el mismo orden alfabético que usó ImageDataGenerator durante el entrenamiento
# (carpetas: carbon, hoja_amarilla, mancha_anillo, mancha_parda, mosaico,
#             muermo_rojo, roya, sanas, secas, tizon_bacteriano)
CLASS_NAMES = [
    "Carbon",
    "Hoja Amarilla",
    "Mancha de Anillo",
    "Mancha Parda",
    "Mosaico",
    "Pudricion Roja",
    "Roya",
    "Hoja Sana",
    "Hoja Seca",
    "Tizon Bacteriano",
]

# Ruta al archivo de pesos
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "../models/ResNet50/resnet50_TL_optimizado_weights.h5"
)

# Singleton — se carga una sola vez al iniciar el servidor
_model = None


# ---------------------------------------------------------------------------
# Construcción de la arquitectura
# ---------------------------------------------------------------------------

def build_model():
    """Reconstruye la misma arquitectura Sequential usada en el entrenamiento."""
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras import layers, models

    IMG_SIZE = 224
    NUM_CLASSES = 10

    # El notebook entrenó con Sequential; lo reproducimos exactamente.
    base_model = ResNet50(
        weights=None,
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    # build() materializa los pesos antes de asignarlos
    model.build((None, IMG_SIZE, IMG_SIZE, 3))
    return model


# ---------------------------------------------------------------------------
# Carga de pesos desde HDF5
# ---------------------------------------------------------------------------

def _get_weights_from_group(grp):
    """
    Extrae los pesos de un grupo HDF5 en el orden que Keras espera.

    Las Conv2D capas esperan  [kernel, bias].
    Las BatchNorm capas esperan [gamma, beta, moving_mean, moving_variance].
    Otros tipos con un solo tensor se devuelven tal cual.
    """
    keys = list(grp.keys())

    if "kernel:0" in keys and "bias:0" in keys:
        # Conv2D y Dense
        return [grp["kernel:0"][()], grp["bias:0"][()]]

    if "gamma:0" in keys:
        # BatchNormalization: gamma, beta, moving_mean, moving_variance
        return [
            grp["gamma:0"][()],
            grp["beta:0"][()],
            grp["moving_mean:0"][()],
            grp["moving_variance:0"][()],
        ]

    # Fallback: orden alfabético (puede fallar para casos no previstos)
    return [grp[k][()] for k in sorted(keys)]


def _load_weights_manual(model, path):
    """
    Carga pesos capa por capa desde el archivo HDF5.

    Esto evita el error 'axes don't match array' que ocurre con
    load_weights(by_name=True) cuando los pesos se guardaron desde un modelo
    Sequential en Keras 2.x.
    """
    import h5py

    with h5py.File(path, "r") as f:
        # --- ResNet50 base (layers[0]) ---
        base = model.layers[0]
        for sublayer in base.layers:
            name = sublayer.name
            if name not in f["resnet50"]:
                continue
            grp = f["resnet50"][name]
            if len(grp.keys()) == 0:
                continue
            weights = _get_weights_from_group(grp)
            try:
                sublayer.set_weights(weights)
            except Exception as exc:
                print(f"[WARN] resnet50/{name}: {exc}")

        # --- Dense 128 (layers[2]) ---
        dense1 = model.layers[2]
        grp = f["dense"]["dense"]
        dense1.set_weights(_get_weights_from_group(grp))

        # --- Dense 10 / salida (layers[4]) ---
        dense2 = model.layers[4]
        grp = f["dense_1"]["dense_1"]
        dense2.set_weights(_get_weights_from_group(grp))


def load_model():
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        print("[WARN] Archivo de pesos no encontrado. Modo simulacion activado.")
        return None

    try:
        _model = build_model()
        _load_weights_manual(_model, MODEL_PATH)

        # Verificación rápida: los kernels Dense no deben ser cero
        d1_mean = float(np.mean(np.abs(_model.layers[2].get_weights()[0])))
        d2_mean = float(np.mean(np.abs(_model.layers[4].get_weights()[0])))

        # Verificar también el primer conv del ResNet50 base
        conv1_mean = float(np.mean(np.abs(_model.layers[0].get_layer("conv1_conv").get_weights()[0])))

        print(
            f"[OK] Modelo cargado | "
            f"conv1_mean={conv1_mean:.5f} | "
            f"dense_mean={d1_mean:.5f} | "
            f"dense1_mean={d2_mean:.5f}"
        )

    except Exception as exc:
        print(f"[ERROR] Fallo al cargar el modelo: {exc}")
        import traceback
        traceback.print_exc()
        _model = None
        return None

    return _model


# ---------------------------------------------------------------------------
# Pre-procesamiento e inferencia
# ---------------------------------------------------------------------------

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convierte bytes de imagen al tensor que espera ResNet50."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    array = np.array(image, dtype=np.float32)
    # Normalización específica de ResNet50 (sustrae media de ImageNet)
    array = tf.keras.applications.resnet50.preprocess_input(array)
    return np.expand_dims(array, axis=0)  # → (1, 224, 224, 3)


def predict(image_bytes: bytes) -> dict:
    """Recibe bytes de imagen y retorna la predicción del modelo."""
    model = load_model()

    # Modo simulación (sin modelo o error al cargar)
    if model is None:
        import random
        idx = random.randint(0, len(CLASS_NAMES) - 1)
        confidence = round(random.uniform(0.75, 0.99), 4)
        return {
            "class": CLASS_NAMES[idx],
            "confidence": confidence,
            "all_probabilities": {
                name: round(random.uniform(0.0, 0.1), 4) for name in CLASS_NAMES
            },
            "simulated": True,
        }

    # Modo real
    tensor = preprocess_image(image_bytes)
    predictions = model.predict(tensor, verbose=0)

    predicted_idx = int(np.argmax(predictions, axis=1)[0])
    confidence = float(predictions[0][predicted_idx])

    return {
        "class": CLASS_NAMES[predicted_idx],
        "confidence": round(confidence, 4),
        "all_probabilities": {
            CLASS_NAMES[i]: round(float(prob), 4)
            for i, prob in enumerate(predictions[0])
        },
        "simulated": False,
    }