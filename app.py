from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf


st.set_page_config(
    page_title="DFU Classifier",
    layout="centered",
)

IMG_SIZE = 224
CLASS_NAMES = {0: "Normal", 1: "Ulcer"}
MODEL_DIR = Path(__file__).resolve().parent / "models"

MODEL_CONFIGS = {
    "MobileNetV2": {
        "file": "best_MobileNetV2.keras",
        "threshold": 0.58,
        "accuracy": 98.11,
        "auc": 99.27,
    },
    "ResNet50": {
        "file": "best_ResNet50.keras",
        "threshold": 0.24,
        "accuracy": 97.48,
        "auc": 99.95,
    },
    "LiteDFU-Net": {
        "file": "best_LiteDFU-Net.keras",
        "threshold": 0.29,
        "accuracy": 96.23,
        "auc": 99.13,
    },
    "EfficientNetB0": {
        "file": "best_EfficientNetB0.keras",
        "threshold": 0.25,
        "accuracy": 99.37,
        "auc": 99.71,
    },
}

PREFERRED_MODEL_ORDER = ["MobileNetV2", "ResNet50", "LiteDFU-Net", "EfficientNetB0"]

MODEL_CUSTOM_OBJECTS = {
    "MobileNetV2": {
        "preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input,
    },
    "ResNet50": {
        "preprocess_input": tf.keras.applications.resnet50.preprocess_input,
    },
    "EfficientNetB0": {
        "preprocess_input": tf.keras.applications.efficientnet.preprocess_input,
    },
    "LiteDFU-Net": {},
}


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def get_model_status() -> tuple[list[str], list[str]]:
    available_models = []
    missing_models = []

    for model_name in PREFERRED_MODEL_ORDER:
        model_path = MODEL_DIR / MODEL_CONFIGS[model_name]["file"]
        if model_path.exists():
            available_models.append(model_name)
        else:
            missing_models.append(model_name)

    return available_models, missing_models


@st.cache_resource(show_spinner=False)
def load_selected_model(model_name: str) -> tf.keras.Model:
    model_path = MODEL_DIR / MODEL_CONFIGS[model_name]["file"]
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file পাওয়া যায়নি: {model_path.name}. "
            f"এই file-টা `{MODEL_DIR}` folder-এ রাখতে হবে।"
        )

    return tf.keras.models.load_model(
        model_path,
        custom_objects=MODEL_CUSTOM_OBJECTS.get(model_name, {}),
        compile=False,
    )


def predict_image(model_name: str, image: Image.Image) -> dict[str, float | int | str]:
    model = load_selected_model(model_name)
    threshold = float(MODEL_CONFIGS[model_name]["threshold"])
    batch = prepare_image(image)

    probability = float(model.predict(batch, verbose=0)[0][0])
    predicted_index = int(probability >= threshold)
    confidence = probability if predicted_index == 1 else 1.0 - probability

    return {
        "probability": probability,
        "threshold": threshold,
        "predicted_index": predicted_index,
        "predicted_label": CLASS_NAMES[predicted_index],
        "confidence": confidence,
    }


st.title("Diabetic Foot Ulcer Detection")
st.caption(
    "Image upload করে dropdown থেকে model select করলে selected model দিয়েই "
    "`Normal` বা `Ulcer` prediction হবে।"
)

available_models, missing_models = get_model_status()

if missing_models:
    st.warning(
        "এই model file-gulo এখনো পাওয়া যায়নি: "
        + ", ".join(f"`{MODEL_CONFIGS[name]['file']}`" for name in missing_models)
    )
else:
    st.success(f"সবগুলো `{len(available_models)}` model file ready আছে।")

if not available_models:
    st.error("`models/` folder-এ usable model file পাওয়া যায়নি।")
    st.stop()

selected_model = st.selectbox(
    "Model Select করো",
    options=available_models,
)
selected_config = MODEL_CONFIGS[selected_model]

col1, col2, col3 = st.columns(3)
col1.metric("Threshold", f"{selected_config['threshold']:.2f}")
col2.metric("Accuracy", f"{selected_config['accuracy']:.2f}%")
col3.metric("AUC", f"{selected_config['auc']:.2f}%")

uploaded_file = st.file_uploader(
    "Foot ulcer image upload করো",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict", type="primary"):
        try:
            with st.spinner(f"{selected_model} দিয়ে prediction হচ্ছে..."):
                result = predict_image(selected_model, image)

            if result["predicted_label"] == "Ulcer":
                st.error(
                    f"Prediction: {result['predicted_label']} "
                    f"(confidence: {format_percentage(result['confidence'])})"
                )
            else:
                st.success(
                    f"Prediction: {result['predicted_label']} "
                    f"(confidence: {format_percentage(result['confidence'])})"
                )

            result_col1, result_col2 = st.columns(2)
            result_col1.metric("Ulcer Probability", f"{result['probability']:.4f}")
            result_col2.metric("Confidence", format_percentage(result["confidence"]))
            st.write(f"Decision threshold: `{result['threshold']:.2f}`")
        except FileNotFoundError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.exception(exc)
else:
    st.caption("Prediction run করতে আগে একটি image upload করো।")

with st.expander("Model file setup"):
    model_file_list = "\n".join(
        f"- `{MODEL_CONFIGS[model_name]['file']}`" for model_name in PREFERRED_MODEL_ORDER
    )
    st.markdown(
        f"""
        `models/` folder-এ এই model files রাখো:

        {model_file_list}

        Kaggle notebook-এ `checkpoint_path = f"/kaggle/working/best_{{model_name}}.keras"`
        use করা হয়েছে, তাই ওই files download করে এখানে রাখলেই app prediction দেবে।

        যদি Kaggle notebook থেকে save করা files অন্য নামে থাকে, তাহলে
        `MODEL_CONFIGS` update করলেই হবে।
        """
    )
