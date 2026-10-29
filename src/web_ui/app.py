import os
import sys
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processecing.data_config import CLASS_MAP, IMAGE_SIZE
from src.data_processecing.img_feed_pipeline import image_feed
from src.models.single_object_detection.sod_predict import sod_predict

# Inverted mapping: id -> class name
ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}


def format_class_name(raw_name: str) -> str:
    """Format raw 'category:breed_name' into user-friendly 'Breed Name (Category)'."""
    if ":" in raw_name:
        category, species = raw_name.split(":", 1)
        species_formatted = species.replace("_", " ").title()
        return f"{species_formatted} ({category.capitalize()})"
    return raw_name.replace("_", " ").title()


def draw_bounding_box(
    image: Image.Image,
    bbox_norm: list,
    label: str,
    confidence: float,
    box_color: str = "#00FFAA",
    thickness: int = 3,
) -> Image.Image:
    """
    Draw normalized bounding box and label tag on PIL Image.
    bbox_norm: [xmin, ymin, xmax, ymax] with values in [0, 1]
    """
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    xmin_norm, ymin_norm, xmax_norm, ymax_norm = bbox_norm
    # Convert normalized coordinates to absolute pixels
    x1 = int(np.clip(xmin_norm * width, 0, width))
    y1 = int(np.clip(ymin_norm * height, 0, height))
    x2 = int(np.clip(xmax_norm * width, 0, width))
    y2 = int(np.clip(ymax_norm * height, 0, height))

    # Ensure valid order
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    # Draw multi-line border for thickness
    for i in range(thickness):
        draw.rectangle(
            [x_min - i, y_min - i, x_max + i, y_max + i],
            outline=box_color,
        )

    # Label text banner
    label_text = f"{label} | {confidence:.1%}"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if font and hasattr(draw, "textbbox"):
        text_bbox = draw.textbbox((x_min, y_min), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
    else:
        text_w = len(label_text) * 7
        text_h = 14

    banner_top = max(0, y_min - text_h - 6)
    banner_bottom = y_min
    banner_right = min(width, x_min + text_w + 12)

    draw.rectangle([x_min, banner_top, banner_right, banner_bottom], fill=box_color)
    draw.text((x_min + 6, banner_top + 2), label_text, fill="#000000", font=font)

    return annotated


def get_sample_images():
    """Retrieve up to 10 sample images from data/voc-animal if available."""
    sample_dir = PROJECT_ROOT / "data" / "voc-animal"
    samples = {}
    if sample_dir.exists():
        jpg_files = sorted(list(sample_dir.glob("*.jpg")))[:15]
        for f in jpg_files:
            samples[f.name] = str(f)
    return samples


def main():
    st.set_page_config(
        page_title="Single Object Detection (SOD)",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            color: #64748B;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-header">🎯 Single Object Detection Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Feed an image through <code>image_feed</code>, run <code>sod_predict</code>, and view the predicted class, confidence, and bounding box coordinates.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.header("⚙️ Configuration & Input")
    input_source = st.sidebar.radio("Choose Image Source:", ["Upload an Image", "Use Sample Dataset Image"])

    selected_image = None
    image_name = ""

    samples = get_sample_images()

    if input_source == "Upload an Image":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Image (PNG, JPG, JPEG):",
            type=["png", "jpg", "jpeg", "webp"],
            help="Select an image file to run detection on.",
        )
        if uploaded_file is not None:
            try:
                selected_image = Image.open(uploaded_file).convert("RGB")
                image_name = uploaded_file.name
            except Exception as e:
                st.sidebar.error(f"Error opening image: {e}")
    else:
        if samples:
            sample_choice = st.sidebar.selectbox("Select Sample Image:", list(samples.keys()))
            if sample_choice:
                selected_image = Image.open(samples[sample_choice]).convert("RGB")
                image_name = sample_choice
        else:
            st.sidebar.info("No sample images found in `data/voc-animal`.")

    # UI Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Visualization Options")
    box_color = st.sidebar.color_picker("Bounding Box Color:", "#00FFAA")
    thickness = st.sidebar.slider("Line Thickness (px):", min_value=1, max_value=8, value=3)

    st.sidebar.markdown("---")
    st.sidebar.caption("Model: VGG16 Backbone | 256x256 Input | 37 Oxford-IIIT Pet Classes")

    # Main Area
    if selected_image is None:
        st.info("👈 Please upload an image or select a sample image from the sidebar to begin.")
        return

    # Two Column Layout
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.subheader("🖼️ Detection Visualization")

    with col_right:
        st.subheader("📊 Prediction Results")

    # Pipeline Execution
    with st.spinner("Processing image and predicting bounding box..."):
        # 1. Process image via image_feed
        processed_img = image_feed(selected_image)

        if processed_img is None:
            st.error("❌ Failed to process image with `image_feed`.")
            return

        # 2. Run model prediction via sod_predict
        preds = sod_predict(processed_img)

        # sod_model outputs [class_output, bbox_output]
        if isinstance(preds, (list, tuple)) and len(preds) == 2:
            class_probs = preds[0][0]  # shape (37,)
            bbox = preds[1][0]         # shape (4,) -> [xmin, ymin, xmax, ymax]
        else:
            st.error(f"Unexpected prediction output format: {type(preds)}")
            return

        # 3. Decode prediction
        pred_class_id = int(np.argmax(class_probs))
        confidence = float(class_probs[pred_class_id])
        raw_class_name = ID_TO_CLASS.get(pred_class_id, f"Class_{pred_class_id}")
        formatted_class = format_class_name(raw_class_name)

        # Bounding box values
        xmin_norm = float(bbox[0])
        ymin_norm = float(bbox[1])
        xmax_norm = float(bbox[2])
        ymax_norm = float(bbox[3])

        orig_w, orig_h = selected_image.size
        x1_px = int(np.clip(xmin_norm * orig_w, 0, orig_w))
        y1_px = int(np.clip(ymin_norm * orig_h, 0, orig_h))
        x2_px = int(np.clip(xmax_norm * orig_w, 0, orig_w))
        y2_px = int(np.clip(ymax_norm * orig_h, 0, orig_h))

        # 4. Render annotated image
        annotated_img = draw_bounding_box(
            selected_image,
            [xmin_norm, ymin_norm, xmax_norm, ymax_norm],
            formatted_class,
            confidence,
            box_color=box_color,
            thickness=thickness,
        )

    # Display in Left Column
    with col_left:
        tab_detection, tab_original, tab_processed = st.tabs(
            ["🎯 Detection Overlay", "📷 Original Image", "📐 Model Input (256x256)"]
        )

        with tab_detection:
            st.image(
                annotated_img,
                caption=f"Detected: {formatted_class} ({confidence:.1%})",
                use_container_width=True,
            )

        with tab_original:
            st.image(
                selected_image,
                caption=f"Source: {image_name} ({orig_w}x{orig_h})",
                use_container_width=True,
            )

        with tab_processed:
            # Display normalized 256x256 image
            preview_arr = (processed_img[0] * 255.0).clip(0, 255).astype(np.uint8)
            st.image(
                preview_arr,
                caption=f"Shape: {processed_img.shape} | Normalized [0, 1]",
                use_container_width=True,
            )

    # Display in Right Column
    with col_right:
        # KPI Metrics
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Predicted Class", formatted_class)
        with m2:
            st.metric("Confidence Score", f"{confidence:.2%}")

        st.markdown("#### 📦 Bounding Box Coordinates")

        # Pixel Coordinates Table
        st.markdown("**Pixel Coordinates (Original Image)**")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("X min", f"{x1_px} px")
        b2.metric("Y min", f"{y1_px} px")
        b3.metric("X max", f"{x2_px} px")
        b4.metric("Y max", f"{y2_px} px")

        # Normalized Coordinates
        st.markdown("**Normalized Coordinates [0.0 - 1.0]**")
        nb1, nb2, nb3, nb4 = st.columns(4)
        nb1.metric("xmin", f"{xmin_norm:.4f}")
        nb2.metric("ymin", f"{ymin_norm:.4f}")
        nb3.metric("xmax", f"{xmax_norm:.4f}")
        nb4.metric("ymax", f"{ymax_norm:.4f}")

        # Top 5 Predictions
        st.markdown("---")
        st.markdown("#### 🏆 Top 5 Class Probabilities")
        top5_indices = np.argsort(class_probs)[::-1][:5]
        
        top5_data = []
        for idx in top5_indices:
            cls_name = format_class_name(ID_TO_CLASS.get(int(idx), f"Class_{idx}"))
            prob = float(class_probs[idx])
            top5_data.append({"Class": cls_name, "Probability": prob})

        for item in top5_data:
            c_label, c_bar = st.columns([1.5, 2])
            c_label.write(item["Class"])
            c_bar.progress(item["Probability"], text=f"{item['Probability']:.1%}")

        # Raw Data Details Expander
        with st.expander("🔍 View Raw Prediction Tensors"):
            st.json(
                {
                    "class_index": pred_class_id,
                    "raw_class_name": raw_class_name,
                    "confidence_score": float(confidence),
                    "normalized_bbox": [xmin_norm, ymin_norm, xmax_norm, ymax_norm],
                    "pixel_bbox": [x1_px, y1_px, x2_px, y2_px],
                    "image_dimensions": {"width": orig_w, "height": orig_h},
                    "model_input_shape": list(processed_img.shape),
                }
            )


if __name__ == "__main__":
    main()