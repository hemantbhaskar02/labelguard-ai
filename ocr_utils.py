import cv2
import numpy as np
import easyocr
from PIL import Image
import logging
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cap the longest side of the image before running OCR. Bumped up from the
# earlier 1200px — small/faded label text (MRP, dates, FSSAI numbers) needs
# more resolution to recognize correctly; 1500px is still fast enough on
# CPU while keeping fine print legible.
MAX_DIMENSION = 1500

# Detections below this confidence are treated as noise (garbled OCR like
# "Somn Grcen Ion") and dropped before the text is handed to the regex
# compliance checker, which otherwise gets confused by junk tokens.
MIN_CONFIDENCE = 0.25


@st.cache_resource(show_spinner="Loading OCR engine (first time only)...")
def get_ocr_reader():
    """
    Create the EasyOCR reader ONCE and cache it for the lifetime of the app.
    Without this, a brand-new reader (and its model files) was being loaded
    on every single scan, which is why scans were taking minutes and
    sometimes crashing the app on Streamlit Cloud's limited free-tier memory.
    """
    return easyocr.Reader(['en'], gpu=False)


def preprocess_image(image_path):
    """
    Preprocess image to improve OCR accuracy.
    Steps: downscale, grayscale, denoise, CLAHE contrast, then a light
    unsharp-mask sharpen — the sharpening pass is what most helps with
    slightly out-of-focus phone photos of product labels.
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")

        # Downscale large images - this drastically speeds up OCR on CPU
        # without hurting accuracy (label text stays readable well below
        # typical phone-camera resolutions).
        h, w = img.shape[:2]
        if max(h, w) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Light unsharp-mask sharpening — makes character edges crisper,
        # which measurably helps EasyOCR's recognition step on slightly
        # blurry or low-resolution phone photos.
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

        # Convert back to RGB for EasyOCR
        processed_img = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)

        return processed_img

    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise


def _reading_order_key(detection):
    """
    Sort key that approximates natural top-to-bottom, left-to-right reading
    order from an EasyOCR bounding box. EasyOCR's raw detection order is
    based on internal processing, not layout position, which can scatter
    a label's "MRP" and its number far apart in the joined text and break
    the regex-based compliance checks. Rounding the y-center into ~25px
    bands groups words on the same line together before sorting by x.
    """
    bbox = detection[0]
    y_center = sum(point[1] for point in bbox) / len(bbox)
    x_left = min(point[0] for point in bbox)
    row_band = round(y_center / 25)
    return (row_band, x_left)


def extract_text(image_path):
    """
    Extract text from a product label image using EasyOCR.
    Includes image preprocessing, confidence-based noise filtering, and
    reading-order sorting for better accuracy.

    Args:
        image_path (str): Path to the image file

    Returns:
        str: Extracted text from the image
    """
    try:
        # Preprocess the image
        processed_img = preprocess_image(image_path)

        # Get the cached reader instead of creating a new one every call
        reader = get_ocr_reader()

        # Extract text with tuned parameters for small/faded label print:
        #   - decoder='beamsearch': more accurate than the default greedy
        #     decode on noisy/garbled text, at a modest CPU-time cost
        #   - low_text / text_threshold lowered: catches lighter or
        #     slightly faded printing that the defaults miss
        #   - contrast_ths / adjust_contrast: boosts low-contrast regions
        #     (e.g. glare-washed sections of a label) before recognition
        #   - mag_ratio: upscales small text internally for recognition
        results = reader.readtext(
            processed_img,
            decoder='beamsearch',
            beamWidth=8,
            contrast_ths=0.1,
            adjust_contrast=0.6,
            text_threshold=0.6,
            low_text=0.35,
            mag_ratio=1.5,
        )

        if not results:
            logger.info(f"No text detected in {image_path}")
            return ""

        # Drop low-confidence garbage detections (blur/glare noise).
        # Safety net: if filtering would remove EVERYTHING, keep the
        # original unfiltered results rather than returning nothing.
        filtered = [r for r in results if r[2] >= MIN_CONFIDENCE]
        if not filtered:
            filtered = results

        # Sort into natural reading order so related tokens (e.g. "MRP"
        # and its number) end up adjacent in the joined text.
        filtered.sort(key=_reading_order_key)

        extracted_text = ' '.join(r[1] for r in filtered)

        avg_confidence = sum(r[2] for r in filtered) / len(filtered)
        logger.info(
            f"Successfully extracted text from {image_path} "
            f"({len(filtered)}/{len(results)} detections kept, "
            f"avg confidence {avg_confidence:.2f})"
        )
        return extracted_text

    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        raise
