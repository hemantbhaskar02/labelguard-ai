import cv2
import numpy as np
import easyocr
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(image_path):
    """
    Preprocess image to improve OCR accuracy.
    Steps: grayscale, denoising, contrast enhancement
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Convert back to RGB for EasyOCR
        processed_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        return processed_img
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def extract_text(image_path):
    """
    Extract text from a product label image using EasyOCR.
    Includes image preprocessing for better accuracy.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Extracted text from the image
    """
    try:
        # Preprocess the image
        processed_img = preprocess_image(image_path)
        
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'], gpu=False)
        
        # Extract text
        results = reader.readtext(processed_img)
        
        # Combine all detected text
        extracted_text = ' '.join([text[1] for text in results])
        
        logger.info(f"Successfully extracted text from {image_path}")
        return extracted_text
        
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        raise
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        raise
