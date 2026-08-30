import os
from dotenv import load_dotenv
import google.generativeai as genai
import base64
import logging
import json
from typing import Dict, Any

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_compliance_with_gemini(image_path: str) -> Dict[str, Any]:
    """
    Check compliance using Google Gemini AI vision capabilities.
    
    Args:
        image_path (str): Path to the product label image
        
    Returns:
        dict: Dictionary containing:
            - fields: dict with field names and their presence, value, and confidence
            - overall_compliant: boolean indicating overall compliance
            - score: string like "X/8"
            - error: error message if any (None if successful)
    """
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        error_msg = "GEMINI_API_KEY environment variable not set. Please set it and try again, or use Fast/Manual mode."
        logger.error(error_msg)
        return {
            "error": error_msg,
            "fields": {},
            "overall_compliant": False,
            "score": "0/8"
        }
    
    try:
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Use a current vision-capable Gemini model
        model = genai.GenerativeModel('gemini-3.6-flash')
        
        # Read and encode image
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        
        # Create the prompt for Gemini
        prompt = """
        Analyze this product label image for compliance with India's Legal Metrology (Packaged Commodities) Rules, 2011.
        
        Identify and extract the following mandatory fields:
        1. MRP (Maximum Retail Price)
        2. Net Quantity
        3. Manufacturing Date
        4. Expiry/Best Before Date
        5. Customer Care Number
        6. Manufacturer Address
        7. Unit Sale Price
        8. Country of Origin
        
        For food products, also identify:
        9. FSSAI License Number (14-digit number)
        
        For each field, provide:
        - Whether it's present (true/false)
        - The exact extracted value if present
        - Your confidence score (0-100) for this detection
        
        Return ONLY a valid JSON object in this exact format:
        {
            "MRP": {"present": true/false, "value": "extracted text or null", "confidence": 0-100},
            "Net Quantity": {"present": true/false, "value": "extracted text or null", "confidence": 0-100},
            "Manufacturing Date": {"present": true/false, "value": "extracted text or null", "confidence": 0-100},
            "Expiry/Best Before Date": {"present": true/false, "value": "extracted text or null", "confidence": 0-100},
            "Customer Care Number": {"present": true/false, "value": "extracted text or null", "confidence": 0-100},
            "Manufacturer Address": {"present": true/false, "value": "extracted text or null", "confidence": 0-100},
            "Unit Sale Price": {"present": true/false, "value": "extracted text or null", "confidence": 0-100},
            "Country of Origin": {"present": true/false, "value": "extracted text or null", "confidence": 0-100},
            "FSSAI License Number": {"present": true/false, "value": "extracted text or null", "confidence": 0-100},
            "overall_compliant": true/false,
            "score": "X/8"
        }
        
        Be thorough and accurate. If a field is not found, set present to false and value to null. Ensure the JSON is valid and properly formatted.
        """
        
        # Create image part for the API call
        image_part = {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(image_data).decode()
        }
        
        # Generate content
        response = model.generate_content([prompt, image_part])
        
        # Parse the response
        response_text = response.text
        
        # Extract JSON from response (in case there's extra text)
        try:
            # Try to find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
            else:
                result = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            logger.error(f"Response text: {response_text}")
            return {
                "error": f"Failed to parse AI response. Please try again or use Fast/Manual mode.",
                "fields": {},
                "overall_compliant": False,
                "score": "0/8"
            }
        
        # Validate the result structure
        required_fields = ["MRP", "Net Quantity", "Manufacturing Date", "Expiry/Best Before Date", 
                          "Customer Care Number", "Manufacturer Address", "Unit Sale Price", "Country of Origin", "FSSAI License Number"]
        
        # Handle different possible response structures
        fields_data = {}
        if "fields" in result:
            fields_data = result["fields"]
        else:
            # Assume fields are at top level
            fields_data = {field: result.get(field, {"present": False, "value": None, "confidence": 0}) 
                          for field in required_fields}
        
        # Ensure all required fields exist with proper structure
        for field in required_fields:
            if field not in fields_data:
                fields_data[field] = {"present": False, "value": None, "confidence": 0}
            elif not isinstance(fields_data[field], dict):
                fields_data[field] = {"present": False, "value": None, "confidence": 0}
            else:
                # Ensure all required keys exist
                if "present" not in fields_data[field]:
                    fields_data[field]["present"] = False
                if "value" not in fields_data[field]:
                    fields_data[field]["value"] = None
                if "confidence" not in fields_data[field]:
                    fields_data[field]["confidence"] = 0
        
        # Calculate score if not provided (base 8 fields, FSSAI is extra)
        if "score" not in result:
            # Count only the 8 mandatory fields for score
            mandatory_fields = ["MRP", "Net Quantity", "Manufacturing Date", "Expiry/Best Before Date", 
                              "Customer Care Number", "Manufacturer Address", "Unit Sale Price", "Country of Origin"]
            present_count = sum(1 for field in mandatory_fields if fields_data[field].get("present", False))
            result["score"] = f"{present_count}/8"
        
        # Calculate overall compliance if not provided (based on 8 mandatory fields)
        if "overall_compliant" not in result:
            mandatory_fields = ["MRP", "Net Quantity", "Manufacturing Date", "Expiry/Best Before Date", 
                              "Customer Care Number", "Manufacturer Address", "Unit Sale Price", "Country of Origin"]
            result["overall_compliant"] = all(fields_data[field].get("present", False) for field in mandatory_fields)
        
        # Restructure to match expected format
        final_result = {
            "fields": fields_data,
            "overall_compliant": result["overall_compliant"],
            "score": result["score"],
            "error": None
        }
        
        logger.info(f"Gemini compliance check complete. Score: {final_result['score']}, Compliant: {final_result['overall_compliant']}")
        
        return final_result
        
    except Exception as e:
        error_msg = f"Error calling Gemini API: {str(e)}. Please check your internet connection and API key, or use Fast/Manual mode."
        logger.error(error_msg)
        return {
            "error": error_msg,
            "fields": {},
            "overall_compliant": False,
            "score": "0/8"
        }
