import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_compliance(text, category="Other"):
    """
    Check if OCR-extracted text contains mandatory fields for Legal Metrology compliance.
    
    Args:
        text (str): Extracted text from OCR
        category (str): Product category (Food, Cosmetic, Household, Electronics, Textile, Other)
        
    Returns:
        dict: Dictionary containing:
            - fields_found: dict with field names and boolean status
            - compliance_score: percentage of fields found
            - is_compliant: boolean indicating overall compliance
            - category: the category used for checking
            - extra_fields: dict with category-specific fields
    """
    # Convert text to uppercase for case-insensitive matching
    text_upper = text.upper()
    
    # Define regex patterns for each mandatory field
    patterns = {
        'MRP': [
            r'MRP\s*[:\s]*\₹?\s*\d+\.?\d*',
            r'MAXIMUM\s+RETAIL\s+PRICE\s*[:\s]*\₹?\s*\d+\.?\d*',
            r'M\.R\.P\.?\s*[:\s]*\₹?\s*\d+\.?\d*'
        ],
        'Net Quantity': [
            r'NET\s+WEIGHT\s*[:\s]*\d+\.?\d*\s*(?:g|kg|ml|l|L|G|KG)',
            r'NET\s+QUANTITY\s*[:\s]*\d+\.?\d*\s*(?:g|kg|ml|l|L|G|KG)',
            r'NET\s+VOL\s*[:\s]*\d+\.?\d*\s*(?:ml|l|L)',
            r'WEIGHT\s*[:\s]*\d+\.?\d*\s*(?:g|kg|ml|l|L|G|KG)',
            r'QUANTITY\s*[:\s]*\d+\.?\d*\s*(?:g|kg|ml|l|L|G|KG)'
        ],
        'Manufacturing Date': [
            r'MFG\s*[:\s]*\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}',
            r'MANUFACTURED\s*[:\s]*\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}',
            r'MANUFACTURING\s+DATE\s*[:\s]*\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}',
            r'DATE\s+OF\s+MANUFACTURE\s*[:\s]*\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}'
        ],
        'Expiry/Best Before Date': [
            r'EXP\s*[:\s]*\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}',
            r'EXPIRY\s*[:\s]*\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}',
            r'EXPIRY\s+DATE\s*[:\s]*\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}',
            r'BEST\s+BEFORE\s*[:\s]*\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}',
            r'USE\s+BY\s*[:\s]*\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}'
        ],
        'Customer Care Number': [
            r'CUSTOMER\s+CARE\s*[:\s]*\+?\d{10,15}',
            r'CUSTOMER\s+SERVICE\s*[:\s]*\+?\d{10,15}',
            r'HELPLINE\s*[:\s]*\+?\d{10,15}',
            r'TOLL\s+FREE\s*[:\s]*\+?\d{10,15}',
            r'CONTACT\s*[:\s]*\+?\d{10,15}'
        ],
        'Manufacturer Address': [
            r'MANUFACTURED\s+BY\s*[:\s]*.+(?:street|road|city|state|india|IN)',
            r'MANUFACTURER\s*[:\s]*.+(?:street|road|city|state|india|IN)',
            r'MADE\s+BY\s*[:\s]*.+(?:street|road|city|state|india|IN)',
            r'ADDRESS\s*[:\s]*.+(?:street|road|city|state|india|IN)',
            r'Pvt\.?\s+Ltd\.?',
            r'LIMITED'
        ],
        'Unit Sale Price': [
            r'UNIT\s+SALE\s+PRICE\s*[:\s]*\₹?\s*\d+\.?\d*',
            r'UNIT\s+PRICE\s*[:\s]*\₹?\s*\d+\.?\d*',
            r'SALE\s+PRICE\s*[:\s]*\₹?\s*\d+\.?\d*',
            r'PRICE\s*[:\s]*\₹?\s*\d+\.?\d*'
        ],
        'Country of Origin': [
            r'COUNTRY\s+OF\s+ORIGIN\s*[:\s]*\w+',
            r'ORIGIN\s*[:\s]*\w+',
            r'MADE\s+IN\s*\w+',
            r'PRODUCT\s+OF\s*\w+'
        ]
    }
    
    # Category-specific additional fields
    category_patterns = {
        'Food': {
            'FSSAI License Number': [
                r'FSSAI\s*[:\s]*\d{14}',
                r'FSSAI\s+LICENSE\s*[:\s]*\d{14}',
                r'FOOD\s+SAFETY\s+LICENSE\s*[:\s]*\d{14}',
                r'LICENSE\s+NO\.?\s*[:\s]*\d{14}'
            ]
        }
    }
    
    # Add category-specific patterns if applicable
    extra_patterns = {}
    if category in category_patterns:
        extra_patterns = category_patterns[category]
        patterns.update(extra_patterns)
    
    # Check each field
    fields_found = {}
    extra_fields_found = {}
    
    for field, pattern_list in patterns.items():
        found = False
        for pattern in pattern_list:
            if re.search(pattern, text_upper, re.IGNORECASE):
                found = True
                break
        
        # Separate category-specific fields
        is_category_field = category in category_patterns and field in category_patterns[category]
        
        if is_category_field:
            extra_fields_found[field] = found
        else:
            fields_found[field] = found
            
        logger.info(f"Field '{field}': {'Found' if found else 'Not found'}")
    
    # Calculate compliance score (base 8 fields)
    base_fields = ['MRP', 'Net Quantity', 'Manufacturing Date', 'Expiry/Best Before Date', 
                  'Customer Care Number', 'Manufacturer Address', 'Unit Sale Price', 'Country of Origin']
    total_fields = len(base_fields)
    found_fields = sum(fields_found.get(field, False) for field in base_fields)
    compliance_score = (found_fields / total_fields) * 100
    
    # Determine overall compliance (all base fields must be present)
    is_compliant = all(fields_found.get(field, False) for field in base_fields)
    
    # For imported products, flag Country of Origin as extra critical
    is_imported = "IMPORT" in text_upper or "IMPORTED" in text_upper
    if is_imported and not fields_found.get('Country of Origin', False):
        logger.warning("Imported product detected but Country of Origin missing - critical violation")
    
    result = {
        'fields_found': fields_found,
        'compliance_score': round(compliance_score, 2),
        'is_compliant': is_compliant,
        'category': category,
        'extra_fields': extra_fields_found,
        'is_imported': is_imported
    }
    
    logger.info(f"Compliance check complete. Score: {compliance_score:.2f}%, Compliant: {is_compliant}, Category: {category}")
    return result
