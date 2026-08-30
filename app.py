import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from collections import Counter

# Set matplotlib style for dark theme
plt.style.use('dark_background')
sns.set_palette("husl")

from ocr_utils import extract_text
from compliance_check import check_compliance

# Try to import optional dependencies
# QR decode only needs pyzbar (qrcode is optional and used only if generating codes)
try:
    from pyzbar import pyzbar
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False
    pyzbar = None

try:
    from gemini_vision_check import check_compliance_with_gemini
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="LabelGuard AI - Compliance Scanner",
    page_icon="�️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for premium dark theme styling with new header/hero design
st.markdown("""
<style>
    /* Premium dark theme base */
    .stApp {
        background-color: #0F1117;
        color: #E2E8F0;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
        background-color: #0F1117;
    }
    
    /* Custom Header/Navbar */
    .custom-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .logo-badge {
        width: 56px;
        height: 56px;
        background: linear-gradient(135deg, #10B981 0%, #0F6B3F 100%);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .logo-text {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }
    
    .app-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }
    
    .app-subtitle {
        font-size: 0.7rem;
        font-weight: 600;
        color: #6B7280;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-top: 0.25rem;
    }
    
    .theme-toggle {
        font-size: 1.5rem;
        color: #9CA3AF;
        cursor: pointer;
        transition: color 0.2s ease;
    }
    
    .theme-toggle:hover {
        color: #FFFFFF;
    }
    
    /* Hero Section */
    .hero-section {
        padding: 3rem 0;
        margin-bottom: 3rem;
    }
    
    .hero-tag {
        font-size: 0.75rem;
        font-weight: 700;
        color: #10B981;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }
    
    .hero-headline {
        font-size: 3.5rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1.1;
        letter-spacing: -0.03em;
        margin-bottom: 1.5rem;
        max-width: 900px;
    }
    
    .hero-description {
        font-size: 1.125rem;
        color: #9CA3AF;
        line-height: 1.6;
        max-width: 700px;
        margin-bottom: 2rem;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        color: #10B981;
    }
    
    .badge-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    /* Stat Cards */
    .stat-card {
        background-color: #1A1D27;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .stat-card:hover {
        border-color: rgba(16, 185, 129, 0.2);
    }
    
    .stat-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1;
    }
    
    /* Minimal Feature Cards */
    .feature-card {
        background-color: #1A1D27;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        transition: all 0.2s ease;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .feature-card:hover {
        border-color: rgba(16, 185, 129, 0.2);
        transform: translateY(-2px);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        font-size: 0.875rem;
        color: #6B7280;
    }
    
    /* Header styling - minimal */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #9CA3AF;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Compliance score card - clean, minimal */
    .compliance-score {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        transition: all 0.2s ease;
    }
    
    .compliant {
        background-color: #1A1D27;
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .non-compliant {
        background-color: #1A1D27;
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    /* Field check items - clean cards */
    .field-check {
        font-size: 1rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background-color: #1A1D27;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.2s ease;
    }
    
    .field-found {
        border-left: 3px solid #10B981;
        color: #10B981;
    }
    
    .field-missing {
        border-left: 3px solid #EF4444;
        color: #EF4444;
    }
    
    /* Step indicators - clean, minimal */
    .step-indicator {
        font-size: 1.25rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 2rem;
        padding: 1rem;
        background-color: #1A1D27;
        border-left: 3px solid #10B981;
        border-radius: 4px;
    }
    
    /* Mode descriptions - subtle cards */
    .mode-description {
        font-size: 0.95rem;
        color: #9CA3AF;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background-color: #1A1D27;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
    }
    
    /* Disclaimer */
    .disclaimer {
        font-size: 0.85rem;
        color: #6B7280;
        text-align: center;
        margin-top: 4rem;
        padding: 1.5rem;
        background-color: #1A1D27;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
    }
    
    /* Card styling - premium dark cards */
    .info-card {
        background-color: #1A1D27;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Button styling - emerald accent */
    .stButton > button {
        background-color: #10B981;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.2);
    }
    
    .stButton > button:hover {
        background-color: #059669;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .stButton > button[kind="primary"] {
        background-color: #10B981;
    }
    
    /* Input styling - clean, minimal */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        background-color: #1A1D27;
        color: #E2E8F0;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #10B981;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.1);
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #6B7280;
    }
    
    /* Tab styling - clean underline style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        background-color: transparent;
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #6B7280;
        border-radius: 0;
        padding: 1rem 1.5rem;
        font-weight: 500;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        border-bottom: 2px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #10B981;
        border-bottom: 2px solid #10B981;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #9CA3AF;
    }
    
    /* Success and error messages - clean */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 8px;
        padding: 1rem;
        color: #10B981;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        padding: 1rem;
        color: #EF4444;
    }
    
    .stInfo {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 8px;
        padding: 1rem;
        color: #F59E0B;
    }
    
    /* Metric cards - clean dark theme */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Progress bar - emerald accent */
    .stProgress > div > div > div {
        background-color: #10B981;
    }
    
    /* File uploader - clean dark theme */
    .stFileUploader {
        border: 2px dashed rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 2rem;
        background-color: #1A1D27;
        transition: all 0.2s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(16, 185, 129, 0.3);
    }
    
    /* Camera input - match dark theme and emerald accent */
    [data-testid="stCameraInput"] {
        background-color: #1A1D27;
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 12px;
        padding: 0.75rem;
    }
    
    /* Sidebar - clean dark theme */
    .css-1d391kg {
        background-color: #1A1D27;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: #1A1D27;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1A1D27;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        color: #E2E8F0;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background-color: #1A1D27;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Form styling */
    .stForm {
        background-color: #1A1D27;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-headline {
            font-size: 2.5rem;
        }
        
        .custom-header {
            flex-direction: column;
            text-align: center;
            gap: 1rem;
        }
        
        .stat-value {
            font-size: 2rem;
        }
        
        .field-check {
            font-size: 0.95rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.75rem 1rem;
            font-size: 0.85rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for scan history
if 'scan_history' not in st.session_state:
    st.session_state.scan_history = []

def add_to_history(result, method="Manual", product_name="Unknown"):
    """Add scan result to history"""
    history_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'method': method,
        'product_name': product_name,
        'compliant': result.get('is_compliant', result.get('overall_compliant', False)),
        'score': result.get('compliance_score', result.get('score', '0/8')),
        'missing_fields': []
    }
    
    # Extract missing fields
    if 'fields_found' in result:
        missing = [field for field, found in result['fields_found'].items() if not found]
        history_entry['missing_fields'] = missing
    elif 'fields' in result:
        missing = [field for field, data in result['fields'].items() if not data.get('present', False)]
        history_entry['missing_fields'] = missing
    
    st.session_state.scan_history.append(history_entry)

def display_compliance_result(result, show_confidence=False):
    """Display compliance results with optional confidence scores"""
    # Get fields data based on result structure
    if 'fields_found' in result:
        fields_data = {field: {'present': found, 'value': None, 'confidence': None} 
                      for field, found in result['fields_found'].items()}
        overall_compliant = result['is_compliant']
        score = result['compliance_score']
    else:
        fields_data = result.get('fields', {})
        overall_compliant = result.get('overall_compliant', False)
        score = result.get('score', '0/8')
    
    # Display compliance score
    score_class = "compliant" if overall_compliant else "non-compliant"
    st.markdown(f"""
    <div class="compliance-score {score_class}">
        {score}
    </div>
    """, unsafe_allow_html=True)
    
    # Overall status
    if overall_compliant:
        st.success("✅ Product Label is COMPLIANT")
    else:
        st.error("❌ Product Label is NON-COMPLIANT")
    
    # Field checklist
    st.subheader("Mandatory Fields Check")
    for field, data in fields_data.items():
        present = data.get('present', False)
        value = data.get('value')
        confidence = data.get('confidence')
        
        status_class = "field-found" if present else "field-missing"
        status_icon = "✅" if present else "❌"
        
        # Build field display text
        field_text = f"{status_icon} {field}"
        if value and present:
            field_text += f" - {value}"
        if show_confidence and confidence is not None:
            field_text += f" ({confidence}% confidence)"
        
        st.markdown(f"""
        <div class="field-check {status_class}">
            {field_text}
        </div>
        """, unsafe_allow_html=True)
    
    # Show category-specific fields if present
    if 'extra_fields' in result and result['extra_fields']:
        st.subheader("Category-Specific Fields")
        for field, found in result['extra_fields'].items():
            status_class = "field-found" if found else "field-missing"
            status_icon = "✅" if found else "❌"
            st.markdown(f"""
            <div class="field-check {status_class}">
                {status_icon} {field}
            </div>
            """, unsafe_allow_html=True)
    
    # Show import warning if applicable
    if result.get('is_imported', False):
        st.warning("⚠️ Imported product detected - Country of Origin is critical")

def manual_entry_tab():
    """Manual entry form for product details with enhanced styling"""
    st.markdown('<div class="step-indicator">STEP 01: Enter Product Details</div>', unsafe_allow_html=True)
    
    # Get demo data if available
    demo_data = st.session_state.get('demo_data', {})
    
    # Info card
    st.markdown("""
    <div class='info-card'>
        <div style='display: flex; align-items: center; gap: 1rem;'>
            <div style='font-size: 2rem;'>💡</div>
            <div>
                <div style='font-weight: 600; color: #FFFFFF;'>Manual Entry Mode</div>
                <div style='color: #718096; font-size: 0.9rem;'>Enter product label details manually for compliance checking</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("manual_entry_form"):
        # Product information section
        st.markdown("### 🏷️ Product Information")
        st.markdown("")
        col1, col2 = st.columns(2)
        
        with col1:
            product_name = st.text_input("Product Name*", value=demo_data.get('product_name', ''), placeholder="e.g., Organic Honey")
            category = st.selectbox("Category*", ["Food", "Cosmetic", "Household", "Electronics", "Textile", "Other"])
        
        with col2:
            manufacturer_name = st.text_input("Manufacturer Name*", value=demo_data.get('manufacturer_name', ''), placeholder="e.g., Organic Foods Ltd")
            batch_number = st.text_input("Batch Number", value=demo_data.get('batch_number', ''), placeholder="e.g., BATCH12345")
        
        st.markdown("---")
        
        # Address and contact section
        st.markdown("### 📍 Address & Contact")
        st.markdown("")
        col3, col4 = st.columns(2)
        
        with col3:
            manufacturer_address = st.text_area("Manufacturer Address*", value=demo_data.get('manufacturer_address', ''), placeholder="Full address with city, state, country", height=100)
            customer_care = st.text_input("Customer Care Number*", value=demo_data.get('customer_care', ''), placeholder="e.g., 1800-123-4567")
        
        with col4:
            country_of_origin = st.text_input("Country of Origin*", value=demo_data.get('country_of_origin', ''), placeholder="e.g., India")
        
        st.markdown("---")
        
        # Pricing and quantity section
        st.markdown("### 💰 Pricing & Quantity")
        st.markdown("")
        col5, col6 = st.columns(2)
        
        with col5:
            net_quantity = st.text_input("Net Quantity*", value=demo_data.get('net_quantity', ''), placeholder="e.g., 500g, 1L")
            mrp = st.text_input("MRP (Maximum Retail Price)*", value=demo_data.get('mrp', ''), placeholder="e.g., ₹299")
        
        with col6:
            unit_sale_price = st.text_input("Unit Sale Price*", value=demo_data.get('unit_sale_price', ''), placeholder="e.g., ₹299")
        
        st.markdown("---")
        
        # Dates section
        st.markdown("### 📅 Important Dates")
        st.markdown("")
        col7, col8 = st.columns(2)
        
        with col7:
            mfg_date = st.text_input("Manufacturing Date*", value=demo_data.get('mfg_date', ''), placeholder="e.g., 01/01/2024")
        
        with col8:
            expiry_date = st.text_input("Expiry/Best Before Date*", value=demo_data.get('expiry_date', ''), placeholder="e.g., 01/01/2026")
        
        st.markdown("---")
        
        # Submit button with better styling
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            submitted = st.form_submit_button("🔍 Check Compliance", use_container_width=True, type="primary")
        
        if submitted:
            # Clear demo data after use
            if 'demo_data' in st.session_state:
                del st.session_state.demo_data
            
            # Combine all text for compliance checking
            all_text = f"{product_name} {manufacturer_name} {manufacturer_address} {net_quantity} {mrp} {unit_sale_price} {mfg_date} {expiry_date} {customer_care} {country_of_origin} {batch_number}"
            
            result = check_compliance(all_text, category)
            
            st.markdown("---")
            st.markdown('<div class="step-indicator">STEP 02: Compliance Results</div>', unsafe_allow_html=True)
            display_compliance_result(result)
            
            # Add to history
            add_to_history(result, method="Manual", product_name=product_name or "Unknown")

def camera_access_note():
    """Note about browser permission and HTTPS for st.camera_input."""
    st.markdown("""
    <div class='info-card' style='background-color: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3);'>
        <div style='display: flex; align-items: center; gap: 1rem;'>
            <div style='font-size: 1.5rem;'>📷</div>
            <div>
                <div style='font-weight: 600; color: #10B981;'>Camera access</div>
                <div style='color: #9CA3AF; font-size: 0.85rem;'>Allow camera permission in the browser. This works on localhost in most modern browsers and works fully over HTTPS when deployed (for example Streamlit Cloud).</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def choose_single_image(radio_key, uploader_key, camera_key, upload_help, camera_label):
    """Return one image from file upload or st.camera_input, plus the selected source label."""
    source = st.radio(
        "Image source",
        ["Upload Image", "Take Photo"],
        horizontal=True,
        key=radio_key,
        help="Upload a saved image or capture one with your camera",
    )
    if source == "Upload Image":
        image_file = st.file_uploader(
            "Choose an image file...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help=upload_help,
            key=uploader_key,
        )
        return image_file, source

    camera_access_note()
    image_file = st.camera_input(
        camera_label,
        key=camera_key,
        help="Allow camera access, then capture a clear photo.",
    )
    return image_file, source


def save_image_file(image_file, temp_path):
    """Write an uploaded or camera-captured file to disk for OCR / Gemini."""
    with open(temp_path, "wb") as f:
        f.write(image_file.getbuffer())
    return temp_path


def process_qr_image_file(image_file, method, preview_caption, data_key, category_key, button_key):
    """Decode a QR image (upload or camera) and run the shared compliance checklist."""
    try:
        image = Image.open(image_file)
        st.image(image, caption=preview_caption, use_container_width=True)
        decoded_objects = pyzbar.decode(image)
        if not decoded_objects:
            st.error("❌ No QR code detected in the image. Please try a clearer image.")
            return
        for obj in decoded_objects:
            qr_data = obj.data.decode('utf-8')
            st.success("✅ QR Code detected!")
            st.text_area("QR Code Data", qr_data, height=150, key=data_key)
            category = st.selectbox(
                "Product Category",
                ["Food", "Cosmetic", "Household", "Electronics", "Textile", "Other"],
                key=category_key,
            )
            if st.button("Check Compliance from QR Data", key=button_key):
                result = check_compliance(qr_data, category)
                st.markdown("---")
                st.markdown('<div class="step-indicator">STEP 03: Compliance Results</div>', unsafe_allow_html=True)
                display_compliance_result(result)
                add_to_history(result, method=method, product_name="QR Code Data")
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("💡 Please try a clearer QR code or use the other image source.")


def photo_ocr_tab():
    """Photo/OCR upload with Fast and Accurate modes with enhanced styling"""
    st.markdown('<div class="step-indicator">STEP 01: Choose OCR Mode</div>', unsafe_allow_html=True)
    
    # Mode selection with better cards
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("""
            <div class='info-card' style='border-left: 4px solid #667eea;'>
                <div style='display: flex; align-items: center; gap: 1rem;'>
                    <div style='font-size: 2rem;'>⚡</div>
                    <div>
                        <div style='font-weight: 600; color: #FFFFFF;'>Fast Mode (Offline)</div>
                        <div style='color: #718096; font-size: 0.85rem;'>Instant results • Works offline • Manual review needed</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if GEMINI_AVAILABLE:
            with st.container():
                st.markdown("""
                <div class='info-card' style='border-left: 4px solid #764ba2;'>
                    <div style='display: flex; align-items: center; gap: 1rem;'>
                        <div style='font-size: 2rem;'>🤖</div>
                        <div>
                            <div style='font-weight: 600; color: #FFFFFF;'>Accurate Mode (Online)</div>
                            <div style='color: #718096; font-size: 0.85rem;'>AI-powered • Higher accuracy • Fully automatic</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown("""
                <div class='info-card' style='border-left: 4px solid #cbd5e0; opacity: 0.6;'>
                    <div style='display: flex; align-items: center; gap: 1rem;'>
                        <div style='font-size: 2rem;'>🔒</div>
                        <div>
                            <div style='font-weight: 600; color: #FFFFFF;'>Accurate Mode (Locked)</div>
                            <div style='color: #718096; font-size: 0.85rem;'>Install google-generativeai to enable</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Only show Gemini option if available
    mode_options = ["Fast Mode (Offline)"]
    if GEMINI_AVAILABLE:
        mode_options.append("Accurate Mode (Online - Gemini AI)")
    
    mode = st.radio(
        "Select Mode",
        mode_options,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown('<div class="step-indicator">STEP 02: Provide Product Image</div>', unsafe_allow_html=True)
    
    uploaded_file, image_source = choose_single_image(
        radio_key="photo_ocr_image_source",
        uploader_key="photo_ocr_uploader",
        camera_key="photo_ocr_camera",
        upload_help="Upload a clear, well-lit image of the product label",
        camera_label="Take a photo of the product label",
    )
    
    # Category selection
    category = st.selectbox(
        "🏷️ Product Category",
        ["Food", "Cosmetic", "Household", "Electronics", "Textile", "Other"],
        help="Select the category for compliance rules"
    )
    
    if uploaded_file is not None:
        # Image preview with better styling
        image = Image.open(uploaded_file)
        preview_title = "Captured Product Label" if image_source == "Take Photo" else "Uploaded Product Label"
        st.markdown(f"""
        <div class='info-card'>
            <div style='font-weight: 600; color: #FFFFFF; margin-bottom: 0.5rem;'>📸 {preview_title}</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        
        # Save uploaded or captured file temporarily
        temp_path = os.path.join("temp_image.jpg")
        save_image_file(uploaded_file, temp_path)
        
        # Process button with better styling
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            if st.button("🔍 Check Compliance", use_container_width=True, type="primary"):
                with st.spinner("🔄 Processing image and checking compliance..."):
                    try:
                        if mode == "Fast Mode (Offline)":
                            # Use EasyOCR
                            extracted_text = extract_text(temp_path)
                            
                            # Show extracted text for review
                            st.markdown("---")
                            st.markdown('<div class="step-indicator">STEP 03: Review Extracted Text</div>', unsafe_allow_html=True)
                            
                            st.markdown("""
                            <div class='info-card'>
                                <div style='font-weight: 600; color: #FFFFFF; margin-bottom: 0.5rem;'>✏️ Review and Edit OCR Results</div>
                                <div style='color: #718096; font-size: 0.85rem;'>Edit the extracted text if needed before checking compliance</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            edited_text = st.text_area("OCR Extracted Text", extracted_text, height=200, key="fast_ocr_text")
                            
                            col_left, col_center, col_right = st.columns([1, 2, 1])
                            with col_center:
                                if st.button("✅ Check Compliance with Edited Text", use_container_width=True):
                                    result = check_compliance(edited_text, category)
                                    
                                    st.markdown("---")
                                    st.markdown('<div class="step-indicator">STEP 04: Compliance Results</div>', unsafe_allow_html=True)
                                    display_compliance_result(result)
                                    
                                    # Add to history
                                    add_to_history(
                                        result,
                                        method="Fast OCR",
                                        product_name="Camera Photo" if image_source == "Take Photo" else "Uploaded Image",
                                    )
                        
                        else:  # Accurate Mode with Gemini
                            if not GEMINI_AVAILABLE:
                                st.error("⚠️ Gemini AI features require 'google-generativeai' package")
                                st.info("💡 Install it using: pip install google-generativeai")
                                # Fall back to OCR
                                extracted_text = extract_text(temp_path)
                                st.markdown("---")
                                st.markdown('<div class="step-indicator">STEP 03: Review Extracted Text</div>', unsafe_allow_html=True)
                                edited_text = st.text_area("OCR Extracted Text", extracted_text, height=200, key="fallback_ocr_text")
                                
                                col_left, col_center, col_right = st.columns([1, 2, 1])
                                with col_center:
                                    if st.button("✅ Check Compliance with Edited Text", use_container_width=True):
                                        result = check_compliance(edited_text, category)
                                        
                                        st.markdown("---")
                                        st.markdown('<div class="step-indicator">STEP 04: Compliance Results</div>', unsafe_allow_html=True)
                                        display_compliance_result(result)
                                        
                                        # Add to history
                                        add_to_history(
                                            result,
                                            method="Fast OCR (Fallback)",
                                            product_name="Camera Photo" if image_source == "Take Photo" else "Uploaded Image",
                                        )
                            else:
                                # Use Gemini AI
                                result = check_compliance_with_gemini(temp_path)
                                
                                if "error" in result and result["error"]:
                                    st.error(result["error"])
                                    st.info("💡 Try switching to Fast Mode or Manual Entry if the issue persists.")
                                else:
                                    st.markdown("---")
                                    st.markdown('<div class="step-indicator">STEP 03: Compliance Results</div>', unsafe_allow_html=True)
                                    display_compliance_result(result, show_confidence=True)
                                    
                                    # Add to history
                                    add_to_history(
                                        result,
                                        method="Gemini AI",
                                        product_name="Camera Photo" if image_source == "Take Photo" else "Uploaded Image",
                                    )
                        
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
                    except Exception as e:
                        st.error(f"❌ Error processing image: {str(e)}")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

def qr_scanner_tab():
    """QR code scanner for product label data with camera and upload options"""
    if not QR_AVAILABLE:
        st.error("⚠️ QR scanning features require the 'pyzbar' package. Please install it using: pip install pyzbar")
        st.info("💡 You can still use Manual Entry or Photo/OCR modes for compliance checking.")
        return
    
    st.markdown('<div class="step-indicator">STEP 01: Choose QR Scanning Method</div>', unsafe_allow_html=True)
    
    # Add toggle between upload and camera
    scan_method = st.radio(
        "Scanning Method",
        ["Upload QR Image", "Scan with Camera"],
        help="Choose how you want to provide the QR code"
    )
    
    if scan_method == "Upload QR Image":
        st.markdown('<div class="step-indicator">STEP 02: Provide QR Code Image</div>', unsafe_allow_html=True)
        
        qr_file, qr_source = choose_single_image(
            radio_key="qr_upload_image_source",
            uploader_key="qr_uploader",
            camera_key="qr_upload_camera",
            upload_help="Upload an image containing a QR code with product label data",
            camera_label="Take a photo of the QR code",
        )
        
        if qr_file is not None:
            process_qr_image_file(
                qr_file,
                method="QR Scanner (Camera)" if qr_source == "Take Photo" else "QR Scanner (Upload)",
                preview_caption="Captured QR Code" if qr_source == "Take Photo" else "Uploaded QR Code",
                data_key="qr_upload_data",
                category_key="qr_upload_category",
                button_key="qr_upload_check",
            )
    
    else:  # Scan with Camera
        st.markdown('<div class="step-indicator">STEP 02: Scan QR Code with Camera</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-card'>
            <div style='display: flex; align-items: center; gap: 1rem;'>
                <div style='font-size: 2rem;'>📷</div>
                <div>
                    <div style='font-weight: 600; color: #FFFFFF;'>Take Photo of QR Code</div>
                    <div style='color: #9CA3AF; font-size: 0.9rem;'>Capture a still photo of a QR code with your camera</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        camera_access_note()
        camera_image = st.camera_input(
            "Capture QR Code with Camera",
            key="qr_scan_camera",
            help="Allow camera access, then capture a clear photo of the QR code.",
        )
        
        if camera_image is not None:
            process_qr_image_file(
                camera_image,
                method="QR Scanner (Camera)",
                preview_caption="Captured QR Code",
                data_key="qr_camera_data",
                category_key="qr_camera_category",
                button_key="qr_camera_check",
            )
            if st.button("📷 Capture Another Image", key="qr_camera_recapture"):
                st.rerun()

def bulk_scan_tab():
    """Bulk/Batch upload mode for multiple images"""
    st.markdown('<div class="step-indicator">STEP 01: Upload Multiple Images</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-card'>
        <div style='color: #9CA3AF; font-size: 0.9rem;'>For single quick scans, use Photo/OCR tab's Take Photo option</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose multiple image files...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True,
        help="Upload multiple product label images for batch processing"
    )
    
    category = st.selectbox("Product Category for All Images", ["Food", "Cosmetic", "Household", "Electronics", "Textile", "Other"])
    
    if uploaded_files and st.button("Process Batch", type="primary"):
        if not GEMINI_AVAILABLE or not os.getenv("GEMINI_API_KEY"):
            st.warning("⚠️ Gemini AI not available or API key not set. Using Fast Mode (OCR) for batch processing.")
        
        results = []
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # Save uploaded file temporarily
                    temp_path = os.path.join(f"temp_bulk_{i}.jpg")
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Try Gemini first, fall back to OCR
                    if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
                        result = check_compliance_with_gemini(temp_path)
                        if "error" in result and result["error"]:
                            # Fall back to OCR
                            extracted_text = extract_text(temp_path)
                            result = check_compliance(extracted_text, category)
                            method = "Fast OCR (Fallback)"
                        else:
                            method = "Gemini AI"
                    else:
                        # Use OCR directly
                        extracted_text = extract_text(temp_path)
                        result = check_compliance(extracted_text, category)
                        method = "Fast OCR"
                    
                    # Extract missing fields
                    missing_fields = []
                    if 'fields_found' in result:
                        missing_fields = [field for field, found in result['fields_found'].items() if not found]
                    elif 'fields' in result:
                        missing_fields = [field for field, data in result['fields'].items() if not data.get('present', False)]
                    
                    results.append({
                        'filename': uploaded_file.name,
                        'compliant': result.get('is_compliant', result.get('overall_compliant', False)),
                        'score': result.get('compliance_score', result.get('score', '0/8')),
                        'missing_fields': ', '.join(missing_fields) if missing_fields else 'None',
                        'method': method
                    })
                    
                    # Add to history
                    add_to_history(result, method=method, product_name=uploaded_file.name)
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    results.append({
                        'filename': uploaded_file.name,
                        'compliant': False,
                        'score': 'Error',
                        'missing_fields': str(e),
                        'method': 'Error'
                    })
        
        # Display results table
        st.markdown('<div class="step-indicator">STEP 02: Batch Results</div>', unsafe_allow_html=True)
        
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Images", len(results))
            col2.metric("Compliant", sum(1 for r in results if r['compliant']))
            col3.metric("Non-Compliant", sum(1 for r in results if not r['compliant']))

def analytics_dashboard_tab():
    """Analytics dashboard showing compliance insights with enhanced styling"""
    st.markdown('<div class="step-indicator">Analytics Dashboard</div>', unsafe_allow_html=True)
    
    if not st.session_state.scan_history:
        st.markdown("""
        <div class='info-card' style='text-align: center; padding: 2rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>📭</div>
            <div style='font-weight: 600; color: #FFFFFF; margin-bottom: 0.5rem;'>No Scan History Available</div>
            <div style='color: #718096; font-size: 0.9rem;'>Start scanning products to see analytics and insights</div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Convert history to DataFrame
    df = pd.DataFrame(st.session_state.scan_history)
    
    # Summary statistics with better styling
    st.markdown("### 📈 Overall Statistics")
    st.markdown("")
    col1, col2, col3, col4 = st.columns(4)
    
    total_scans = len(df)
    compliant_count = sum(df['compliant'])
    non_compliant_count = total_scans - compliant_count
    compliance_rate = (compliant_count / total_scans * 100) if total_scans > 0 else 0
    
    with col1:
        st.metric("Total Scans", total_scans)
    with col2:
        st.metric("Compliant", compliant_count, delta=f"{compliance_rate:.1f}%")
    with col3:
        st.metric("Non-Compliant", non_compliant_count)
    with col4:
        st.metric("Compliance Rate", f"{compliance_rate:.1f}%")
    
    st.markdown("---")
    
    # Compliance vs Non-Compliance chart with better styling
    st.markdown("### 🎯 Compliance Distribution")
    st.markdown("")
    compliance_counts = df['compliant'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0F1117')
        ax.set_facecolor('#1A1D27')
        colors = ['#10B981' if idx else '#EF4444' for idx in compliance_counts.index]
        compliance_counts.plot(kind='bar', ax=ax, color=colors)
        ax.set_xlabel('Compliance Status', fontweight='bold', color='#E2E8F0')
        ax.set_ylabel('Count', fontweight='bold', color='#E2E8F0')
        ax.set_title('Compliant vs Non-Compliant Scans', fontweight='bold', pad=20, color='#E2E8F0')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#4B5563')
        ax.spines['left'].set_color('#4B5563')
        ax.tick_params(colors='#9CA3AF')
        plt.xticks(rotation=0, fontweight='600')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # Method distribution
        st.markdown("### 🔍 Scan Method Distribution")
        st.markdown("")
        method_counts = df['method'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0F1117')
        ax.set_facecolor('#1A1D27')
        method_counts.plot(kind='bar', ax=ax, color='#10B981')
        ax.set_xlabel('Method', fontweight='bold', color='#E2E8F0')
        ax.set_ylabel('Count', fontweight='bold', color='#E2E8F0')
        ax.set_title('Scans by Method', fontweight='bold', pad=20, color='#E2E8F0')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#4B5563')
        ax.spines['left'].set_color('#4B5563')
        ax.tick_params(colors='#9CA3AF')
        plt.xticks(rotation=45, ha='right', fontweight='600')
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Most frequently missing fields with better styling
    st.markdown("### ⚠️ Most Frequently Missing Fields")
    st.markdown("")
    all_missing = []
    for missing_fields in df['missing_fields']:
        if isinstance(missing_fields, list):
            all_missing.extend(missing_fields)
    
    if all_missing:
        missing_counts = Counter(all_missing)
        missing_df = pd.DataFrame(missing_counts.items(), columns=['Field', 'Count'])
        missing_df = missing_df.sort_values('Count', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#0F1117')
        ax.set_facecolor('#1A1D27')
        missing_df.head(10).plot(kind='bar', x='Field', y='Count', ax=ax, color='#F59E0B')
        ax.set_xlabel('Field', fontweight='bold', color='#E2E8F0')
        ax.set_ylabel('Missing Count', fontweight='bold', color='#E2E8F0')
        ax.set_title('Top 10 Most Frequently Missing Fields', fontweight='bold', pad=20, color='#E2E8F0')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#4B5563')
        ax.spines['left'].set_color('#4B5563')
        ax.tick_params(colors='#9CA3AF')
        plt.xticks(rotation=45, ha='right', fontweight='600')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Table view of missing fields
        st.markdown("### 📋 Detailed Missing Fields Analysis")
        st.dataframe(missing_df.head(10), use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div class='info-card' style='text-align: center;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🎉</div>
            <div style='font-weight: 600; color: #10B981;'>Excellent!</div>
            <div style='color: #9CA3AF; font-size: 0.9rem;'>No missing fields in recent scans</div>
        </div>
        """, unsafe_allow_html=True)

def sidebar_history():
    """Minimal sidebar - only essential branding"""
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 2rem 1rem;'>
        <div style='width: 48px; height: 48px; background: linear-gradient(135deg, #10B981 0%, #0F6B3F 100%); border-radius: 12px; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem;'>
            <svg width='24' height='24' viewBox='0 0 32 32' xmlns='http://www.w3.org/2000/svg'>
                <path d='M16 3 L27 7 V15 C27 22 22 27 16 29 C10 27 5 22 5 15 V7 Z' fill='none' stroke='white' stroke-width='2' stroke-linejoin='round'/>
                <path d='M11 16 L14.5 19.5 L21 12' stroke='white' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round' fill='none'/>
            </svg>
        </div>
        <div style='font-weight: 700; color: #FFFFFF; font-size: 1.1rem;'>LabelGuard <span style='color: #10B981;'>AI</span></div>
        <div style='color: #9CA3AF; font-size: 0.7rem; margin-top: 0.5rem; letter-spacing: 0.2em; text-transform: uppercase;'>Compliance Scanner</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Quick links
    st.sidebar.markdown("""
    <div style='color: #FFFFFF; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.75rem;'>
        Quick Links
    </div>
    """, unsafe_allow_html=True)
    
    # Add quick navigation links
    st.sidebar.markdown("""
    <div style='color: #9CA3AF; font-size: 0.85rem; line-height: 1.6;'>
        <div style='margin-bottom: 0.5rem;'>• Manual Entry</div>
        <div style='margin-bottom: 0.5rem;'>• Photo/OCR Scan</div>
        <div style='margin-bottom: 0.5rem;'>• Analytics Dashboard</div>
        <div style='margin-bottom: 0.5rem;'>• Scan History</div>
    </div>
    """, unsafe_allow_html=True)

def scan_history_tab():
    """New dedicated tab for scan history with full-width layout"""
    st.markdown('<div class="step-indicator">Scan History</div>', unsafe_allow_html=True)
    
    if not st.session_state.scan_history:
        st.markdown("""
        <div class='info-card' style='text-align: center; padding: 3rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>📭</div>
            <div style='font-weight: 600; color: #FFFFFF; margin-bottom: 0.5rem;'>No Scan History</div>
            <div style='color: #9CA3AF; font-size: 0.9rem;'>Start scanning products to see your compliance history</div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Convert history to DataFrame
    df = pd.DataFrame(st.session_state.scan_history)
    
    # Summary statistics with premium styling
    st.markdown("### 📈 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    total_scans = len(df)
    compliant_count = sum(df['compliant'])
    non_compliant_count = total_scans - compliant_count
    compliance_rate = (compliant_count / total_scans * 100) if total_scans > 0 else 0
    
    with col1:
        st.metric("Total Scans", total_scans)
    with col2:
        st.metric("Compliant", compliant_count, delta=f"{compliance_rate:.1f}%")
    with col3:
        st.metric("Non-Compliant", non_compliant_count)
    with col4:
        st.metric("Compliance Rate", f"{compliance_rate:.1f}%")
    
    st.markdown("---")
    
    # Detailed history table
    st.markdown("### 📋 Scan History")
    
    # Create a clean dataframe for display
    display_df = df.copy()
    display_df['status'] = display_df['compliant'].apply(lambda x: '✅ Compliant' if x else '❌ Non-Compliant')
    display_df = display_df[['timestamp', 'product_name', 'method', 'status', 'score']]
    display_df.columns = ['Timestamp', 'Product', 'Method', 'Status', 'Score']
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Clear history button
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        if st.button("🗑️ Clear All History", use_container_width=True):
            st.session_state.scan_history = []
            st.success("✅ History cleared!")
            st.rerun()

def load_demo_product():
    """Load demo product data for testing"""
    demo_data = {
        'product_name': 'Demo Organic Honey',
        'category': 'Food',
        'manufacturer_name': 'Organic Foods Ltd',
        'manufacturer_address': '123 Food Street, Mumbai, India',
        'net_quantity': '500g',
        'mrp': '₹299',
        'unit_sale_price': '₹299',
        'mfg_date': '01/01/2024',
        'expiry_date': '01/01/2026',
        'customer_care': '1800-123-4567',
        'country_of_origin': 'India',
        'batch_number': 'BATCH12345'
    }
    return demo_data

# Main application
def main():
    # Custom Header/Navbar
    st.markdown("""
    <div class='custom-header'>
        <div style='display: flex; align-items: center; gap: 1rem;'>
            <div class='logo-badge'>
                <svg width='32' height='32' viewBox='0 0 32 32' xmlns='http://www.w3.org/2000/svg'>
                    <path d='M16 3 L27 7 V15 C27 22 22 27 16 29 C10 27 5 22 5 15 V7 Z' fill='none' stroke='white' stroke-width='2' stroke-linejoin='round'/>
                    <path d='M11 16 L14.5 19.5 L21 12' stroke='white' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round' fill='none'/>
                </svg>
            </div>
            <div>
                <div class='app-name'>LabelGuard <span style='color: #10B981;'>AI</span></div>
                <div class='app-subtitle'>COMPLIANCE SCANNER</div>
            </div>
        </div>
        <div class='theme-toggle'>🌙</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class='hero-section'>
        <div class='hero-tag'>COLLEGE PROJECT • INDIA</div>
        <div class='hero-headline'>Check a package before it reaches the shelf.</div>
        <div class='hero-description'>
            Three powerful ways to verify product compliance: manual entry for complete control, 
            OCR scanning for quick analysis, or AI-powered detection for maximum accuracy. 
            Ensure every product meets India's compliance standards before distribution.
        </div>
        <div class='hero-badge'>
            <div class='badge-dot'></div>
            AI-Powered
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stat Cards Row
    st.markdown("<div style='margin-bottom: 3rem;'>", unsafe_allow_html=True)
    
    # Calculate stats
    total_scans = len(st.session_state.scan_history) if st.session_state.scan_history else 0
    compliant_count = sum(1 for scan in st.session_state.scan_history if scan['compliant']) if st.session_state.scan_history else 0
    non_compliant_count = total_scans - compliant_count
    saved_count = total_scans  # Using total scans as saved count for now
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>Scans</div>
            <div class='stat-value'>{total_scans}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>Compliant</div>
            <div class='stat-value'>{compliant_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>Needs Review</div>
            <div class='stat-value'>{non_compliant_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-label'>Saved</div>
            <div class='stat-value'>{saved_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Minimal Feature Cards
    st.markdown("<div style='margin-bottom: 3rem;'>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>📝</div>
            <div class='feature-title'>Manual Entry</div>
            <div class='feature-description'>Complete control</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>📷</div>
            <div class='feature-title'>OCR Scan</div>
            <div class='feature-description'>Quick analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>🤖</div>
            <div class='feature-title'>AI Powered</div>
            <div class='feature-description'>Maximum accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>📊</div>
            <div class='feature-title'>Analytics</div>
            <div class='feature-description'>Track compliance</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar with history
    sidebar_history()
    
    # Main tabs with Scan History added
    if QR_AVAILABLE:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Manual Entry", 
            "Photo/OCR", 
            "QR Scanner", 
            "Bulk Scan", 
            "Analytics Dashboard",
            "Scan History"
        ])
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Manual Entry", 
            "Photo/OCR", 
            "Bulk Scan", 
            "Analytics Dashboard",
            "Scan History"
        ])
    
    with tab1:
        # Load Demo Product button with better styling
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            if st.button("🎯 Load Demo Product", use_container_width=True):
                demo_data = load_demo_product()
                st.session_state.demo_data = demo_data
                st.success("✅ Demo product loaded! Fill in the form with the pre-loaded data.")
                st.rerun()
        
        # Pre-fill form if demo data is loaded
        if 'demo_data' in st.session_state:
            st.markdown("""
            <div class='info-card' style='background: linear-gradient(90deg, #84fab0 0%, #8fd3f4 100%);'>
                <div style='display: flex; align-items: center; gap: 1rem;'>
                    <div style='font-size: 2rem;'>✅</div>
                    <div>
                        <div style='font-weight: 600; color: #155724;'>Demo data loaded!</div>
                        <div style='color: #155724; font-size: 0.9rem;'>Form fields are pre-filled with sample compliant product data</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        manual_entry_tab()
    
    with tab2:
        photo_ocr_tab()
    
    if QR_AVAILABLE:
        with tab3:
            qr_scanner_tab()
        with tab4:
            bulk_scan_tab()
        with tab5:
            analytics_dashboard_tab()
        with tab6:
            scan_history_tab()
    else:
        with tab3:
            bulk_scan_tab()
        with tab4:
            analytics_dashboard_tab()
        with tab5:
            scan_history_tab()
    
    # Footer disclaimer
    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
        LabelGuard <span style='color: #10B981;'>AI</span> - This is a rule-based/AI-assisted screening tool for educational and demonstration purposes, not an official government certification.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
