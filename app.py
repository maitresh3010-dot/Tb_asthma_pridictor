import streamlit as st
import pickle
import io
import database
import utils
import numpy as np
from streamlit_mic_recorder import mic_recorder

# 1. Initialize Database
database.init_db()

# Page Configuration
st.set_page_config(page_title="Swaas-Check V2", page_icon="🫁", layout="centered")

# Navigation Sidebar
page = st.sidebar.selectbox("Navigation", ["🏠 Diagnostic Center", "📊 My Admin Dashboard"])

# Load AI Model
@st.cache_resource
def load_model():
    try:
        return pickle.load(open("audio_model.pkl", "rb"))
    except: 
        return None

model = load_model()

# --- PAGE 1: DIAGNOSTIC CENTER ---
if page == "🏠 Diagnostic Center":
    st.title("🫁 Swaas-Check Screening")
    st.write("Complete the details below to enable the cough analysis tools.")

    # Step 1: Registration
    st.subheader("📋 Step 1: Person Details")
    col1, col2 = st.columns(2)
    p_name = col1.text_input("Full Name")
    p_age = col2.number_input("Age", min_value=1, max_value=100, value=20)
    p_phone = st.text_input("Contact Number")

    # Step 2: Unlock Logic
    if p_name and p_phone:
        st.divider()
        st.subheader("🎙️ Step 2: Cough Sample Analysis")
        
        tab1, tab2 = st.tabs(["🎙️ Record Live", "📁 Upload WAV File"])
        audio_source = None

        with tab1:
            st.write("Record a clear 3-second sample:")
            audio_record = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop", key='recorder')
            if audio_record:
                audio_source = io.BytesIO(audio_record['bytes'])

        with tab2:
            uploaded_file = st.file_uploader("Upload recording", type=["wav"])
            if uploaded_file:
                audio_source = uploaded_file

        # Step 3: Analysis
        if audio_source and model:
            if st.button("🔍 Run AI Diagnostic"):
                with st.spinner("Analyzing spectral signatures..."):
                    features = utils.extract_45_features(audio_source)
                    
                    if features is not None:
                        features = features.reshape(1, -1)
                        prediction = model.predict(features)[0]
                        confidence = np.max(model.predict_proba(features)[0]) * 100

                        # Save to Backend
                        database.save_patient(p_name, p_age, p_phone, prediction, confidence)

                        # Feedback
                        st.divider()
                        if prediction == "NORMAL":
                            st.success(f"### Result: Healthy ({confidence:.1f}%)")
                            st.balloons()
                        else:
                            st.error(f"### Result: TB Pattern Detected ({confidence:.1f}%)")
                            st.snow()
    else:
        st.warning("⚠️ Please enter Name and Contact to unlock the diagnostic tools.")

# --- PAGE 2: SECURE ADMIN DASHBOARD ---
elif page == "📊 My Admin Dashboard":
    st.title("🛡️ Admin Results Portal")
    
    # 🔐 Password Gate
    password = st.text_input("Enter Admin Password", type="password")
    
    # Access check against Streamlit Secrets
    if password == st.secrets.get("ADMIN_PASSWORD", "temp_pass"):
        st.success("Access Granted")
        st.write("Reviewing all stored screening results.")

        # Fetch data from SQLite
        records = database.get_all_records()
        
        if not records.empty:
            st.write(f"Total Database Entries: {len(records)}")
            st.dataframe(records, use_container_width=True)
            
            # Export for Exhibition Report
            csv = records.to_csv(index=False).encode('utf-8')
            st.download_button("Download Full Report (CSV)", csv, "swaas_report.csv", "text/csv")
        else:
            st.info("No records found in the database yet.")
    
    elif password == "":
        st.info("Please enter the administrator password to view records.")
    else:
        st.error("❌ Incorrect Password. Access Denied.")