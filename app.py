import streamlit as st
import pandas as pd
import joblib
import sqlite3
import hashlib
import plotly.express as px
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Railway Safety Risk System",
                   page_icon="🚉",
                   layout="wide")

# ---------------- DATABASE ----------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS users(
    username TEXT PRIMARY KEY,
    password TEXT
)
''')
conn.commit()

# ---------------- PASSWORD HASH ----------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

# ---------------- AUTH FUNCTIONS ----------------
def login_user(username, password):
    hashed = hash_password(password)
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hashed))
    return c.fetchone()

def signup_user(username, password):
    try:
        hashed = hash_password(password)
        c.execute("INSERT INTO users(username, password) VALUES (?,?)",
                  (username, hashed))
        conn.commit()
        return True
    except:
        return False

# ---------------- LOGIN / SIGNUP ----------------
if not st.session_state.logged_in:

    st.title("🚉 Railway Station Safety Risk System")

    option = st.selectbox("Select Option", ["Login", "Signup"])

    if option == "Login":
        st.subheader("Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid Credentials")

    else:
        st.subheader("Create Account")

        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")

        if st.button("Signup"):
            if signup_user(new_user, new_pass):
                st.success("Account Created Successfully! Please Login.")
            else:
                st.error("Username already exists")

# ---------------- DASHBOARD ----------------
else:

    st.sidebar.success(f"Logged in as {st.session_state.username}")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

    st.title("🚉 Railway Safety Risk Dashboard")
    st.markdown("Upload dataset to analyze station safety patterns.")

    uploaded_file = st.file_uploader("Upload Raw CSV Dataset", type=["csv"])

    if uploaded_file is not None:

        try:
            # Load models
            scaler = joblib.load("scaler.pkl")
            pca = joblib.load("pca.pkl")
            kmeans = joblib.load("kmeans_model.pkl")

            raw = pd.read_csv(uploaded_file)

            # Keep only numeric columns for model
            numeric_data = raw.select_dtypes(include=["number"])

            # Scale + PCA
            scaled = scaler.transform(numeric_data)
            pca_data = pca.transform(scaled)

            # Predict clusters
            clusters = kmeans.predict(pca_data)

            # Compute distance from centroid (Risk Score)
            distances = kmeans.transform(pca_data)
            risk_score = distances.min(axis=1)

            raw["Risk_Score"] = risk_score

            # 3-Level Risk based on percentiles
            q1 = np.percentile(risk_score, 33)
            q2 = np.percentile(risk_score, 66)

            def classify_risk(x):
                if x <= q1:
                    return "Low Risk"
                elif x <= q2:
                    return "Medium Risk"
                else:
                    return "High Risk"

            raw["Risk_Level"] = raw["Risk_Score"].apply(classify_risk)

            # Add Station IDs
            raw["Station_ID"] = [
                "STN_" + str(i+1).zfill(4) for i in range(len(raw))
            ]

            # ---------------- SUMMARY METRICS ----------------
            total = len(raw)
            high = (raw["Risk_Level"] == "High Risk").sum()
            medium = (raw["Risk_Level"] == "Medium Risk").sum()
            low = (raw["Risk_Level"] == "Low Risk").sum()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Stations", total)
            col2.metric("High Risk", high)
            col3.metric("Medium Risk", medium)
            col4.metric("Low Risk", low)

            st.divider()

            # ---------------- CHART ----------------
            risk_counts = raw["Risk_Level"].value_counts().reset_index()
            risk_counts.columns = ["Risk_Level", "Count"]

            fig = px.bar(risk_counts,
                         x="Risk_Level",
                         y="Count",
                         text="Count",
                         color="Risk_Level")

            fig.update_layout(height=500)

            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # ---------------- TOP HIGH RISK ----------------
            st.subheader("🔴 Top 20 High Risk Stations")

            top_risk = raw.sort_values("Risk_Score",
                                       ascending=False).head(20)

            st.dataframe(
                top_risk[["Station_ID",
                          "Risk_Level",
                          "Risk_Score"]],
                use_container_width=True
            )

            st.divider()

            # ---------------- RECOMMENDATIONS ----------------
            st.subheader("📌 System Interpretation")

            st.info(
                "Risk level is calculated based on distance from cluster centroid. "
                "Stations far from normal operational behavior are categorized "
                "as higher risk."
            )

            st.error("🔴 High Risk: Immediate inspection recommended.")
            st.warning("🟠 Medium Risk: Schedule preventive maintenance.")
            st.success("🟢 Low Risk: Continue routine monitoring.")

            # ---------------- DOWNLOAD ----------------
            report = raw[["Station_ID",
                          "Risk_Level",
                          "Risk_Score"]]

            st.download_button("Download Risk Report",
                               report.to_csv(index=False),
                               "Railway_Risk_Report.csv")

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.info("Please upload dataset to begin analysis.")