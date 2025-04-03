# NOTE: This script requires Streamlit and Transformers or a custom moderation model.
# Run with: streamlit run app.py

# Attempt to import libraries and set a fallback mode if unavailable
streamlit_available = True
custom_model_available = True

try:
    import streamlit as st
except ModuleNotFoundError:
    streamlit_available = False
    st = None

# Try loading custom model, fallback to transformers demo pipeline
try:
    from moderation import load_model, classify_text
except ModuleNotFoundError:
    custom_model_available = False
    from transformers import pipeline

    def load_model():
        return pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

    def classify_text(model, text):
        return model(text)[0]

if not streamlit_available:
    print("Streamlit is not available. Please install it with: pip install streamlit")
else:
    import pandas as pd
    import sqlite3

    # Connect to SQLite and ensure the table exists
    conn = sqlite3.connect("moderation.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS moderation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT,
            severity INTEGER,
            category TEXT,
            flagged TEXT,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    result = None  # Store result globally for use in dashboard

    # Load moderation model (custom or fallback)
    @st.cache_resource
    def load_moderation_model():
        return load_model()

    model = load_moderation_model()

    # Severity and category estimation logic (with 0.75 threshold for flagging)
    def interpret_results(preds):
        if not isinstance(preds, list) or not all('label' in p and 'score' in p for p in preds):
            return {
                "severity": 0,
                "category": "Unknown",
                "explanation": "Invalid prediction format.",
                "flagged": "Yes"
            }

        high_confidence_categories = [p['label'] for p in preds if p['score'] > 0.75]
        severity = min(len(high_confidence_categories), 5)
        explanation = ", ".join(high_confidence_categories) if high_confidence_categories else "No issues detected."
        flagged = "Yes" if high_confidence_categories else "No"
        category = high_confidence_categories[0] if high_confidence_categories else "Safe"

        return {
            "severity": severity,
            "category": category,
            "explanation": explanation,
            "flagged": flagged
        }

    # Streamlit UI Tabs
    st.title("Shimmy Content Moderation System")
    tab1, tab2 = st.tabs(["User Input", "Dashboard"])

    with tab1:
        st.header("Post Submission")
        user_input = st.text_area("Enter a post/comment:", height=150)

        if st.button("Submit") and user_input:
            with st.spinner("Analyzing..."):
                try:
                    predictions = classify_text(model, user_input)
                    result = interpret_results(predictions)
                    cursor.execute("""
                        INSERT INTO moderation_log (message, severity, category, flagged, explanation)
                        VALUES (?, ?, ?, ?, ?)
                    """, (user_input, result['severity'], result['category'], result['flagged'], result['explanation']))
                    conn.commit()
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
                    result = {
                        "severity": 0,
                        "category": "Error",
                        "explanation": str(e),
                        "flagged": "Yes"
                    }

    with tab2:
        st.header("Moderation Dashboard")

        if result:
            st.subheader("Latest Moderation Result")
            st.write(f"**Severity Level**: {result['severity']} / 5")
            st.write(f"**Category**: {result['category']}")
            st.write(f"**Justification**: {result['explanation']}")
            st.write(f"**Flagged for Review**: {result['flagged']}")
            st.success("Content processed successfully.")

        df = pd.read_sql_query("SELECT message AS Message, severity AS Severity, category AS Category, flagged AS Flagged, explanation AS Explanation, created_at AS Timestamp FROM moderation_log ORDER BY created_at DESC", conn)

        if not df.empty:
            st.dataframe(df)

            st.subheader("Flagged Content")
            flagged_df = df[df['Flagged'] == "Yes"]
            if not flagged_df.empty:
                st.write(flagged_df[['Message', 'Severity', 'Category', 'Explanation']])
            else:
                st.write("No content flagged for manual review.")
        else:
            st.info("No moderation data available yet. Submit a post in the User Input tab.")

    st.markdown("---")
    if custom_model_available:
        st.caption("Note: This demo uses a custom moderation model from 'moderation.py'.")
    else:
        st.caption("Note: This demo uses the fallback `toxic-bert` pipeline from HuggingFace Transformers.")
