# NOTE: This script requires Streamlit and Transformers or a custom moderation model.
# Run with: streamlit run app_version0406.py

# Attempt to import libraries and set a fallback mode if unavailable
streamlit_available = True
custom_model_available = True

try:
    import streamlit as st
except ModuleNotFoundError:
    streamlit_available = False
    st = None

from model_config import SentimentClassifier, Reasoner
import os 
import re
from pathlib import Path

# print(st.__version__)  # Check Streamlit version 1.38.0

def load_model():
    return SentimentClassifier()
    # return pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

def load_reasoner(text, file_path):
    return Reasoner(text, file_path)

def classify_text(model, text):
    return model.predict(text)
    # return model(text)[0]

def clear_database():
    if streamlit_available:
        conn = sqlite3.connect("moderation.db", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM moderation_log")
        conn.commit()
        conn.close()

if not streamlit_available:
    print("Streamlit is not available. Please install it with: pip install streamlit==1.38.0")
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

    # Clear the database only once when the app starts or refreshes
    if "db_cleared" not in st.session_state:
        clear_database()
        st.session_state["db_cleared"] = True

    result = None  # Store result globally for use in dashboard

    # Load moderation model (custom or fallback)
    @st.cache_resource
    def load_moderation_model():
        return load_model()
    
    model = load_moderation_model()
    

    # Severity and category estimation logic
    def interpret_results(preds,user_input):
        default_offense_types_file_path_setting = os.path.join('data', 'harm_categories.json')
        flag, category, action = preds
        if flag == 0:
            flagged = "Yes"
            reasoner = load_reasoner(user_input, default_offense_types_file_path_setting)
            response = reasoner.generate_response()
            category_match = re.search(r"\*\*Category:\*\*\s*(.+)", response)
            reason_match = re.search(r"\*\*Reason:\*\*\s*([\s\S]+)", response)
            category = category_match.group(1).strip() if category_match else ""
            reason = reason_match.group(1).strip() if reason_match else ""
            explanation = action + ". " + reason
            severity = 5
        else:
            flagged = "No"
            explanation = action
            severity = 0 if flag == 2 else 3
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
                    result = interpret_results(predictions, user_input)
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

        df = pd.read_sql_query("SELECT id, message AS Message, severity AS Severity, category AS Category, flagged AS Flagged, explanation AS Explanation, created_at AS Timestamp FROM moderation_log ORDER BY created_at DESC", conn)

        if not df.empty:
            
            # This section displays the moderation log in a table format and provides functionality 
            # to delete flagged content. The log includes details such as message, severity, category, 
            # flagged status, explanation, and timestamp. Flagged content is highlighted separately, 
            # and users can expand each flagged entry to view details and delete specific records 
            # from the database if necessary.
            
            display_df = df.drop(columns=["id"]) if "id" in df.columns else df
            st.dataframe(display_df, use_container_width=True)
            st.subheader("Flagged Content")
            flagged_df = df[df['Flagged'] == "Yes"]
            if not flagged_df.empty:
                for idx, row in flagged_df.iterrows():
                    with st.expander(f"{row['Message'][:50]}..."):
                        st.write(f"**Category**: {row['Category']}")
                        st.write(f"**Severity**: {row['Severity']}")
                        st.write(f"**Explanation**: {row['Explanation']}")

                        if st.button("Delete", key=f"delete_{row['id']}"):
                            cursor.execute("DELETE FROM moderation_log WHERE id = ?", (row['id'],))
                            conn.commit()
                            st.success("Record deleted.")
                            st.rerun()
            else:
                st.write("No content flagged for manual review.")
        else:
            st.info("No moderation data available yet. Submit a post in the User Input tab.")

    st.markdown("---")
    if custom_model_available:
        # st.caption("Note: This demo uses a custom moderation model from 'moderation.py'.")
        st.caption("Note: This demo uses fine-tuning `twitter-roberta` and `Gemma3` pipeline from 'model_config.py'.") 
        st.caption("This demo is designed by Group 18 of Applying Data Science, University of Manchester.\n")
        
    else:
        st.caption("Note: This demo uses the `toxic-bert` pipeline from HuggingFace Transformers.")