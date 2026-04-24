import streamlit as st
import joblib 
from xgboost import XGBClassifier

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="NICE Counseling Assistant", page_icon="🧠")

st.title("🧠 NICE Counseling Assistant")
st.write("This AI assistant provides empathetic and thoughtful counseling support.")

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.header("Controls")

    if st.button("Reset Chat"):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! How are you feeling today?"}
        ]
        st.rerun()

# ------------------ LOAD MODEL ------------------
bow_vectorizer_lem = joblib.load('bow_vectorizer_lem.pkl')
xgb_bow_lem = joblib.load('depression_detection_model.pkl')

# ------------------ SESSION STATE ------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! How are you feeling today?"}
    ]

# ------------------ DISPLAY CHAT ------------------
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# ------------------ INPUT ------------------
user_input = st.chat_input("Share how you're feeling...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Process input
    texts = [user_input]

    X_new_bow = bow_vectorizer_lem.transform(texts)
    y_pred_new = xgb_bow_lem.predict(X_new_bow)

    if y_pred_new[0] == 1:
        response = "It must be tough for you. We're here to support you."
    else:
        response = "It seems you're not showing strong signs of distress."

    st.session_state["messages"].append({"role": "assistant", "content": response})

    st.rerun()
