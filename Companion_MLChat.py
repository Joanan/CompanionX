import streamlit as st
import joblib 
from xgboost import XGBClassifier

bow_vectorizer_lem = joblib.load('bow_vectorizer_lem.pkl')
xgb_bow_lem = joblib.load('depression_detection_model.pkl')

# Streamlit Page Configuration
st.set_page_config(page_title="NICE Counseling Assistant", page_icon="🤖")
st.title("🤖 NICE Counseling Assistant")
st.write(
    "Welcome! This AI assistant is here to provide empathetic and thoughtful counseling, adhering to the NICE guidelines."
)


# Submit Data Callback
def submit_data():
    """Process user input and retrieve assistant response."""
    user_input = st.session_state["user_input"]
    st.session_state["messages"].append({"role": "user", "content": user_input})

    if user_input and len(st.session_state["messages"])==2:
        # Append the user's message to the chat history 
        st.session_state["messages"].append({"role": "assistant", "content": "Please give me a few more details into how you feel?"})
        
    if user_input and len(st.session_state["messages"])==4:
        #st.session_state["messages"].append({"role": "user", "content": user_input})
        # Retrieve relevant context from NICE guidelines
        with st.spinner("Thinking..."):
            new_text=[]
            new_texts=[]
            for msg in st.session_state["messages"]:
                if msg["role"] == "user":
                    new_text.append(msg["content"])

            new_texts.append(". ".join([msg for msg in new_text]))    
            #new_texts.append(". ".join([msg["content"] for msg in st.session_state["messages"] and msg["role"] == "user"]))
            X_new_bow = bow_vectorizer_lem.transform(new_texts)
            y_pred_new = xgb_bow_lem.predict(X_new_bow)
           
            if y_pred_new == 1 :
              st.session_state["messages"].append({"role": "assistant", "content": "It must be tough for you, rest assured we are here to assist, i will refere you to our Virtual Counselor in a second"})
            
            else:
              st.session_state["messages"].append({"role": "assistant", "content": "It seams you are not showing any signs of depression, you can talk to a professional if you feel differrent, or get some other help"})

        # Clear the input field
        # Display chat messages

st.session_state["user_input"] = ""


    # Initialize chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I'm here to help.let us start our session by asking, how are you today?"}]
    #st.chat_message("assistant").write("Hello! I'm here to help.let us start our session by asking, how are you today?")

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
# Input box and submit button
#user_input = st.chat_input("Share how you're feeling...")
#st.text_input
st.chat_input(
    "You:", placeholder="Share how you're feeling...", key="user_input", on_change=submit_data
)

# Sidebar Reset Button
if st.sidebar.button("Reset Chat"):
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How are you feeling today?"}]
    #st.session_state["user_input"] = ""
    st.experimental_rerun()
