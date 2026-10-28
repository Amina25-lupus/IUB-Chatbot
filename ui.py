import streamlit as st
from app import get_qa_chain, get_response

# Page configuration
st.set_page_config(page_title="IUB AI Assistant", page_icon="🎓")
st.title("🎓 IUB Student Support Bot")
st.markdown("---")

# Initialize the chain only once (caching for speed)
if 'qa_chain' not in st.session_state:
    with st.spinner("Loading University Database..."):
        st.session_state.qa_chain = get_qa_chain()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me anything about IUB..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            full_response = get_response(prompt, st.session_state.qa_chain)
            st.markdown(full_response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})