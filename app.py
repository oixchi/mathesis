# Import streamlit for app dev
import streamlit as st

# Create centered main title 
st.title('Ask me a question 👩‍💻')

# Chat message storage
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message['content'])

prompt = st.chat_input("Input your prompt here")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    response="Hallo Ketli🐟"
    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role':'assistant', 'content':response})

