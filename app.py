# Import streamlit for app dev
import streamlit as st

# Create centered main title 
st.title('Ask me a question 👩‍💻')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
   response = "Hallo Ketli 🐟"
   st.write(response)