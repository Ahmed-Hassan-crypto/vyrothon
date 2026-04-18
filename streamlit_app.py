import streamlit as st
from inference import run

# Page config
st.set_page_config(
    page_title="Tool-Calling Assistant",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Title
st.title("🤖 Tool-Calling Assistant")
st.markdown("Ask about: **weather**, **calendar**, **convert**, **currency**, **SQL queries**")

# Display chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Type your message..."):
    # Add user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    response = run(prompt, st.session_state.history)
    
    # Add to history
    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.history.append({"role": "assistant", "content": response})
    
    # Display response
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Rerun to update UI
    st.rerun()

# Sidebar with examples
with st.sidebar:
    st.header("📌 Examples")
    
    examples = [
        "What's the weather in London?",
        "Convert 100 meters to feet",
        "Show my calendar for 2026-05-01",
        "Convert 50 USD to EUR",
        "List all users",
        "Hello, how are you?",
    ]
    
    for ex in examples:
        if st.button(ex, use_container_width=True):
            # Add example to input and get response
            response = run(ex, st.session_state.history)
            st.session_state.history.append({"role": "user", "content": ex})
            st.session_state.history.append({"role": "assistant", "content": response})
            st.rerun()
    
    st.divider()
    
    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<center>Fine-tuned Qwen2.5-0.5B | Output: JSON in &lt;tool_call&gt; tags</center>",
    unsafe_allow_html=True
)