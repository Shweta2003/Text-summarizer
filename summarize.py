import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the saved model and tokenizer
@st.cache_resource()  # Cache model to avoid reloading on every interaction
def load_model():
    model_path = "saved_t5_model"  # Path to extracted model folder
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return tokenizer, model

# Summarization function
def summarize_text(text, tokenizer, model, max_length=150):
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=max_length, min_length=50, length_penalty=2.0, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit UI
st.title("Text Summarization App")
st.write("Upload a text file and generate a summary using a pre-trained T5 model.")

tokenizer, model = load_model()  # Load model once

uploaded_file = st.file_uploader("Upload a text document", type=["txt"])
if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    st.subheader("Original Text")
    st.text_area("", text, height=200)

    if st.button("Generate Summary"):
        summary = summarize_text(text, tokenizer, model)
        st.subheader("Summarized Text")
        st.write(summary)
