import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    # Load the BART model for summarization
    summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(device)
    
    # Load a DistilBERT model for question-answering
    question_answering = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
    return summarizer_tokenizer, summarizer_model, question_answering

# Summarize text
def get_summary(text, tokenizer, model):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(device)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=100, early_stopping=True)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output

# Capitalize text of summary
def capitalize_text(source):
    stop, output = '.', ''
    i, j = 0, 0
    for i, w in enumerate(source):
        if w == stop:
            sen = source[j:i+1].capitalize()
            j = i+2
            output += sen
    output += source[j:].capitalize()
    return output

# Translate text
def translate_text(text, tokenizer, model, option):
    prompt = f"translate English to {option}: " + text
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    translation_ids = model.generate(inputs['input_ids'], num_beams=4, no_repeat_ngram_size=2, min_length=30, max_length=100, early_stopping=True)
    output = tokenizer.decode(translation_ids[0], skip_special_tokens=True)
    return output

# Generate questions and answers
def get_questions(text, question_answering):
    questions = ["What is the main point?", "Can you summarize the topic?", "What is this text about?"]
    st.markdown('**Question Answers:**')
    for question in questions:
        answer = question_answering(question=question, context=text)
        st.markdown(f"**{question}**")
        st.text(answer['answer'])

def main():
    st.title('HelpMeRead')
    message = st.text_area("Enter Text:")
    option = st.selectbox("Select language:", ('German', 'French', 'Romanian', 'Arabic', 'Urdu', 'Persian'))
    Translate = st.button("Translate")
    Summarize = st.button("Summarize")
    QAs = st.button("QAs")
    
    # Load models only once
    summarizer_tokenizer, summarizer_model, question_answering = load_models()

    if Summarize:
        result = get_summary(message, summarizer_tokenizer, summarizer_model)
        st.markdown('**Summary:**')
        st.write(capitalize_text(result))

    if Translate:
        result = translate_text(message, summarizer_tokenizer, summarizer_model, option)
        st.markdown('**Translation:**')
        st.write(result)

    if QAs:
        get_questions(message, question_answering)

if __name__ == "__main__":
    main()
