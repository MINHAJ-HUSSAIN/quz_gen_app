import streamlit as st
from transformers import BertForQuestionAnswering, BertTokenizer
import random

# Load pre-trained BERT model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

def generate_question_answer(text):
    # Encode the input text using the tokenizer
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    
    # Get model output
    outputs = model(**inputs)
    
    # Get start and end token positions of the answer
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    
    # Find the most probable answer
    start_index = start_scores.argmax()
    end_index = end_scores.argmax() + 1
    answer_tokens = inputs['input_ids'][0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens)
    
    return answer

def generate_question_from_answer(context, answer):
    # Simple way to generate a question from an answer
    question_templates = [
        f"What is the meaning of '{answer}' in the context?",
        f"Can you explain the significance of '{answer}'?",
        f"What role does '{answer}' play in this text?",
        f"How is '{answer}' related to the main idea?",
    ]
    
    # Select a random question template
    question = random.choice(question_templates)
    
    return question

def generate_quiz(context):
    # Get the answer from the model
    answer = generate_question_answer(context)
    
    # Generate a quiz question based on the answer
    question = generate_question_from_answer(context, answer)
    
    return question, answer

# Streamlit app
st.title("AI-Based Quiz Generator")
st.write("Enter a text (article or document), and the model will generate quiz questions based on it.")

# Input text area
context = st.text_area("Enter your text here:")

# When the button is pressed
if st.button("Generate Quiz"):
    if context:
        # Generate the quiz question and answer
        question, answer = generate_quiz(context)
        
        # Display the generated quiz
        st.subheader("Generated Question:")
        st.write(question)
        
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.write("Please enter some text to generate a quiz.")
