import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary resources (comment out if already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
  """
  This function preprocesses the text for training.
  Args:
      text: String containing the text data.
  Returns:
      A list of preprocessed tokens.
  """
  # Tokenization
  tokens = nltk.word_tokenize(text.lower())  # Lowercase and tokenize

  # Stop word removal
  stop_words = set(stopwords.words('english'))
  tokens = [word for word in tokens if word not in stop_words]

  # Stemming (optional - Experiment with stemming vs lemmatization)
  stemmer = PorterStemmer()
  tokens = [stemmer.stem(word) for word in tokens]

  return tokens

# Read the Bhagavad Gita text file
with open("Geeta.txt", "r") as f:
  bhagavad_gita_text = f.read()

# Preprocess the text
preprocessed_text = preprocess_text(bhagavad_gita_text)

# Install spaCy (if not already installed)
# pip install spacy
import spacy

# Load a spaCy model for English language processing
nlp = spacy.load("en_core_web_sm")

def extractive_qa(question, text):
  """
  This function attempts to answer a question by extracting relevant phrases from the text.
  Args:
      question: The user's question.
      text: The text to search for answers (Bhagavad Gita text).
  Returns:
      A potential answer extracted from the text (or None if not found).
  """
  doc = nlp(text)
  doc_question = nlp(question)

  # Identify named entities and noun phrases in the question that might be relevant for searching the text
  answer_candidates = []
  for ent in doc_question.ents:
      answer_candidates.append(ent.text)
  for chunk in doc_question.noun_chunks:
      answer_candidates.append(chunk.text)

  # Search for the answer candidates within the text and return the first match
  for candidate in answer_candidates:
      if candidate in text:
          return candidate
  return None

# Use extractive_qa to generate some question-answer pairs from the Bhagavad Gita text
qa_pairs = []
for question in ["What is karma?", "Who is Arjuna?"]:
  answer = extractive_qa(question, bhagavad_gita_text)
  if answer:
      qa_pairs.append((question, answer))

# You can combine manually curated and extractive QA pairs for a richer dataset.
# Create a list of question-answer pairs manually (replace with your examples)
qa_pairs = [
  ("What is the central message of the Bhagavad Gita?", "The Bhagavad Gita emphasizes the importance of fulfilling one's duty without attachment to the outcome."),
  ("What is the role of Krishna in the Bhagavad Gita?", "Krishna acts as Arjuna's charioteer and divine guide, offering him philosophical knowledge and motivation to perform his duty."),
  # Add more question-answer pairs...
 ]
from transformers import BertTokenizer, TFBertForQuestionAnswering
from transformers import Adam  # Optimizer (optional)
from transformers import SquadLoss  # Loss function (optional)

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForQuestionAnswering.from_pretrained('bert-base-uncased')


# Function to prepare training data for transformers (omitted for brevity)
def prepare_training_data(qa_pairs, tokenizer):
  """
  This function prepares training data for a question answering model by converting
  question-answer pairs into model inputs (token IDs, attention masks).
  Args:
      qa_pairs: A list of tuples containing (question, answer) pairs.
      tokenizer: A pre-trained tokenizer (e.g., BertTokenizer).
  Returns:
      A list of dictionaries containing model inputs for each question-answer pair.
  """

  encoded_data = []
  for question, answer in qa_pairs:
    # Tokenize question and answer
    question_encoded = tokenizer(question, add_special_tokens=True, return_tensors="pt")
    answer_encoded = tokenizer(answer, add_special_tokens=True, return_tensors="pt")

    # Create attention masks to identify relevant parts of the sequence
    question_mask = question_encoded["attention_mask"]
    answer_mask = answer_encoded["attention_mask"]

    # Get start and end token IDs for the answer within the context (Bhagavad Gita text)
    # This step might require adjustments depending on how you represent the context.
    # Here, we assume the context is a single long string.
    context = "your_bhagavad_gita_text_here"  # Replace with your preprocessed Bhagavad Gita text
    context_encoded = tokenizer(context, add_special_tokens=True, return_tensors="pt")
    start_positions = answer_encoded.input_ids == tokenizer.convert_tokens_to_ids(tokenizer.sep_token)[0]  # Find first SEP token
    end_positions = answer_encoded.input_ids == tokenizer.convert_tokens_to_ids(tokenizer.eos_token)[0]  # Find first EOS token

    # Combine all data into a dictionary for each QA pair
    encoded_data.append({
      "question_input_ids": question_encoded["input_ids"],
      "question_attention_mask": question_mask,
      "answer_start_positions": start_positions,
      "answer_end_positions": end_positions,
    })

  return encoded_data

# Prepare training data
train_data = prepare_training_data(qa_pairs)

# Train the model
learning_rate = 2e-5
epochs = 3  # Adjust these values as needed
model.compile(optimizer=Adam(learning_rate=learning_rate), loss=SquadLoss())
model.fit(train_data, epochs=epochs)

# Save the trained model and tokenizer
model.save_pretrained("bhagavad_gita_qa_model")
tokenizer.save_pretrained("bhagavad_gita_qa_model")

print("Model and tokenizer saved successfully!")

import streamlit as st
from transformers import pipeline  # For loading the QA model

qa_pipeline = pipeline("question-answering", model="bhagavad_gita_qa_model")

st.title("Bhagavad Gita Question Answering")
st.subheader("Ask your questions about the Bhagavad Gita here.")

user_question = st.text_input("Enter your question:")

if user_question:
  # Pass the user question and Bhagavad Gita text to the loaded model
  answer = qa_pipeline(question=user_question, context=bhagavad_gita_text)
  st.write(f"Answer: {answer['answer']}")
  # Optionally, display additional information like confidence score
  # st.write(f"Confidence Score: {answer['score']}")

