# P4-Implementation-of-ChatBot-by-NLP
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np

# Download the NLTK punkt tokenizer
nltk.download("punkt")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define chatbot responses and intents
responses = {
    "greeting": "Hello! How can I assist you today?",
    "name_query": "I'm a chatbot created to help you. What's your name?",
    "farewell": "Goodbye! Have a nice day!",
    "default": "I'm sorry, I didn't quite understand that. Could you elaborate?",
    "about_chatbot": "I'm an AI-powered chatbot designed to understand and respond to your queries."
}

# Training phrases for intent detection
training_phrases = {
    "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
    "name_query": ["what is your name", "who are you", "tell me your name"],
    "farewell": ["bye", "goodbye", "see you later", "exit"],
    "about_chatbot": ["what can you do", "tell me about yourself", "who built you"],
}

# Flatten training phrases and map them to intents
corpus = []
intent_mapping = []

for intent, phrases in training_phrases.items():
    corpus.extend(phrases)
    intent_mapping.extend([intent] * len(phrases))

# TF-IDF Vectorizer for text matching
vectorizer = TfidfVectorizer().fit(corpus)


def predict_intent(user_input):
    """
    Predicts the intent of the user's input based on cosine similarity with training phrases.
    """
    user_vector = vectorizer.transform([user_input])
    corpus_vectors = vectorizer.transform(corpus)
    
    similarity_scores = cosine_similarity(user_vector, corpus_vectors).flatten()
    best_match_index = np.argmax(similarity_scores)
    
    if similarity_scores[best_match_index] > 0.5:  # Confidence threshold
        return intent_mapping[best_match_index]
    else:
        return "default"


def handle_ambiguity(user_input):
    """
    Handles ambiguity by analyzing sentence structure and entities using spaCy.
    """
    doc = nlp(user_input)
    if doc.ents:
        return f"I noticed you mentioned: {', '.join(ent.text for ent in doc.ents)}. Could you clarify?"
    return responses["default"]


def chatbot_response(user_input):
    """
    Generates a response based on the predicted intent or handles ambiguous inputs.
    """
    intent = predict_intent(user_input)
    
    if intent == "default":
        return handle_ambiguity(user_input)
    return responses[intent]


# Main chatbot loop
print("Chatbot: Hi! I am your chatbot. Type 'bye' to exit.")
while True:
    user_input = input("You: ").strip().lower()
    
    if user_input in ["bye", "exit"]:
        print("Chatbot:", responses["farewell"])
        break
    
    response = chatbot_response(user_input)
    print("Chatbot:", response)


 
  
   


# Happy Learning and Coding 😊
