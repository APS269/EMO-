import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from langdetect import detect
import pickle
from converter import hinglish_to_english

# Download NLTK stopwords (if not already downloaded)
# nltk.download('stopwords')

data = pd.read_csv('tweet_emotions.csv')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english')])
    return text

data['cleaned_text'] = data['content'].apply(clean_text)

X = data['cleaned_text']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train_tfidf, y_train)

# Save the trained model and vectorizer using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(svm_classifier, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

# Rest of your code...

# Load the trained model and vectorizer in the analyze_emotion function
def analyze_emotion(text):
    # Detect the language of the input text
    detected_language = detect(text)
    
    if detected_language != 'en':
        text = hinglish_to_english(text)
        
    # Load the trained model and vectorizer
    with open('model.pkl', 'rb') as model_file:
        loaded_svm_classifier = pickle.load(model_file)

    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        loaded_tfidf_vectorizer = pickle.load(vectorizer_file)

    cleaned_text = clean_text(text)
    tfidf_text = loaded_tfidf_vectorizer.transform([cleaned_text])
    emotion = loaded_svm_classifier.predict(tfidf_text)[0]
    return emotion

# Example usage

