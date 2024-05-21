import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv('C:\\Users\\Musae\\Documents\\GitHub-REPOs\\NLP-Project\\data\\ar_reviews_100k.tsv', sep='\t')
df.columns = ['label', 'text']

# Filter out mixed labels and remove duplicates
df = df[df['label'] != 'Mixed'].drop_duplicates()

# Download and prepare stopwords
nltk.download('stopwords')
ar_stopwords = set(stopwords.words('arabic'))

# Clean text directly in the DataFrame
df['cleaned_text'] = df['text'].apply(lambda text: ' '.join(word for word in word_tokenize(text) if word not in ar_stopwords))

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.3, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(x_train_tfidf, y_train)
y_pred = model.predict(x_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print("Logistic Regression Confusion Matrix:\n", conf_matrix)
