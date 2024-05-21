import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('C:\\Users\\Musae\\Documents\\GitHub-REPOs\\NLP-Project\\data\\ar_reviews_100k.tsv', sep='\t')
df.columns = ['label', 'text']

df = df[df['label'] != 'Mixed']
df = df.drop_duplicates()


nltk.download('stopwords')
stop_words = set(stopwords.words('arabic'))

def clean_text(text):
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])


df['cleaned_text'] = df['text'].apply(clean_text)



x_train, x_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

# Train and evaluate 
model = LogisticRegression(max_iter=1000)
model.fit(x_train_tfidf, y_train)
y_pred = model.predict(x_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Logistic Regression Accuracy: {accuracy:.4f}")
print(f"Logistic Regression Confusion Matrix:\n{conf_matrix}")
