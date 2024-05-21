import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tashaphyne.stemming import ArabicLightStemmer
# Load and preprocess data
data = pd.read_csv('C:\\Users\\Musae\\Documents\\GitHub-REPOs\\NLP-Project\\data\\ar_reviews_100k.tsv', sep='\t')
data = data[data['label'] != 'Mixed']
# Initialize stemmer
ArListem = ArabicLightStemmer()

# Normalization function
def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

# Load stopwords and punctuations
arabic_stopwords = set(stopwords.words('arabic'))
arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''

# Preprocess text function
def preprocess_text(text):
    text = normalize_arabic(text)
    tokens = word_tokenize(text)
    cleaned_tokens = [word.lower() for word in tokens if word not in arabic_punctuations and word not in arabic_stopwords]
    stemmed_tokens = [ArListem.light_stem(word) or ArListem.get_root() for word in cleaned_tokens]
    return ' '.join(stemmed_tokens)

# Apply preprocessing
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Define keywords for classification
positive_keywords = {'ممتاز', 'جيد', 'رائع', 'سعيد', 'لذيذ', 'مبهج', 'فرح', 'استثنائي', 'جميل', 'محبب', 'ممتع',
                     'مذهل', 'مريح', 'راض', 'أحب', 'استمتع', 'مفاجأة', 'مميز', 'لطيف', 'مرح', 'معجزة', 'ملهم', 'أسعد',
                     'خيالي', 'مذهل', 'فريد', 'هائل', 'راقي', 'أنيق', 'بهجة', 'مفيد', 'قيمة', 'بسيط', 'ناجح', 'موفق',
                     'مشجع', 'ما شاء الله', 'ظريف', 'محبوب', 'مبهر', 'إيجابي', 'تفاؤل', 'إعجاب', 'ممتن', 'شجاع', 'آمن', 'مثالي'}

negative_keywords = {'سيء', 'مخيب', 'حزين', 'مؤلم', 'كريه', 'قبيح', 'فشل', 'محبط', 'بشع', 'فظيع', 'مزعج', 'مروع',
                     'أسوأ', 'كره', 'كارثة', 'رعب', 'كئيب', 'مزعزع', 'اكتئاب', 'بائس', 'معقد', 'إحباط', 'تعب', 'مضجر',
                     'ممل', 'فضيحة', 'سلبي', 'كاذب', 'فظاظة', 'احتيال', 'احراج', 'بشع', 'تعيس', 'مستاء', 'مروع', 'مشؤوم',
                     'عداء', 'مزري', 'عنيف', 'ضعيف', 'متشائم', 'غاضب', '', 'مروع'}


# Rule-based classification function
def classify_review(text):
    words = set(text.split())
    if words & positive_keywords and not words & negative_keywords:
        return "Positive"
    elif words & negative_keywords and not words & positive_keywords:
        return "Negative"
    return "Mixed"

def classify_review(text):
    words = set(text.split())
    if words & negative_keywords:
        return "Negative"
    elif words & positive_keywords:
        return "Positive"
    return "Unclassified"  # or some default classification if neither

# Apply classification
data['predicted_label'] = data['cleaned_text'].apply(classify_review)

# Evaluate classifier
accuracy = (data['predicted_label'] == data['label']).mean()
print(f"\nAccuracy: {accuracy:.2%}")
