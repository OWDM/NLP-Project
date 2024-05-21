import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tashaphyne.stemming import ArabicLightStemmer


# Load and preprocess data
data = pd.read_csv('C:\\Users\\Musae\\Documents\\GitHub-REPOs\\NLP-Project\\data\\ar_reviews_100k.tsv', sep='\t')
data = data[data['label'] != 'Mixed']


ArListem = ArabicLightStemmer()

arabic_stopwords = set(stopwords.words('arabic'))

def preprocess_text(text):
    tokens = word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word not in arabic_stopwords]
    stemmed_tokens = [ArListem.light_stem(word) for word in cleaned_tokens]
    return ' '.join(stemmed_tokens)

data['cleaned_text'] = data['text'].apply(preprocess_text)


positive_keywords = {'ممتاز', 'جيد', 'رائع', 'سعيد', 'لذيذ', 'مبهج', 'فرح', 'استثنائي', 'جميل', 'محبب', 'ممتع',
                     'مذهل', 'مريح', 'راض', 'أحب', 'استمتع', 'مفاجأة', 'مميز', 'لطيف', 'مرح', 'معجزة', 'ملهم', 'أسعد',
                     'خيالي', 'مذهل', 'فريد', 'هائل', 'راقي', 'أنيق', 'بهجة', 'مفيد', 'قيمة', 'بسيط', 'ناجح', 'موفق',
                     'مشجع', 'ما شاء الله', 'ظريف', 'محبوب', "اوف",'مبهر', 'إيجابي', 'تفاؤل', 'إعجاب', 'ممتن', 'شجاع', 'آمن', 'مثالي'}

negative_keywords = {'سيء', 'مخيب', 'حزين', 'مؤلم', 'كريه', 'قبيح', 'فشل', 'محبط', 'بشع', 'فظيع', 'مزعج', 'مروع',
                     'أسوأ', 'كره', 'كارثة', 'رعب', 'كئيب', 'مزعزع', 'اكتئاب', 'بائس', 'معقد', 'إحباط', 'تعب', 'مضجر',
                     'ممل', 'فضيحة', 'سلبي', 'كاذب', 'احتيال', 'احراج', 'بشع', 'تعيس', 'مستاء', 'مشؤوم',
                     'عداء', 'مزري', 'عنيف', 'ضعيف', 'متشائم', 'غاضب', 'اهب', 'مروع'}


def classify_review(text):
    words = set(text.split())
    scores = {'Positive': 0, 'Negative': 0}
    
    for word in words:
        if word in positive_keywords:
            scores['Positive'] += 1
        if word in negative_keywords:
            scores['Negative'] += 1

    if scores['Positive'] > scores['Negative']:
        return "Positive"
    elif scores['Negative'] > scores['Positive']:
        return "Negative"
    return "Unclassified"  



data['predicted_label'] = data['cleaned_text'].apply(classify_review)

accuracy = (data['predicted_label'] == data['label']).mean()
print(f"\nAccuracy: {accuracy:.2%}")
