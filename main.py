import pandas as pd
import re
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

stop_words = set(stopwords.words('english'))
def load_datasets():
    train_df = pd.read_csv('/home/aitor/Escritorio/ClasificadorEmociones/datos/raw/train.txt', sep=';', names=['text', 'emotion'])
    val_df = pd.read_csv('/home/aitor/Escritorio/ClasificadorEmociones/datos/raw/val.txt', sep=';', names=['text', 'emotion'])
    test_df = pd.read_csv('/home/aitor/Escritorio/ClasificadorEmociones/datos/raw/test.txt', sep=';', names=['text', 'emotion'])
    return train_df, val_df, test_df

def preprocess(text):
    text=text.lower() #minusculas
    text = re.sub(r'[^a-z\s]', '', text) #quitar signos de puntuación y numeros
    tokens=word_tokenize(text)#tokenizar
    # Quitar stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]#quitar stopwoards
    return ' '.join(filtered_tokens)

def preprocess_df(df):
    df['text']=df['text'].apply(preprocess)
    return df

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    print("\n--- Evaluación en conjunto de validación ---")
    y_pred_val = clf.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, y_pred_val))
    print(classification_report(y_val, y_pred_val))

    print("\n--- Evaluación en conjunto de test ---")
    y_pred_test = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test))


def main():
    train_df, val_df, test_df = load_datasets()
    print("Train samples:", len(train_df))
    print("Validation samples:", len(val_df))
    print("Test samples:", len(test_df))
    print(train_df.head())
    train_df= preprocess_df(train_df)
    print(train_df.head())
    label_encoder = LabelEncoder()
    train_df['emotion'] = label_encoder.fit_transform(train_df['emotion'])
    val_df['emotion'] = label_encoder.transform(val_df['emotion'])
    test_df['emotion'] = label_encoder.transform(test_df['emotion'])
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['text']) #aprender el vocabulario y vectorizar.
    X_val = vectorizer.transform(val_df['text']) #usar el mismo vocab
    X_test = vectorizer.transform(test_df['text'])

    y_train = train_df['emotion']
    y_val = val_df['emotion']
    y_test = test_df['emotion']
    train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == '__main__':
    main()

