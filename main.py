import pandas as pd
import re
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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



def main():
    train_df, val_df, test_df = load_datasets()
    print("Train samples:", len(train_df))
    print("Validation samples:", len(val_df))
    print("Test samples:", len(test_df))
    print(train_df.head())
    train_df= preprocess_df(train_df)
    print(train_df.head())

if __name__ == '__main__':
    main()
