# alat untuk crawling
from urllib.request import urlopen
from bs4 import BeautifulSoup

# praproses
import pandas as pd
import re
import pickle
import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# library untuk proses pembentukan vsm
from sklearn.feature_extraction.text import TfidfVectorizer

# library untuk proses modeling
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# library untuk evaluasi model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# monitoring
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt
import streamlit as st

# ---------- Defining Function -----------
# fungsi untuk mengambil link yang akan dilakukan crawling
def extract_urls(url):
    html = urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')

    urls = soup.find_all("a", {"class": "paging__link"})
    urls = [url.get('href') for url in urls]

    return urls

# fungsi untuk mengambil isi dari berita
def get_content(url):
    html = urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')

    div = soup.find("div", {"class": "read__content"})
    paragraf = div.find_all("p")

    content = ''
    for p in paragraf:
        content += p.text

    return content

# fungsi utama crawling
def crawl(link = "https://indeks.kompas.com", max_money = 1, max_otomotif = 1, allow_category = ["OTOMOTIF", "MONEY"], is_train = True, title_old = []):
    # inisialisasi variabel penampung hasil berita
    news_data = []

    # inisialisasi persiapan untuk crawling berita
    last_url = extract_urls(link).pop()
    page = last_url.split('=').pop() # jumlah halaman secara otomatis
    # page = 1 # jumlah halaman secara manual

    # persiapan link yang akan dilakukan crawling
    urls = [link + '/?page=' + str(a) for a in range(1, int(page) + 1)]
    count_money = 0
    count_otomotif = 0

    # menelusuri semua link yang telah ditentukan
    for idx, url in enumerate(urls):
        if (len(news_data) == max_money + max_otomotif) :
            break

        html = urlopen(url).read()
        soup = BeautifulSoup(html, 'html.parser')

        # mengambil data yang diperlukan pada struktur html
        links       = soup.find_all("a", {"class": "article-link"})
        titles      = soup.find_all("h2", {"class": "articleTitle"})
        dates       = soup.find_all("div", {"class": "articlePost-date"})
        categories  = soup.find_all("div", {"class": "articlePost-subtitle"})

        news_per_page = len(links) # berita artikel yang ditampilkan

        # memasukkan data ke dalam list
        for elem in tqdm(range(news_per_page), desc=f"Crawling page {idx+1}"):
            news = {}
            category = categories[elem].text
            title = titles[elem].text

            if (category in allow_category):
                if (is_train):
                    cond = (category == "MONEY" and count_money < max_money) or (category == "OTOMOTIF" and count_otomotif < max_otomotif)
                else:
                    cond = (category == "MONEY" and count_money < max_money) or (category == "OTOMOTIF" and count_otomotif < max_otomotif) and title not in title_old


                if (cond):
                    news['No'] = len(news_data) + 1
                    news['Judul Berita']     = title
                    news['Isi Berita']       = get_content(links[elem].get("href"))
                    news['Tanggal Berita']   = dates[elem].text
                    news['Kategori Berita']  = category
                    news_data.append(news)

                if (category == "MONEY"):
                    count_money += 1
                else:
                    count_otomotif += 1

        print(f"=======> Money: {count_money} | Otomotif: {count_otomotif} | Total: {count_money + count_otomotif}")

    return news_data

# Case Folding
def clean_lower(lwr):
    lwr = lwr.lower() # lowercase text
    return lwr

# Menghapus tanda baca, angka, dan simbol
def clean_punct(text):
    clean_spcl = re.compile('[/(){}\[\]\|@,;_]')
    clean_symbol = re.compile('[^0-9a-z]')
    clean_number = re.compile('[0-9]')
    text = clean_spcl.sub('', text)
    text = clean_symbol.sub(' ', text)
    text = clean_number.sub('', text)
    return text

# Menghaps double atau lebih whitespace
def _normalize_whitespace(text):
    corrected = str(text)
    corrected = re.sub(r"//t",r"\t", corrected)
    corrected = re.sub(r"( )\1+",r"\1", corrected)
    corrected = re.sub(r"(\n)\1+",r"\1", corrected)
    corrected = re.sub(r"(\r)\1+",r"\1", corrected)
    corrected = re.sub(r"(\t)\1+",r"\1", corrected)
    return corrected.strip(" ")

# Menghapus stopwords
def clean_stopwords(text):
    stopword = set(stopwords.words('indonesian'))
    text = ' '.join(word for word in text.split() if word not in stopword) # hapus stopword dari kolom deskripsi
    return text

# Stemming with Sastrawi
def sastrawistemmer(text):
    factory = StemmerFactory()
    ste = factory.create_stemmer()
    text = ' '.join(ste.stem(word) for word in tqdm(text.split()) if word in text)
    return text


# ------------------ Page Navigation ---------------------
def main():
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["Home", "Crawling Berita", "Praproses Teks", "Pembuatan VSM", "Modeling", "Testing"])

    if page == "Home":
        show_home()
    elif page == "Crawling Berita":
        show_crawling()
    elif page == "Praproses Teks":
        show_preprocessing()
    elif page == "Pembuatan VSM":
        show_creating()
    elif page == "Modeling":
        show_modeling()
    elif page == "Testing":
        show_testing()

def show_home():
    st.title("Prediksi Berita Online menggunakan Algoritma Logistic Regression")

    # Explain what is Random Forest
    st.header("Apa itu Logistic Regression?")
    st.write("Logistic Regression adalah salah satu algoritma yang digunakan dalam machine learning untuk masalah klasifikasi. Tujuan utama dari Logistic Regression adalah untuk memprediksi probabilitas atau kemungkinan terjadinya suatu kejadian berdasarkan variabel input yang diberikan.")

    # Explain the purpose of this website
    st.header("Tujuan Website")
    st.write("Website ini bertujuan untuk memberikan pemahaman mengenai tahapan proses prediksi berita online menggunakan algoritma Logistic Regression.")

    # Explain the data
    st.header("Data")
    st.write("Data yang digunakan diambil dari website online yaitu KOMPAS. Data yang diambil berisi judul berita, kategori berita, tanggal upload, isi berita dan kategori berita")

    # Explain the process of Random Forest
    st.header("Tahapan Proses Random Forest")
    st.write("1. **Crawling Berita**")
    st.write("2. **Praproses Data**")
    st.write("3. **Pembuatan VSM**")
    st.write("4. **Modeling**")
    st.write("5. **Evaluation**")
    st.write("6. **Testing**")

def show_crawling():
    st.title("Crawling Berita")

    # Load data
    main_df = pd.read_csv("dataset/data_berita.csv", delimiter=",")

    st.subheader("Data yang didapat setelah proses crawling:")
    st.write(main_df)
    st.write(f"Jumlah baris: {main_df.shape[0]} - Jumlah Kolom: {main_df.shape[1]}")

def show_preprocessing():
    st.title("Preprocessing Data")

    # --------------- Load Data -----------------
    main_df = pd.read_csv("dataset/data_berita.csv", delimiter=",")

    st.write("### Data yang tersedia:")
    st.write(main_df.head())

    # --------------- Merubah Data ke Bentuk Lowercase -----------------
    st.write("### Merubah Data ke Bentuk Lowercase")
    main_df['lwr'] = main_df['Isi Berita'].apply(clean_lower)
    st.write(main_df['lwr'])

    # --------------- Menghapus Tanda Baca -----------------
    st.write("### Menghapus Tanda Baca")
    main_df['clean_punct'] = main_df['lwr'].apply(clean_punct)
    st.write(main_df['clean_punct'])

    # --------------- Menghapus Spasi Berlebih -----------------
    st.write("### Menghapus Spasi Berlebih")
    main_df['clean_double_ws'] = main_df['clean_punct'].apply(_normalize_whitespace)
    st.write(main_df['clean_double_ws'])

    # --------------- Menghapus Stopwords -----------------
    st.write("### Menghapus Stopwords")
    main_df['clean_sw'] = main_df['clean_double_ws'].apply(clean_stopwords)
    st.write(main_df['clean_sw'])

    # --------------- Stemming -----------------
    st.write("### Stemming")
    main_df = pd.read_csv('dataset/data_berita_praproses.csv', delimiter=',')
    st.write(main_df['desc_clean_stem'])

    st.session_state['preprocessed_data'] = main_df

def show_creating():
    st.title("Pembuatan VSM")

    if 'preprocessed_data' not in st.session_state:
        st.write("Silakan lakukan preprocessing data terlebih dahulu.")
        return
    
    # load data
    main_df = st.session_state['preprocessed_data']

    # Splitting data
    st.write("### Split Data Train dan Test")
    X = main_df['desc_clean_stem']
    y = main_df['Kategori Berita']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_df = pd.DataFrame({'desc_clean_stem': X_train, 'label': y_train})
    test_df = pd.DataFrame({'desc_clean_stem': X_test, 'label': y_test})

    st.write("Jumlah Data Train: ", train_df.shape[0])
    st.write("Jumlah Data Test: ", test_df.shape[0])

    # fitting words to tfidf
    st.write("### Ukuran Shape setelah TfidfVectorizer")
    vectorizer = TfidfVectorizer()
    corpus = train_df['desc_clean_stem'].tolist()
    tfidf = vectorizer.fit_transform(corpus)

    # creating vsm 
    st.write("### Data setelah pembentukan VSM")
    vocabulary = vectorizer.get_feature_names_out().tolist()
    train_tfidf_df = pd.DataFrame(tfidf.toarray(), columns=vocabulary)
    train_tfidf_df['label'] = train_df['label'].tolist()
    st.write("#### Data Train")
    st.write("Jumlah Baris :", train_tfidf_df.shape[0], " - Jumlah Kolom : ", train_tfidf_df.shape[1])
    st.write(train_tfidf_df.head())

    filename = 'model/tfidf_vectorizer.sav'
    vectorizer = pickle.load(open(filename, 'rb'))
    test = test_df['desc_clean_stem']
    vocabulary = vectorizer.get_feature_names_out().tolist()
    tfidf = vectorizer.transform(test)
    test_tfidf_df = pd.DataFrame(tfidf.toarray(), columns=vocabulary)
    test_tfidf_df['label'] = test_df['label'].tolist()
    st.write("#### Data Test")
    st.write("Jumlah Baris :", test_tfidf_df.shape[0], " - Jumlah Kolom : ", test_tfidf_df.shape[1])
    st.write(test_tfidf_df.head())

    # Encoding Label
    st.write("### Data setelah Encoding Label")
    label_encoder = preprocessing.LabelEncoder()
    train_tfidf_df['label']= label_encoder.fit_transform(train_tfidf_df['label'])
    st.write("#### Data Train")
    st.write(train_tfidf_df.head())

    test_tfidf_df['label']= label_encoder.fit_transform(test_tfidf_df['label'])
    st.write("#### Data Test")
    st.write(test_tfidf_df.head())

    st.session_state['train_tfidf_df'] = train_tfidf_df
    st.session_state['test_tfidf_df'] = test_tfidf_df

def show_modeling():
    st.title("Modeling")
    
    if 'train_tfidf_df' not in st.session_state and 'test_tfidf_df' not in st.session_state:
        st.write("Silakan lakukan proses pembuatan vsm data terlebih dahulu.")
        return
    
    # load data
    train_tfidf_df = st.session_state['train_tfidf_df']
    st.write("Data Train")
    st.write(train_tfidf_df.head())

    # data test
    test_tfidf_df = st.session_state['test_tfidf_df']
    st.write("Data Test")
    st.write(test_tfidf_df.head())

    # fit model untuk training
    X_train = train_tfidf_df.drop('label', axis=1)
    y_train = train_tfidf_df['label']

    X_test = test_tfidf_df.drop('label', axis=1)
    y_test = test_tfidf_df['label']

    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)

    # mencoba prediksi dari hasi fitting model
    st.write("### Perbandingan Hasil Prediksi dan Data Asli")
    y_pred = lr_model.predict(X_test)
    a = pd.DataFrame({'Actual value': y_test, 'Predicted value':y_pred})
    st.write(a.head())

    # Evaluasi model
    st.subheader("Akurasi")
    st.write(f"{accuracy_score(y_test, y_pred)*100:.2f}%")

    st.subheader("Classification Report")
    cr = classification_report(y_test, y_pred)
    st.text(cr)

    # Confusion matrix dan classification report
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=[f'Actual {i}' for i in range(len(cm))], columns=[f'Predicted {i}' for i in range(len(cm))])
    st.table(cm_df)

def show_testing():
    labels_encode = {
        1: ":blue[OTOMOTIF]",
        0: ":red[MONEY]",
    }

    st.title("Prediksi Kategori Berita Online")

    st.write("Masukan Link Berita dari website KOMPAS")
    link_news = st.text_input("Link Judul Berita [Kategori MONEY | OTOMOTIF]")
    
    check= st.button("Prediksi")

    if check and link_news != "":
        # create dataframe
        link_news = get_content(link_news)
        data = [{ "Isi Berita" : link_news }]
        test_df = pd.DataFrame(data)

        # preprocessing text
        test_df['lwr'] = test_df['Isi Berita'].apply(clean_lower)
        test_df['clean_punct'] = test_df['lwr'].apply(clean_punct)
        test_df['clean_double_ws'] = test_df['clean_punct'].apply(_normalize_whitespace)
        test_df['clean_sw'] = test_df['clean_double_ws'].apply(clean_stopwords)
        test_df['desc_clean_stem'] = test_df['clean_sw'].apply(sastrawistemmer)

        # creating vsm
        # Load the saved model from file
        filename = 'model/tfidf_vectorizer.sav'
        tfidf_vectorizer = pickle.load(open(filename, 'rb'))
        corpus = test_df['desc_clean_stem']
        tfidf = tfidf_vectorizer.transform(corpus)
        vocabulary = tfidf_vectorizer.get_feature_names_out().tolist()
        tfidf_df = pd.DataFrame(tfidf.toarray(), columns=vocabulary)
        # st.write(tfidf_df)
        
        # Predict Label
        # Load the saved model from file
        filename = 'model/lr_model.sav'
        lr_model = pickle.load(open(filename, 'rb'))
        prediction = lr_model.predict(tfidf_df)

        st.write(f"Berita termasuk Kategori: {labels_encode[prediction[0]]}")
    else:
        st.write("Silakan masukkan link berita terlebih dahulu")

if __name__ == "__main__":
    st.set_page_config(page_title="News Classification - Logistic Regression", page_icon="📑")
    main()