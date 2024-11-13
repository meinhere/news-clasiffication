from flask import Flask, render_template, url_for, request

from preprocess import *
import pickle


# Load model
lr_model = pickle.load(open('model/lr_model.sav', 'rb'))
tfidf_model = pickle.load(open('model/tfidf_vectorizer.sav', 'rb'))

labels_encode = {
        1: "OTOMOTIF",
        0: "MONEY",
    }

def prediksi_berita_cnn(link_news):
    
    news = get_news(link_news)
    news['cleaned_text'] = preprocess_text(news['isi'])

    tfidf = model_tf_idf(news['cleaned_text'], tfidf_model)

    prediction = lr_model.predict(tfidf)

    return f'{labels_encode[prediction[0]]}', news['judul'][0], news['url'][0]



def prediksi_berita_public(link_news):
    
    news = get_news_public(link_news)
    news['cleaned_text'] = preprocess_text(news['isi'])

    tfidf = model_tf_idf(news['cleaned_text'], tfidf_model)

    prediction = lr_model.predict(tfidf)

    return f'{labels_encode[prediction[0]]}', news['judul'][0], news['url'][0]

 

# instance of flask application
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        url_kompas = request.form.get('url_kompas')
        print(f"=================================== {url_kompas}")
        if ("kompas.com" not in url_kompas or 
                            ("money" not in url_kompas and "otomotif" not in url_kompas)):
            return render_template("index.html", prediksi="URL tidak valid")
        
        else: 
            prediksi, judul, url = prediksi_berita_cnn(url_kompas)
            return render_template("index.html", prediksi=prediksi, judul=judul, url=url)
    else:
        return render_template("index.html")

@app.route("/about/")
def about():
    return render_template("about.html")

@app.route("/prediksi_berita", methods=['POST', 'GET'])
def prediksi_berita():
    if request.method == 'POST':
        url_prediksi = request.form.get('url_prediksi')

        if request.method == 'POST' and url_prediksi != '': 
            prediksi, judul, url = prediksi_berita_public(url_prediksi)
            return render_template("public.html", prediksi=prediksi, judul=judul, url=url)
    else:
        return render_template("public.html")

# @app.route("/public", methods=['POST', 'GET'])
# def public():
#     if request.method == 'POST':
#         url_public = request.form.get('url_public')

#         if request.method == 'POST' and url_public != '': 
#             prediksi, judul, url = prediksi_berita_public(url_public)
#             return render_template("public.html", prediksi=prediksi, judul=judul, url=url)
#     else:
#         return render_template("public.html")

@app.route("/ringkasan-berita", methods=['POST', 'GET'])
def ringkasan_berita():
    if request.method == 'POST':
        url = request.form.get('ringkasan')
        ctrl = request.form.get('centrality')
        if request.method == 'POST' and url != '': 
            rksbrt, judul, url = ringkas_berita(url, ctrl)
            return render_template("ringkasan-berita.html", ringkasan=rksbrt, judul=judul, url=url, centrality=ctrl)
    else:
        return render_template("ringkasan-berita.html")

@app.route("/ringkasan-text", methods=['POST', 'GET'])
def ringkasan_text():
    if request.method == 'POST':
        text = request.form.get('ringkasan')
        ctrl = request.form.get('centrality')
        if request.method == 'POST' and text != '': 
            rksbrt = ringkas_text(text, ctrl)
            return render_template("ringkasan-text.html", ringkasan=rksbrt, centrality=ctrl)
    else:
        return render_template("ringkasan-text.html")


if __name__ == '__main__':
    app.run(debug=True)