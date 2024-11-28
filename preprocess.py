import requests as req
from bs4 import BeautifulSoup as bs

# Library untuk data manipulation & visualisasi
import pandas as pd
import networkx as nx
import re

# Library untuk text preprocessing
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('punkt')

# Library untuk plotting
import matplotlib.pyplot as plt
import os

# Library untuk text vectorization & Similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import io
import base64


# Cleaning text Berita
def clean_text(text: str=None) -> str:
	"""
	Mmembersihkan text dari karakter-karakter yang tidak diperlukan
	"""
	text = text.lower() # Mengubah menjadi lowercase
	text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))', ' ', text) # Menghapus https* and www*
	text = re.sub(r'@[^\s]+', ' ', text) # Menghapus username
	text = re.sub(r'[\s]+', ' ', text) # Menghapus tambahan spasi
	text = re.sub(r'#([^\s]+)', ' ', text) # Menghapus hashtags
	text = re.sub(r"[^a-zA-Z0-9 / :\./(){}\[\]\|@,;_]", "", text) # Menghapus tanda baca
	text = text.encode('ascii','ignore').decode('utf-8') # Menghapus ASCII dan unicode
	text = re.sub(r'[^\x00-\x7f]',r'', text)
	text = text.replace('\n','') #Menghapus baris baru
	text = text.strip()
	# Menghapus whitespace
	text = re.sub(r"//t",r"\t", text)
	text = re.sub(r"( )\1+",r"\1", text)
	text = re.sub(r"(\n)\1+",r"\1", text)
	text = re.sub(r"(\r)\1+",r"\1", text)
	text = re.sub(r"(\t)\1+",r"\1", text)
	return text


def stemming_indo(text: str) -> str:
	"""
	Menstemming kata atau lemmisasi kata dalam bahasa Indonesia
	"""
	factory = StemmerFactory()
	stemmer = factory.create_stemmer()
	text = ' '.join(stemmer.stem(word) for word in text)
	return text

def clean_stopword(tokens: list) -> list:
	"""
	Membersihkan kata yang merupakan stopword
	"""
	listStopword =  set(stopwords.words('indonesian'))
	removed = []
	for t in tokens:
		if t not in listStopword:
			removed.append(t)
	return removed

def preprocess_text(content):
	"""
	Memproses text berita, membersihkan text, memperbagus kata, dan menghilangkan stopword
	"""
	result = []
	for text in content:
		tokens = nltk.tokenize.word_tokenize(text)
		cleaned_stopword = clean_stopword(tokens)
		stemmed_text = stemming_indo(cleaned_stopword)
		result.append(stemmed_text)
	return result

def preprocess_text_ringkas(text):
	"""
	Memproses text berita untuk ringkasan
	"""
	result = ""
	cleaned_text = clean_text(text)
	tokens = nltk.tokenize.word_tokenize(cleaned_text)
	result = ' '.join(tokens)
	kalimat = nltk.sent_tokenize(result)
	return kalimat

# Scraping berita
def scrape_news(soup: str) -> dict:
	"""
	Mengambil informasi berita dari url
	"""
	berita = {}
	texts = []

	berita["judul"] = soup.title.text

	text_list = soup.find("div", class_="read__content")
	for text in text_list.find_all("p"):
		if 'para_caption' not in text.get('class', []):
			cleaned_text = clean_text(text.text)
			texts.append(cleaned_text)


	berita["tanggal"] = soup.find("div", class_="read__time")
	berita['tanggal'] = berita['tanggal'].text.split(",")[1].strip()

	berita["isi"] = "\n".join(texts)
	berita["kategori"] = soup.find("meta", attrs={'name': 'content_category'})['content'].upper()
	berita["url"] = soup.find("meta", attrs={'property': 'og:url'})['content']
	return berita

# Mengambil html dari url
def get_html(url: str) -> str:
	"""
	Mengambil html dari url
	"""
	try:
		response = req.get(url).text
		return bs(response, "html5lib")
	
	except Exception as e:
		print(e)
		return ""

def get_news(news_url: str) -> pd.DataFrame:
	"""
	Mengambil informasi dari isi berita yang ada pada url
	"""
	news = []

	result = scrape_news(get_html(news_url))
	news.append(result)

	df = pd.DataFrame.from_dict(news)

	return df

def model_tf_idf(data, _model):
	"""
	Membuat model TF-IDF dari data
	"""
	tfidf_matrix = _model.transform(data)
	feature_names = _model.get_feature_names_out()

	df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

	return df_tfidf

def model_svd(data, _model):
	"""
	Membuat model SVD dari data
	"""
	svd_matrix = _model.transform(data)
	n = svd_matrix.shape[1]
	df_svd = pd.DataFrame(svd_matrix, columns=[f"fitur_{i}" for i in range(n)])

	return df_svd

def network_graph(tfidf_matrix, tfidf_vectorizer, threshold=0.05):
	"""
	Membuat adjacency matrix dari similarity matrix
	"""
	data_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
	cosine_sim = pd.DataFrame(cosine_similarity(data_tfidf))

	data_cossine_binary = cosine_sim.map(lambda x: 1 if x > threshold else 0)
	graph = nx.from_pandas_adjacency(data_cossine_binary)

	return graph

def centrality(G, centrality_type="degree"):
	"""
	Opsi untuk menghitung nilai centrality dari graph
	- degree
	- eigenvector
	- betweenness
	- closeness
	- pagerank
	"""
	if centrality_type == "degree":
		return nx.degree_centrality(G)
	elif centrality_type == "eigenvector":
		return nx.eigenvector_centrality(G)
	elif centrality_type == "betweenness":
		return nx.betweenness_centrality(G)
	elif centrality_type == "closeness":
		return nx.closeness_centrality(G)
	elif centrality_type == "pagerank":
		return nx.pagerank(G)
	else:
		raise ValueError(f"Unknown centrality type: {centrality_type}")

def sorted_result(node, kalimat, total=3):
	"""
	Mengurutkan hasil berdasarkan nilai centrality
	"""
	closeness_centrality = sorted(node.items(), key=lambda x: x[1], reverse=True)

	ringkasan = ""
	for node, closeness_preprocessing in closeness_centrality[:total]:
		top_sentence = kalimat[node]
		ringkasan += top_sentence + " "

	return ringkasan

def plot_graph(G):
		"""
		Plotting graph dan mengembalikan HTML untuk ditampilkan
		"""
		plt.figure(figsize=(12, 8))
		pos = nx.spring_layout(G)
		nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=15, font_color='black', font_weight='bold')
		plt.title("Graph Visualization")

		nx.draw_networkx_labels(G, pos)

		# Save plot to a bytes buffer
		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		buf.seek(0)

		# Encode the bytes to base64
		plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
		buf.close()

		# Create HTML to display the image
		html = f'data:image/png;base64,{plot_base64}'
		return html
	

def ringkas_berita(url: str, ctrl: str) -> str:
	"""
	Mengambil ringkasan berita dari url
	"""
	df = get_news(url)
	link = df['url'][0]
	judul = df['judul'][0]
	preprocessed = preprocess_text_ringkas(df['isi'][0])
	tfidf_vectorizer = TfidfVectorizer()
	tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed)

	G = network_graph(tfidf_matrix, tfidf_vectorizer)
	C = centrality(G, ctrl)
	
	result = sorted_result(C, preprocessed)
	image = plot_graph(G)

	return result, judul, link, image

def ringkas_text(text: str, ctrl: str) -> str:
	"""
	Mengambil ringkasan dari text
	"""
	preprocessed = preprocess_text_ringkas(text)
	tfidf_vectorizer = TfidfVectorizer()
	tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed)
	
	G = network_graph(tfidf_matrix, tfidf_vectorizer)
	C = centrality(G, ctrl)
	
	result = sorted_result(C, preprocessed)
	image = plot_graph(G)

	return result, image