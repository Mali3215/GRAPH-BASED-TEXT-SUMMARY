import sys
import gensim
from gensim.models import KeyedVectors
from rouge import Rouge
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QTextEdit, QDesktopWidget, QVBoxLayout, \
    QDialog, QLabel, QLineEdit, QHBoxLayout, QTextBrowser, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import networkx as nx
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import random
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import re

nlp = spacy.load("en_core_web_sm")

# NLTK stopwords ve punkt nesnelerinin yüklenmesi
nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))
punct = set(punctuation)

# vectorizer nesnesinin tanımlanması
vectorizer = TfidfVectorizer()

# bert modelin yüklenmesi
model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Word2Vec modelini yüklenmesi
model_word = KeyedVectors.load_word2vec_format("C:\\Users\\akdme\\Documents\\pythonProject1\\model.bin", binary=True)

# cümle skoru
sentence_scores = []

# cümleler arası benzerlik
similarity_between_sentences = []
# node sayısı
node_scores = []


def nodes_passing_the_threshold(sentence, sentences, threshold):
    count = 0
    for node in sentences:
        if node.text != sentence and sentence_similarity_word(node.text, sentence) > threshold:
            count += 1

    return count


def sentence_similarity(sentence1, sentence2):
    inputs = tokenizer.encode_plus(sentence1, sentence2, return_tensors='pt', add_special_tokens=True)

    with torch.no_grad():
        outputs = model(**inputs)[0]
        embeddings = outputs[-1]

    similarity = cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))[0][0]

    return similarity


def sentence_similarity_word(sentence1, sentence2):
    tokens1 = sentence1.lower().split()
    tokens2 = sentence2.lower().split()

    embeddings1 = [model_word.get_vector(word) for word in tokens1 if word in model_word.key_to_index]

    embeddings2 = [model_word.get_vector(word) for word in tokens2 if word in model_word.key_to_index]

    # Eğer cümlelerde kelime bulunamadıysa benzerlik 0 dönsün
    if not embeddings1 or not embeddings2:
        return 0.0

    mean_embedding1 = np.mean(embeddings1, axis=0)
    mean_embedding2 = np.mean(embeddings2, axis=0)

    # Cosine benzerliğini hesapla
    similarity = cosine_similarity(mean_embedding1.reshape(1, -1), mean_embedding2.reshape(1, -1))[0][0]

    return similarity


def calculate_sentence_score(sentence, title_keywords, document_keywords, threshold, choice, sentences):
    sentence_length = len(sentence.split())

    # P1: Cümledeki özel isim kontrolü
    named_entity_count = 0
    doc = nlp(sentence)
    for token in doc:
        if token.ent_type_ != "":
            named_entity_count += 1
    p1 = named_entity_count / sentence_length

    # P2: Cümledeki numerik veri kontrolü
    numerics = re.findall(r'\b(?:\d+[a-zA-Z]*\b|\b[a-zA-Z]*\d+)\b', sentence)  # Numerik ifadeleri tespit ettik
    numeric_count = len(numerics)
    p2 = numeric_count / sentence_length

    p3 = 0
    if choice == 0:  # her iki algoritmada
        # P3: Cümle benzerliği threshold'unu geçen node'larını bul
        threshold_nodes = 0
        for node in sentences:
            if node.text != sentence and ((sentence_similarity(sentence, node.text)
                                           + sentence_similarity_word(sentence,
                                                                      node.text)) / 2) > threshold:
                threshold_nodes += 1

        p3 = threshold_nodes / len(similarity_between_sentences)

    elif choice == 1:  # bert embeding
        # P3: Cümle benzerliği threshold'unu geçen node'larını bul

        threshold_nodes = 0
        for node in sentences:
            if node.text != sentence and sentence_similarity(sentence, node.text) > threshold:
                threshold_nodes += 1

        p3 = threshold_nodes / len(similarity_between_sentences)

    elif choice == 2:  # word embeding
        # P3: Cümle benzerliği threshold'unu geçen node'larını bul
        threshold_nodes = 0
        for node in sentences:
            if node.text != sentence and sentence_similarity_word(sentence, node.text) > threshold:
                threshold_nodes += 1

        p3 = threshold_nodes / len(similarity_between_sentences)

    # P4: Cümlede başlıktaki kelimelerin kontrolü
    title_word_count = sum(token in sentence for token in title_keywords)
    p4 = title_word_count / sentence_length

    # P5: Her kelimenin TF-IDF değerinin hesapla
    theme_words_count = int(len(document_keywords) * 0.1)  # Tema kelimelerin sayısı
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([sentence] + document_keywords)
    tfidf_scores = tfidf_matrix[0].toarray()[0]
    theme_words_score = np.sum(np.sort(tfidf_scores)[-theme_words_count:])
    p5 = theme_words_score / sentence_length

    sentence_score = p1 + p2 + p3 + p4 + p5

    return sentence_score


class SummaryDialog(QDialog):
    def __init__(self, summary_text):
        super().__init__()
        self.setWindowTitle("Özet")
        self.layout = QVBoxLayout()
        self.summary_text = summary_text

        self.summary_label = QLabel("ÖZETİMİZ")
        self.summary_label.setStyleSheet("font-size: 16px; font-weight: bold")
        self.layout.addWidget(self.summary_label)

        self.text_browser = QTextBrowser()
        self.text_browser.setPlainText(self.summary_text)
        self.text_browser.setStyleSheet("font-size: 14px")
        self.layout.addWidget(self.text_browser)

        self.reference_label = QLabel("GERÇEK ÖZET")
        self.reference_label.setStyleSheet("font-size: 16px; font-weight: bold")
        self.layout.addWidget(self.reference_label)

        self.reference_text_edit = QTextEdit()
        self.reference_text_edit.setPlaceholderText("Referans metni girin")
        self.layout.addWidget(self.reference_text_edit)

        self.compare_button = QPushButton("Karşılaştır")
        self.compare_button.clicked.connect(self.compareSummaries)
        self.layout.addWidget(self.compare_button)

        self.setLayout(self.layout)
        self.resize(500, 400)

    def calculateROUGEScore(self, summary, reference):
        rouge = Rouge()
        scores = rouge.get_scores(summary, reference)
        return scores[0]

    def compareSummaries(self):
        reference_summary = self.reference_text_edit.toPlainText()
        rouge_scores = self.calculateROUGEScore(self.summary_text, reference_summary)


        rouge_1_f_score = rouge_scores['rouge-1']['f']
        rouge_1_p_score = rouge_scores['rouge-1']['p']
        rouge_1_r_score = rouge_scores['rouge-1']['r']

        rouge_2_f_score = rouge_scores['rouge-2']['f']
        rouge_2_p_score = rouge_scores['rouge-2']['p']
        rouge_2_r_score = rouge_scores['rouge-2']['r']

        rouge_l_f_score = rouge_scores['rouge-l']['f']
        rouge_l_p_score = rouge_scores['rouge-l']['p']
        rouge_l_r_score = rouge_scores['rouge-l']['r']

        # Skorları iletişim kutusunda göster
        message = f"ROUGE-1 Skoru:\nF-Skor: {rouge_1_f_score}\nP-Skor: {rouge_1_p_score}\nR-Skor: {rouge_1_r_score}\n\n"
        message += f"ROUGE-2 Skoru:\nF-Skor: {rouge_2_f_score}\nP-Skor: {rouge_2_p_score}\nR-Skor: {rouge_2_r_score}\n\n"
        message += f"ROUGE-L Skoru:\nF-Skor: {rouge_l_f_score}\nP-Skor: {rouge_l_p_score}\nR-Skor: {rouge_l_r_score}"

        QMessageBox.information(self, "ROUGE Skorları", message)


class ThresholdDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Threshold Değeri")

        self.threshold_label = QLabel("Threshold değerini girin:")
        self.threshold_input = QLineEdit()
        self.threshold_input.setText("0.0")

        self.ok_button = QPushButton("Tamam")
        self.ok_button.clicked.connect(self.accept)

        layout = QVBoxLayout()
        layout.addWidget(self.threshold_label)
        layout.addWidget(self.threshold_input)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def get_threshold(self):
        return float(self.threshold_input.text())


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.threshold_similarity = 0.0
        self.threshold_score = 0.0
        self.title = 'Belge Grafiği'
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setMinimumSize(800, 800)
        self.setMaximumSize(QDesktopWidget().availableGeometry().size())

        self.button_load = QPushButton('Belge Yükle', self)
        self.button_load.setToolTip('Bir belge seçmek için tıklayın')
        self.button_load.clicked.connect(self.showDialog)

        self.textbox = QTextEdit(self)
        self.textbox.setReadOnly(True)
        self.textbox.setFixedHeight(100)

        # Cümle benzerliği için threshold değeri
        self.threshold_label_sentence = QLabel("Cümle Benzerliği İçin Threshold Değeri:", self)
        self.threshold_input_sentence = QLineEdit(self)
        self.threshold_input_sentence.setValidator(QDoubleValidator())  # Ondalıklı girişe izin ver
        self.threshold_input_sentence.setText(str(self.threshold_similarity))  # Mevcut threshold değerini göster

        # Cümle skoru için threshold değeri
        self.threshold_label_score = QLabel("Cümle Skoru İçin Threshold Değeri:", self)
        self.threshold_input_score = QLineEdit(self)
        self.threshold_input_score.setValidator(QDoubleValidator())  # Ondalıklı girişe izin ver
        self.threshold_input_score.setText(str(self.threshold_score))  # Mevcut threshold değerini göster

        # Threshold değerleri güncellendiğinde self.threshold değişkenlerini güncelle
        self.threshold_input_sentence.textChanged.connect(self.updateThresholdSimilarity)
        self.threshold_input_score.textChanged.connect(self.updateThresholdScore)

        # Grafiği çiz butonu
        self.button_draw_graph = QPushButton(' Her İki Algoritmaya Göre Grafiği Çiz', self)
        self.button_draw_graph.clicked.connect(self.drawGraph)
        self.button_draw_graph.setMinimumWidth(self.button_draw_graph.sizeHint().width())

        # BERT Embedding'e göre grafik çiz butonu
        self.button_draw_graph_bert = QPushButton('BERT Embedding\'e Göre Grafiği Çiz', self)
        self.button_draw_graph_bert.clicked.connect(self.drawGraphBert)
        self.button_draw_graph_bert.setMinimumWidth(self.button_draw_graph_bert.sizeHint().width())

        # Word Embedding'e göre grafik çiz butonu
        self.button_draw_graph_word = QPushButton('WORD Embedding\'e Göre Grafiği Çiz', self)
        self.button_draw_graph_word.clicked.connect(self.drawGraphWord)
        self.button_draw_graph_word.setMinimumWidth(
            self.button_draw_graph_word.sizeHint().width())

        # Özeti çıkar butonu
        self.button_extract_summary = QPushButton('Özeti Çıkar', self)
        self.button_extract_summary.clicked.connect(self.extractSummary)
        self.button_extract_summary.setMinimumWidth(
            self.button_extract_summary.sizeHint().width())

        # Ana layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Üst layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.button_load)
        top_layout.addWidget(self.textbox)
        main_layout.addLayout(top_layout)

        # Orta layout
        middle_layout = QHBoxLayout()
        middle_layout.addWidget(self.threshold_label_sentence)
        middle_layout.addWidget(self.threshold_input_sentence)
        middle_layout.addWidget(self.threshold_label_score)
        middle_layout.addWidget(self.threshold_input_score)
        main_layout.addLayout(middle_layout)

        # Alt layout
        bottom_layout = QHBoxLayout()

        bottom_layout.addWidget(self.button_draw_graph)
        bottom_layout.addWidget(self.button_draw_graph_bert)
        bottom_layout.addWidget(self.button_draw_graph_word)
        # Özeti çıkar butonunu ekle
        bottom_layout.addWidget(self.button_extract_summary)

        main_layout.addLayout(bottom_layout)

        # Grafiği göstermek için bir alan ekle
        self.graph_area = plt.figure(figsize=(12, 12))
        self.graph_canvas = FigureCanvas(self.graph_area)
        main_layout.addWidget(self.graph_canvas)

        font = self.font()
        font.setPointSize(12)
        self.button_load.setFont(font)
        self.button_draw_graph.setFont(font)
        self.button_draw_graph_bert.setFont(font)
        self.button_draw_graph_word.setFont(font)
        self.button_extract_summary.setFont(font)
        self.threshold_label_sentence.setFont(font)
        self.threshold_input_sentence.setFont(font)
        self.threshold_label_score.setFont(font)
        self.threshold_input_score.setFont(font)
        self.textbox.setFont(font)
        self.showMaximized()
        self.show()

    def updateThresholdSimilarity(self):
        try:
            self.threshold_similarity = float(self.threshold_input_sentence.text())
        except ValueError:
            self.threshold_similarity = 0.0

    def updateThresholdScore(self):
        try:
            self.threshold_score = float(self.threshold_input_score.text())
        except ValueError:
            self.threshold_score = 0.0

    def extractSummary(self):
        content = self.textbox.toPlainText()
        # Başlığı belirle
        if "\n" in content:
            title = content.split("\n", 1)[0]
            content = content.replace(title, "")
        else:
            title = ""
        # Cümleleri listeye at
        doc = nlp(content)
        sentences = list(doc.sents)

        # Cümle skorlarına göre sırala
        sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

        # Özetin yüzdesini belirle
        summary_percentage = 0.5

        # İstenen cümle sayısını hesapla
        desired_sentences_count = int(len(sentences) * summary_percentage)

        # İstenen cümleleri al
        selected_sentences = sorted_sentences[:desired_sentences_count]

        # Cümle sırasına göre özet metnini oluştur
        summary = ""
        for index, _ in selected_sentences:
            summary += str(sentences[index]) + " "

        # Özeti göstermek için yeni bir pencere aç
        summary_dialog = SummaryDialog(summary)
        summary_dialog.exec_()

    def drawGraph(self):  # her iki algoritmaya göre çizme

        threshold = self.threshold_similarity
        content = self.textbox.toPlainText()
        # Başlığı belirle
        if "\n" in content:
            title = content.split("\n", 1)[0]
            content = content.replace(title, "")
        else:
            title = ""

        # Tokenization
        tokens = word_tokenize(content)

        # Stop-word
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.casefold() not in stop_words]

        # Stemming
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

        processed_content = ' '.join(stemmed_tokens)

        # Spacy modelini yükler ve cümleleri çıkarırız
        nlp = spacy.load("en_core_web_sm")

        doc = nlp(processed_content)
        sentences = list(doc.sents)

        doc2 = nlp(content)
        sentences2 = list(doc2.sents)

        document_keywords = content.split()
        title_keywords = title.split()

        similarity_between_sentences.clear()
        sentence_scores.clear()
        node_scores.clear()

        # Node sayısı

        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity_between_sentence = sentence_similarity(sentences[i].text, sentences[j].text)  # bert
                print('similarity_between_sentence:', similarity_between_sentence)
                similarity_between_sentence_word = sentence_similarity_word(sentences[i].text,
                                                                            sentences[j].text)  # word
                print('similarity_between_sentence_word:', similarity_between_sentence_word)
                similarity_sentence = (similarity_between_sentence + similarity_between_sentence_word) / 2
                print('-------->', similarity_sentence)
                similarity_between_sentences.append((i, j, similarity_sentence))
        print('-------------------------------------------------------------------------')

        for i in range(len(sentences)):
            node_score = 0
            for j in range(len(sentences)):
                if i != j:
                    for similarity_item in similarity_between_sentences:
                        if (similarity_item[0] == i and similarity_item[1] == j) or (
                                similarity_item[0] == j and similarity_item[1] == i):
                            similarity = similarity_item[
                                2]  # similarity_between_sentences listesinden benzerlik değerini al
                            if similarity > threshold:
                                node_score += 1
                                break
            node_scores.append((i, node_score))

        for i, sentence in enumerate(sentences2):
            score = calculate_sentence_score(sentence.text, title_keywords, document_keywords, self.threshold_score,
                                             0, sentences2)  # her iki algoritma
            sentence_scores.append((i, score))

        self.graph_area.clear()  # Grafik alanını temizle

        G = nx.Graph()
        # kenarları ekle -> içini doldur
        for i, j, similarity in similarity_between_sentences:
            if similarity > 0:
                label_i = f"Cümle {i + 1} \n Skor: {sentence_scores[i][1]:.2f} \n Nodes: {node_scores[i][1]}"
                label_j = f"Cümle {j + 1} \n Skor: {sentence_scores[j][1]:.2f} \n Nodes: {node_scores[j][1]}"
                G.add_edge(label_i, label_j, weight=similarity)

        # Ağı çiz
        pos = nx.spring_layout(G)

        # Kenar Kalınlığını Ayarla
        for u, v, data in G.edges(data=True):
            similarity = data['weight']
            if similarity > threshold:  # Threshold değerini geçen cümle çiftleri için
                # Kalınlık değerini belirle
                thickness = 4.5
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=thickness)

        # Kenar Renklendirmesi Yap
        for u, v, data in G.edges(data=True):
            similarity = data['weight']
            if similarity > threshold:  # Threshold değerini geçen cümle çiftleri için
                # Renk değerini belirle
                color = tuple([random.uniform(0, 1) for _ in range(3)])
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color)

        for u, v, data in G.edges(data=True):
            color = tuple([random.uniform(0, 1) for _ in range(3)])
            similarity_score = round(data['weight'], 2)
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): similarity_score}, font_size=12)
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color, width=data['weight'])

        nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif")
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2000)

        self.graph_canvas.draw()

    def drawGraphBert(self):
        threshold = self.threshold_similarity
        content = self.textbox.toPlainText()
        # Başlığı belirle
        if "\n" in content:
            title = content.split("\n", 1)[0]
            content = content.replace(title, "")
        else:
            title = ""

        # Tokenization
        tokens = word_tokenize(content)

        # Stop-word
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.casefold() not in stop_words]

        # Stemming
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

        processed_content = ' '.join(stemmed_tokens)

        # Spacy modelini yükle ve cümleleri çıkar
        nlp = spacy.load("en_core_web_sm")

        doc = nlp(processed_content)
        sentences = list(doc.sents)

        doc2 = nlp(content)
        sentences2 = list(doc2.sents)

        document_keywords = content.split()
        title_keywords = title.split()

        similarity_between_sentences.clear()
        sentence_scores.clear()
        node_scores.clear()

        for i in range(len(sentences)):

            for j in range(i + 1, len(sentences)):
                similarity_between_sentence = sentence_similarity(sentences[i].text,
                                                                  sentences[j].text)  # bert

                similarity_between_sentences.append((i, j, similarity_between_sentence))

        for i in range(len(sentences)):
            node_score = 0
            for j in range(len(sentences)):
                if i != j:
                    for similarity_item in similarity_between_sentences:
                        if (similarity_item[0] == i and similarity_item[1] == j) or (
                                similarity_item[0] == j and similarity_item[1] == i):
                            similarity = similarity_item[
                                2]  # similarity_between_sentences listesinden benzerlik değerini al
                            if similarity > threshold:
                                node_score += 1
                                break
            node_scores.append((i, node_score))

        for i, sentence in enumerate(sentences2):
            score = calculate_sentence_score(sentence.text, title_keywords, document_keywords, self.threshold_score,
                                             1, sentences2)  # word embeding
            sentence_scores.append((i, score))

        self.graph_area.clear()  # Grafik alanını temizle

        G = nx.Graph()
        # Diğer cümlelerle benzerliklerini hesapla ve kenarları ekle
        for i, j, similarity in similarity_between_sentences:
            if similarity > 0:  # burası dinamik olacak
                label_i = f"Cümle {i + 1} \n Skor: {sentence_scores[i][1]:.2f} \n Nodes: {node_scores[i][1]}"
                label_j = f"Cümle {j + 1} \n Skor: {sentence_scores[j][1]:.2f} \n Nodes: {node_scores[j][1]}"
                G.add_edge(label_i, label_j, weight=similarity)

        # Ağı çiz
        pos = nx.spring_layout(G)

        # Kenar Kalınlığını Ayarla
        for u, v, data in G.edges(data=True):
            similarity = data['weight']
            if similarity > threshold:  # Threshold değerini geçen cümle çiftleri için
                # Kalınlık değerini belirle
                thickness = 4.5
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=thickness)

        # Kenar Renklendirmesi Yap
        for u, v, data in G.edges(data=True):
            similarity = data['weight']
            if similarity > threshold:  # Threshold değerini geçen cümle çiftleri için
                # Renk değerini belirle
                color = tuple([random.uniform(0, 1) for _ in range(3)])
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color)

        for u, v, data in G.edges(data=True):
            color = tuple([random.uniform(0, 1) for _ in range(3)])
            similarity_score = round(data['weight'], 2)
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): similarity_score}, font_size=12)
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color, width=data['weight'])

        nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif")
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2000)

        self.graph_canvas.draw()

    def drawGraphWord(self):

        threshold = self.threshold_similarity
        content = self.textbox.toPlainText()
        # Başlığı belirle
        if "\n" in content:
            title = content.split("\n", 1)[0]
            content = content.replace(title, "")
        else:
            title = ""

        # Tokenization
        tokens = word_tokenize(content)

        # Stop-word
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.casefold() not in stop_words]

        # Stemming
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

        processed_content = ' '.join(stemmed_tokens)

        # Spacy modelini yükle ve cümleleri çıkar
        nlp = spacy.load("en_core_web_sm")

        doc = nlp(processed_content)
        sentences = list(doc.sents)

        doc2 = nlp(content)
        sentences2 = list(doc2.sents)

        document_keywords = content.split()
        title_keywords = title.split()

        similarity_between_sentences.clear()
        sentence_scores.clear()
        node_scores.clear()
        # Node sayısı

        for i, sentence in enumerate(sentences):
            node_score = nodes_passing_the_threshold(sentence.text, sentences, threshold)
            node_scores.append((i, node_score))

        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity_between_sentence_word = sentence_similarity_word(sentences[i].text,
                                                                            sentences[j].text)  # word
                similarity_between_sentences.append((i, j, similarity_between_sentence_word))

        for i, sentence in enumerate(sentences2):
            score = calculate_sentence_score(sentence.text, title_keywords, document_keywords, self.threshold_score,
                                             2, sentences2)  # word embeding
            sentence_scores.append((i, score))

        self.graph_area.clear()  # Grafik alanını temizle

        G = nx.Graph()
        # Diğer cümlelerle benzerliklerini hesapla ve kenarları ekle
        for i, j, similarity in similarity_between_sentences:
            if similarity > 0:
                label_i = f"Cümle {i + 1} \n Skor: {sentence_scores[i][1]:.2f} \n Nodes: {node_scores[i][1]}"  # İlgili düğüm etiketini alın
                label_j = f"Cümle {j + 1} \n Skor: {sentence_scores[j][1]:.2f} \n Nodes: {node_scores[j][1]}"  # İlgili düğüm etiketini alın
                G.add_edge(label_i, label_j, weight=similarity)

        # Ağı çiz
        pos = nx.spring_layout(G)

        # Kenar Kalınlığını Ayarla
        for u, v, data in G.edges(data=True):
            similarity = data['weight']
            if similarity > threshold:  # Threshold değerini geçen cümle çiftleri için
                # Kalınlık değerini belirle
                thickness = 4.5
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=thickness)

        # Kenar Renklendirmesi Yap
        for u, v, data in G.edges(data=True):
            similarity = data['weight']
            if similarity > threshold:  # Threshold değerini geçen cümle çiftleri için
                # Renk değerini belirle
                color = tuple([random.uniform(0, 1) for _ in range(3)])
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color)

        for u, v, data in G.edges(data=True):
            color = tuple([random.uniform(0, 1) for _ in range(3)])
            similarity_score = round(data['weight'], 2)
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): similarity_score}, font_size=12)
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color, width=data['weight'])

        nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif")
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2000)

        self.graph_canvas.draw()

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Belge Yükle', '', 'Metin Dosyaları (*.txt);;Tüm Dosyalar (*)')
        if fname[0]:
            with open(fname[0], 'r', encoding="utf-8") as f:
                content = f.read()
                self.textbox.setText(content)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    app.exec_()
