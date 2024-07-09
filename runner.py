import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from gensim.models import KeyedVectors
import pickle
from scipy import spatial
from scipy.stats import pearsonr

def load_file(file_path):
    with open(file_path, 'rb') as handle:
        file = pickle.load(handle)
    return file

def sts_score(sim_score):
    sts_score = (sim_score+1) * 2.5
    return sts_score

def get_sts_scores(emb1, emb2):
    sim_score = 1 - spatial.distance.cosine(emb1, emb2)
    return sim_score

def pearson_corr(y_true, y_pred):
    corr, _ = pearsonr(y_true, y_pred)
    return corr

class STS_Score:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.ps = PorterStemmer()
        self.word_dict = load_file("data/word_dict.pickle")
        model_path = "data/GoogleNews-vectors-negative300.bin"
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    def remove_punctuation(self,text):
        return re.sub(r'[^\w\s]', '', text)

    def remove_number(self,text):
        return re.sub(r'\d+', 'num', text)

    def replace_url(self,text):
        return re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', 'url', text)

    def replace_hashtags(self,text):
        return re.sub(r'#[a-zA-Z\d]+', 'hashtag', text)

    def replace_email(self,text):
        return re.sub(r'[a-zA-Z\.]+@[a-zA-Z\.\d]+', 'email', text)

    def replace_mentions(self,text):
        return re.sub(r'@[a-zA-Z\.\d_]+', 'mention', text)

    def preprocess_text(self,text):
        text = self.remove_punctuation(text)
        text = self.remove_number(text)
        text = self.replace_url(text)
        text = self.replace_hashtags(text)
        text = self.replace_email(text)
        text = self.replace_mentions(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        text = text.lower()
        sentence = text.split()
        sentence = [word for word in sentence if word not in stop_words]
        sentence = [lemmatizer.lemmatize(word) for word in sentence]
        sentence = [stemmer.stem(word) for word in sentence]
        return sentence

    def unk_replace(self, word, word_dict):
        if word not in word_dict:
            return "unk"
        else:
            if word_dict[word] < 2:
                return "unk"
        return word

    def get_sentence_embedding(self, sentence):
        words = sentence
        words = [word for word in words if word in self.model.key_to_index]
        embeddings = [self.model[word] for word in words]
        embedding = np.mean(embeddings, axis=0)
        return embedding

    def generate_similarity_score(self, sent1, sent2):
        sent1_token = self.preprocess_text(sent1)
        sent2_token = self.preprocess_text(sent2)
        sent1_unk_token = [self.unk_replace(word, self.word_dict) for word in sent1_token]
        sent2_unk_token = [self.unk_replace(word, self.word_dict) for word in sent2_token]
        sent1_embedding = self.get_sentence_embedding(sent1_unk_token)
        sent2_embedding = self.get_sentence_embedding(sent2_unk_token)
        normalized_cos_scores = sts_score(get_sts_scores(sent1_embedding, sent2_embedding))
        return normalized_cos_scores


if __name__ == "__main__":
    input1 = input("Enter your first sentence: ")
    input2 = input("Enter your second sentence: ")

    print("Generating the similarity score......")
    sts = STS_Score()
    print("The semantic similarity score is", sts.generate_similarity_score(input1, input2))
    