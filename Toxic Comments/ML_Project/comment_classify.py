#importing required packages
import numpy as np
import nltk
import sklearn
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import sys, re, string, pickle
from itertools import product
from scipy.sparse import csr_matrix, hstack
from spellchecker import SpellChecker
# from nltk.stem import PorterStemmer
sys.setrecursionlimit(10**6) #Setting recurssion limit

#downloading required set of words from nltk
import nltk
nltk.download('stopwords')
nltk.download('words')
nltk.download('brown')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
get_ipython().system('pip3 install pyspellchecker')
get_ipython().system('pip3 install -U symspellpy')

#TO SAVE THE FILE IN A PICKLE FORMAT
def save_to_pickle(filename, model):
    pkl_filename = filename
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

#TO LOAD MODEL FROM A PICKLE FILE
def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        pickle_model = pickle.load(file)
    return pickle_model

#Loading data
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

#Loading extra data
data_train = data_train.append(pd.read_csv("train_big.csv"), ignore_index=True)
dt = data_train.copy(deep = True)
dt.set_index('id', inplace = True)
dt1 = dt.drop_duplicates()
dt1.reset_index(inplace = True)
data_train = dt1.copy(deep = True)

#Creating important sets of words to check or remove
stop_words = list(set(stopwords.words('english')))
english_words = set(nltk.corpus.words.words())
brown_words = set(nltk.corpus.brown.words())
single_letters = set(map(chr, range(97, 123)))
english_words = english_words.union(single_letters)
english_words = english_words.union(brown_words)

# def Remove_Noise(input_text):
#     stem_p = PorterStemmer()
#     result = re.sub(r"http\S+", "", input_text)
#     result = result.translate(str.maketrans('', '', string.punctuation))
#     result = ' '.join(s for s in result.split() if not any(c.isdigit() for c in s))

#     result = " ".join(w for w in nltk.wordpunct_tokenize(result) \
#          if w.lower() in english_words or not w.isalpha())

#     words = result.split()

#     nfw= [re.sub(r'[^\x00-\x7F]+',' ', word) for word in words if word not in stop_words]
#     nfw_stem = [stem_p.stem(x) for x in nfw]
#     noise_free_text = " ".join(nfw_stem)
#     return noise_free_text

# from nltk.corpus import wordnet

# def get_wordnet_pos(word):
#     """Map POS tag to first character lemmatize() accepts"""
#     tag = nltk.pos_tag([word])[0][1][0].upper()
#     tag_dict = {"J": wordnet.ADJ,
#                 "N": wordnet.NOUN,
#                 "V": wordnet.VERB,
#                 "R": wordnet.ADV}
#     print(tag_dict.get(tag, wordnet.NOUN))
#     return tag_dict.get(tag, wordnet.NOUN)

# from nltk.stem import WordNetLemmatizer
# Lemm = WordNetLemmatizer()

# def Lemmetizer(input_text):
#     result = re.sub(r"http\S+", "", input_text)
#     result = result.translate(str.maketrans('', '', string.punctuation))
#     result = ' '.join(s for s in result.split() if not any(c.isdigit() for c in s))

#     result = " ".join(w for w in nltk.wordpunct_tokenize(result) \
#          if w.lower() in english_words or not w.isalpha())
#     words = result.split()

#     nfw= [re.sub(r'[^\x00-\x7F]+',' ', word) for word in words if word not in stop_words]
#     nfw_lemm = [Lemm.lemmatize(x,pos = get_wordnet_pos(x)) for x in nfw]
#     Lem_text = " ".join(nfw_lemm)
#     return Lem_text

#Maximum number of consecutive characters in the text.
def MaxRep(text):
    n = len(text)
    count = 0
    res = text[0]
    cur_count = 1
    # Traverse string except
    # last character
    for i in range(n):

        # If current character
        # matches with next
        if (i < n - 1 and
            text[i] == text[i + 1]):
            cur_count += 1

        # If doesn't match, update result
        # (if required) and reset count
        else:
            if cur_count > count:
                count = cur_count
                res = text[i]
            cur_count = 1
    return count

import pkg_resources
from symspellpy import SymSpell, Verbosity
from textblob import TextBlob, Word

sym_spell = SymSpell(max_dictionary_edit_distance=2)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

#this function lemmatizes the input_text and returns the updated text
def lemmatize_with_postag(input_text,T_dict,Id):
    result = re.sub(r"http\S+", "", input_text)
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    result = regex.sub(' ', result)
    # result = result.translate(str.maketrans(' ', '', string.punctuation))
    result = ' '.join(s for s in result.split() if not any(c.isdigit() for c in s))

    # result = " ".join(w for w in nltk.wordpunct_tokenize(result) \
    #     if w.lower() in english_words or not w.isalpha())
    word_s = result.split()
    #words = [str(sym_spell.lookup(x, Verbosity.CLOSEST,max_edit_distance=2, include_unknown=True)[0]).split(',')[0] for x in word_s if x not in english_words]
    words = [x for x in word_s if (MaxRep(x))<3]
    # words = [x for x in words if len(x)>2]

    #words = [spell.correction(x) for x in words if x not in english_words]
    nfw= [re.sub(r'[^\x00-\x7F]+',' ', word) for word in words if word not in stop_words]
    Lem_text = " ".join(nfw)
    sent = TextBlob(Lem_text)
    tag_dict = {"J": 'a',
                "N": 'n',
                "V": 'v',
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]
    # lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags if len(wd.lemmatize(tag))>2 and MaxRep(wd.lemmatize(tag))<3]
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags if MaxRep(wd.lemmatize(tag))<3]
    Ret = " ".join(lemmatized_list)
    T_dict[Id] = Ret
    return Ret

X = np.array(data_train["comment_text"])
Y = np.array(data_train[['toxic' , 'severe_toxic', 'obscene' , 'threat', 'insult' , 'identity_hate' ]])

# from imblearn.over_sampling import ADASYN
# ada= ADASYN(sampling_strategy='auto', random_state=None, n_neighbors=5, n_jobs=1, ratio=None)
# data2_train = data_train.copy(deep=True)
# data2_test = data_test.copy(deep=True)

# Xr,Yr = ada.fit_resample(np.transpose(X),np.transpose(Y))

# from sklearn.feature_extraction.text import TfidfVectorizer

# def TFIDF(token_dict):
#   tfidf = TfidfVectorizer(stop_words='english')
#   tfs = tfidf.fit(token_dict.values())
#   return tfs

data2_train = data_train.copy(deep=True)
M_D = []
Train_dict = {}
# print(data2.shape[0])
for i in tqdm(range(data2_train.shape[0])):
    # s = data2_train["comment_text"][i]
    # print(type(s))
    M_D.append(lemmatize_with_postag(data2_train["comment_text"][i].lower(),Train_dict,data2_train["id"][i]))

data2_train['ModifiedData'] = M_D
# data2.head()

data2_test = data_test.copy(deep=True)
M_D = []
Test_dict = {}
# print(data2.shape[0])
for i in tqdm(range(data2_test.shape[0])):
    M_D.append(lemmatize_with_postag(data2_test["comment_text"][i].lower(),Test_dict,data2_test["id"][i]))

data2_test['ModifiedData'] = M_D

from sklearn.feature_extraction.text import TfidfVectorizer
# from spellchecker import SpellChecker
# spell = SpellChecker()
tfidf = TfidfVectorizer(sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        min_df=5,
        norm='l2',
        ngram_range=(1, 1),
        max_features=50000)
result = set(Train_dict.values())
#result = [spell.correction(x) for x in tqdm(result)]
TFS = tfidf.fit(result)

tfidf1 = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    token_pattern=None,
    min_df=5,
    ngram_range=(2, 4),
    max_features=25000)
TFS1 = tfidf1.fit(result)

TrainTFS_W = TFS.transform(Train_dict.values())
TrainTFS_C= TFS1.transform(Train_dict.values())
TestTFS_W = TFS.transform(Test_dict.values())
TestTFS_C = TFS1.transform(Test_dict.values())

TrainTFS = hstack([TrainTFS_W,TrainTFS_C]).tocsr()
TestTFS = hstack([TestTFS_W,TestTFS_C]).tocsr()

temp = [ x for x in a if len(x)==15 if x in english_words]
temp1 = [ x for x in a if len(x)==15]
temp2 = [ x for x in a if len(x)==15 if x not in english_words]

# import pkg_resources
# from symspellpy import SymSpell, Verbosity

# sym_spell = SymSpell(max_dictionary_edit_distance=2)
# dictionary_path = pkg_resources.resource_filename(
#     "symspellpy", "frequency_dictionary_en_82_765.txt")
# sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# temp = [ str(sym_spell.lookup(x, Verbosity.CLOSEST,max_edit_distance=2, include_unknown=True)[0]).split(',')[0] for x in tqdm(TFS.get_feature_names()) if x not in english_words]
# #  and (str(sym_spell.lookup(x, Verbosity.CLOSEST,max_edit_distance=2, include_unknown=True)[0]).split(',')[0] not in stop_words or len(sym_spell.lookup(x, Verbosity.CLOSEST,max_edit_distance=2, include_unknown=True)[0]).split(',')[0])>2)  ]
# temp1 = [x for x in temp if len(x)>2 or x not in stop_words]
# TFS.set_feature_names = temp1
# len(TFS.get_feature_names())

# TrainTFS = TFS.transform(Train_dict.values())
# TestTFS = TFS.transform(Test_dict.values())
# Train_dict['04237ea01685d4b6']

# import gensim
# from gensim.models import Word2Vec
# data = Train_dict.values()
# x = [l.split(" ") for l in data]
# y = []
# for i in range(len(x)):
#   y+=x[i]
# #len(y)
# print("Ready\n")
# #CBOW
# model1 = gensim.models.Word2Vec(y, min_count = 1,  size = 300, window = 5)
# print("Traing Model Train\n")
# model1.train(y, total_examples=len(y), epochs=4
# print("REady Ready\n")
# data_test = Test_dict.values()
# x1 = [l.split(" ") for l in data_test]
# y1 = []
# for i in range(len(x1)):
#   y1+=x1[i]

# print("Training Test Model\n")
# model1_test = gensim.models.Word2Vec(y1, min_count = 1,  size = 300, window = 5)
# model1_test.train(y1, total_examples=len(y1), epochs=4)

#words = list(model1.wv.vocab)
#words
#SKIP GRAM
#model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100,  window = 5, sg = 1)
#words2 = list(model2.wv.vocab)

data_temp_train = data2_train[['id', 'ModifiedData']]
data_temp_test = data2_test[['id', 'ModifiedData']]

texts = data_temp_train['ModifiedData']
cv = CountVectorizer()
cv.fit(texts)
cv_vocab = cv.vocabulary_
cv_vo_sort = {}
for k, v in sorted(cv_vocab.items()):
    cv_vo_sort[k] = cv_vocab[k]
vec_train = cv.transform(texts)

texts_test = data_temp_test['ModifiedData']
vec_test = cv.transform(texts_test)

#PCA
from sklearn.decomposition import TruncatedSVD
from scipy import sparse as sp

n_op = 10
clf = TruncatedSVD(n_op)
vec_pca_train = clf.fit_transform(TrainTFS)
vec_pca_test = clf.fit_transform(TestTFS)

# from sklearn.pipeline import Pipeline
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import GridSearchCV

# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(stop_words=stop_words)),
#     ('clf', OneVsRestClassifier(MultinomialNB(
#         fit_prior=True, class_prior=None))),
# ])

# parameters = {
#     'tfidf__max_df': (0.25, 0.5, 0.75),
#     'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
#     'clf__estimator__alpha': (1e-2, 1e-3)
# }

# grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=2, verbose=3)
# grid_search_tune.fit(TrainTFS, y_train)

# print("Best parameters set:")
# print (grid_search_tune.best_estimator_.steps)

save_to_pickle('pre_prc_testTFS.pkl', TestTFS)

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble  import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import ComplementNB

#toxic,severe_toxic,obscene,threat,insult,identity_hate
log_clf = {'toxic' : None, 'severe_toxic' : None, 'obscene' : None, 'threat': None, 'insult' : None, 'identity_hate' : None}
svm_clf = {'toxic' : None, 'severe_toxic' : None, 'obscene' : None, 'threat': None, 'insult' : None, 'identity_hate' : None}
Rf_clf = {'toxic' : None, 'severe_toxic' : None, 'obscene' : None, 'threat': None, 'insult' : None, 'identity_hate' : None}
SGD_clf = {'toxic' : None, 'severe_toxic' : None, 'obscene' : None, 'threat': None, 'insult' : None, 'identity_hate' : None}
Ridge_clf = {'toxic' : None, 'severe_toxic' : None, 'obscene' : None, 'threat': None, 'insult' : None, 'identity_hate' : None}
CNB_clf = {'toxic' : None, 'severe_toxic' : None, 'obscene' : None, 'threat': None, 'insult' : None, 'identity_hate' : None}
NN_clf = {'toxic' : None, 'severe_toxic' : None, 'obscene' : None, 'threat': None, 'insult' : None, 'identity_hate' : None}

y_pred_svm = {'toxic' : None, 'severe_toxic' : None, 'obscene' : None, 'threat': None, 'insult' : None, 'identity_hate' : None}
y_pred_log = {'toxic' : None, 'severe_toxic' : None, 'obscene' : None, 'threat': None, 'insult' : None, 'identity_hate' : None}
y_pred_Rf = {'toxic' : None, 'severe_toxic' : None, 'obscene' : None, 'threat': None, 'insult' : None, 'identity_hate' : None}
y_pred_SGD = {'toxic' : None, 'severe_toxic' : None, 'obscene' : None, 'threat': None, 'insult' : None, 'identity_hate' : None}
y_pred_Ridge = {'toxic' : None, 'severe_toxic' : None, 'obscene' : None, 'threat': None, 'insult' : None, 'identity_hate' : None}
y_pred_CNB = {'toxic' : None, 'severe_toxic' : None, 'obscene' : None, 'threat': None, 'insult' : None, 'identity_hate' : None}
y_pred_NN = {'toxic' : None, 'severe_toxic' : None, 'obscene' : None, 'threat': None, 'insult' : None, 'identity_hate' : None}

y_train = {'toxic' : data2_train['toxic'], 'severe_toxic' : data2_train['severe_toxic'], 'obscene' : data2_train['obscene'], 'threat': data2_train['threat'], 'insult' : data2_train['insult'], 'identity_hate' : data2_train['identity_hate']}

for k in tqdm(log_clf.keys()) :
    clf = LogisticRegression()
    clf.fit(TrainTFS, y_train[k])
    log_clf[k] = clf
print('Done Training Log')

# for k in tqdm(svm_clf.keys()) :
#     clf = svm_clf[k]
#     clf = SVC(gamma='auto')
#     clf.fit(vec_train, y_train[k])
#     svm_clf[k] = clf
# print('Done Training SVM')

# for k in tqdm(Ridge_clf.keys()):
#     Ridgemodel = RidgeClassifier()
#     Ridgemodel.fit(TrainTFS, y_train[k])
#     Ridge_clf[k] = Ridgemodel
# print('Done Training Ridge')

# for k in tqdm(CNB_clf.keys()):
#     CNBmodel = ComplementNB(alpha=1.0, class_prior=None, fit_prior=True, norm=False)
#     CNBmodel.fit(TrainTFS, y_train[k])
#     CNB_clf[k] = CNBmodel
# print('Done Training CNB')

# for k in tqdm(SGD_clf.keys()):
#      SGDModel = linear_model.SGDClassifier(alpha=0.0001, average=False, class_weight=None,
#        early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
#         learning_rate='optimal', loss='modified_huber', max_iter=1000,
#        n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,
#        random_state=None, shuffle=True, tol=0.001, validation_fraction=0.1,
#        verbose=0, warm_start=False)
#      SGDModel.fit(TrainTFS,y_train[k])
#      SGD_clf[k] = SGDModel
# print('Done Training SGD')

# for k in tqdm(Rf_clf.keys()):
#     Model = RandomForestClassifier()
#     Model.fit(vec_train, y_train[k])
#     Rf_clf[k] = Model
# print('Done Training Rf')

for k in y_pred_log.keys():
    y_pred_log[k] = log_clf[k].predict_proba(TestTFS)
    y_pred_log[k] = np.transpose(y_pred_log[k])[1]

submission = data_test.drop(['comment_text'], axis = 1)
for x in tqdm(y_pred_log.keys()):
  submission[x] = y_pred_log[x]

submission.to_csv('submission.csv', index = False)

for key in tqdm(log_clf.keys()):
    fname = str(key) + '_log.pkl'
    save_to_pickle(fname, log_clf[key])
