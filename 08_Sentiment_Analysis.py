import re
import pyprind
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation

author = "Oscar Ding"

"""
Cleaning and preparing text data 
"""
df = pd.read_csv('movie_data.csv', encoding='utf-8')
print(df.head(3))

# bag-of-words model
# Transforming documents into feature vectors
count = CountVectorizer(ngram_range=(1, 1))
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)

print(count.vocabulary_)
print(bag.toarray())

# Assessing word relevancy via term frequency-inverse document frequency
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())


# Cleaning text data
# regular expression (regex)


def preprocessor(text):
    text = re.sub('<[^>]*', '', text)  # remove all the HTML markup
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)  # find all the emoticons
    # remove all non-word characters and lowercase
    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))

    return text


preprocessor("</a>This :) is :( a test :-)!")
df['review'] = df['review'].apply(preprocessor)

# Processing documents into tokens
porter = PorterStemmer()


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


tokenizer_porter('runners like running and thus they run')

stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
 if w not in stop]

"""
# Training a logistic regression model for document classification
"""
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

# param_grid = [{'vect__ngram_range': [(1, 1)],
#                'vect__stop_words': [stop, None],
#                'vect__tokenizer': [tokenizer, tokenizer_porter],
#                'clf__penalty': ['l1', 'l2'],
#                'clf__C': [1.0, 10.0, 100.0]},
#               {'vect__ngram_range': [(1, 1)],
#                'vect__stop_words': [stop, None],
#                'vect__tokenizer': [tokenizer, tokenizer_porter],
#                'vect__use_idf':[False],
#                'vect__norm':[None],
#                'clf__penalty': ['l1', 'l2'],
#                'clf__C': [1.0, 10.0, 100.0]}
#               ]
param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(solver='liblinear', multi_class='auto', random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=10)

print("start fitting gs models")
gs_lr_tfidf.fit(X_train, y_train)

print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

"""
online algorithms and out-of-core learning
"""


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


next(stream_docs(path='movie_data.csv'))


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None

    return docs, y


vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)

clf = SGDClassifier(loss='log', random_state=1, max_iter=1, tol=1e-3)
doc_stream = stream_docs(path='movie_data.csv')

pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])

for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

X_test, y_test = get_minibatch(doc_stream, size = 5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

clf = clf.partial_fit(X_test, y_test)

"""
Topic modeling
"""
# Decomposing text documents with Latent Dirichlet Allocation
df = pd.read_csv('movie_data.csv', encoding='utf-8')

count = CountVectorizer(stop_words='english',
                        max_df=.1,
                        max_features=5000)

X = count.fit_transform(df['review'].values)

lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch',
                                n_jobs=8)

X_topics = lda.fit_transform(X)
print(lda.components_.shape)

n_top_words = 5
feature_names = count.get_feature_names()

for topic_idx, topic in enumerate(lda.components_):
    print('Topic {}'.format(topic_idx + 1))
    print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words -1: -1]]))

horror = X_topics[:, 5].argsort()[::-1]

for iter_idx, movie_idx in enumerate(horror[:3]):
    print('\nHorror movie #%d:' % (iter_idx + 1))
    print(df['review'][movie_idx][:300], '...')








