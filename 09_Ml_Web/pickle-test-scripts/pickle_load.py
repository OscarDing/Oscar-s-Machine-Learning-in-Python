import pickle
import numpy as np
from vectorizer import vect

clf = pickle.load(open('classifier.pkl', 'rb'))

label = {0: 'negative', 1: 'positive'}
example = ['I love this movie']

X = vect.transform(example)

print(
    'Prediction: {} \nProbability: {}%'.format(label[clf.predict(X)[0]], round(np.max(clf.predict_proba(X)) * 100, 2)))
