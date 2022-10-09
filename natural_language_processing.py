#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:13:51 2019

@author: juangabriel
"""

# Natural Language Processing

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)      
# /t es el tabulador
# quoting = 3 significa ignorar las comillas dobles

# Limpieza de texto
import re
import nltk
nltk.download('stopwords')      # stopwords son las palabras irrelevantes
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):        # un bucle de la critica 0 a la 1000
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])     
    # ^ se usa este simbolo para las cosas que no quiero borrar
    # ' ' significa que cualquier simbolo que no sea una letra se intercambia por un espacio en blanco
    review = review.lower()     # Pasamos el texto a minuscula
    review = review.split()     # Separa las palabras y las pone en una lista
    ps = PorterStemmer()        # Pasar palabras al infinitivo
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]      
    # Eliminamos la palabras irrelevantes
    # ps.stem(word) pasa las palabras al infinitivo
    review = ' '.join(review)       # join es para unir las palabras sacandolas de la lista y ' ' para separarlas por un espacio
    corpus.append(review)
    
# Crear el Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)       # max_features es para que muestre las palabras mas repetidas osea las mas importantes
X = cv.fit_transform(corpus).toarray()      # Pasamos de palabras a numeros
y = dataset.iloc[:, 1].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Ajustar el clasificador en el Conjunto de Entrenamiento
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
(55+91)/200