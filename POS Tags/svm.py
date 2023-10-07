#import nltk
import pickle

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# preprocessing
# Each sentence is expressed as a list of word and tag tupples
train_data = []
tags = []
with open('train.txt', 'r') as f:
    train_content = f.read()  # Raw Data without separation

    sentences_raw = train_content.split('\n\n')  # separates sentence
    for sentence in sentences_raw:
        lines = sentence.split('\n')  #separates rows
        for line in lines:
            tagged_words = line.split()
            if len(tagged_words) >= 2:
                word = tagged_words[0]  # Every line has words only
                tag = tagged_words[1]  # Prints tags
                train_data.append(word)
                tags.append(tag)

encoder = LabelEncoder()
encoded_tags = encoder.fit_transform(tags)

def feature_extraction(data):
    features = []
    for i, x in enumerate(data):
        feature = {}
        # Store previous word
        if i > 0:
            feature["previous"] = data[i - 1]
        else:
            feature["previous"] = ''
        #check if capitalized
        if x.isupper():
            feature["capitalized"] = True
            #lower the word
            feature["word"] = x.lower()
        else:
            feature["capitalized"] = False
            feature["word"] = x
        feature["length"] = len(x)
        feature["prefix"] = x[:3]
        feature["suffix"] = x[-3:]
        features.append(feature)
    return features

extracted_data = feature_extraction(train_data)
dicVectorizer = DictVectorizer(sparse=True)
vectors = dicVectorizer.fit_transform(extracted_data)

x_train, x_test, y_train, y_test = train_test_split(vectors, encoded_tags, test_size=0.2, random_state=42)

classification_svm = SVC(kernel='linear')
classification_svm.fit(x_train, y_train)

predictions = classification_svm.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Cross-validation Accuracy: {accuracy * 100:.2f}%')  #96.58%

print(classification_report(y_test, predictions))

filename = 'finalized_model.sav'
pickle.dump(classification_svm, open(filename, 'wb'))

testing_processed= []
with open('unlabeled_test_test.txt', 'r') as testing:
    test_data = testing.read()
    sentences_raw = test_data.split('\n\n')  # separates sentence
    for sentence in sentences_raw:
        line = sentence.split('\n')   #separate lines
        for word in line:
            word_list = word.split()     #word in individual list of length = 1
            if len(word_list) > 0:
                word_string = word_list[0]       #getting string insted of list
                testing_processed.append(word_string)  #list of word as strings

#feature extraction
final_data = feature_extraction(testing_processed)
vectors = dicVectorizer.transform(final_data)

#preddictions for unlabeled data
predictions_for_testing = classification_svm.predict(vectors)

#formatting output
tag_result = encoder.inverse_transform(predictions_for_testing)     #resumes with old instead of creating new instace

print(tag_result[:5])

with (open('svm_output.txt','w') as output_for_svm):

    for word, tag in zip(testing_processed, tag_result):
        output_for_svm.write(f'{word} {tag}\n')      #write word tag lines
        
