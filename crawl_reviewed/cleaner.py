from pathlib import Path
import re

pathlist = Path("/Users/russellmoore/Documents/GitHub/tosback2/crawl_reviewed").rglob('*Alexa Terms of Use.txt')
#pathlist2 = Path("/Users/russellmoore/Documents/GitHub/tosback2/crawl_reviewed").rglob('*Google Terms of Use.txt')
# use papthlist to specify what documents to collect


"""
TO DO:
add second file path to the list and compare sentences between the two files and write the scores to compare.txt
"""
file_list = []
for path in pathlist:
     file_list.append(path)


print(len(file_list))
print(file_list)

amazontos = []

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

for file in file_list:
    with open(file) as file:
        for tag in file:
            amazontos.append(cleanhtml(tag))

import nltk
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords

documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(amazontos)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(amazontos[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

import csv
line = []

data = ["ID", "Vendor", "Text"]

with open("Amazon.csv", "w") as csv_file:
    writer = csv.writer(csv_file, delimiter = ",")
    writer.writerow(data)
    for id, text in enumerate(documents):
        if len(text) > 3:
            line = id, documents[0], text
            writer.writerow(line)
        else:
            pass

with open("compare.csv", "w") as csv_file:
    writer = csv.writer(csv_file, delimiter = ",")
    writer.writerow(data)
    for id, text in enumerate(documents):
        if len(text) > 3:
            line = id, documents[0], text
            writer.writerow(line)
        else:
            pass

from difflib import SequenceMatcher

a = documents[8]
b = documents[9]

ratio = SequenceMatcher(None, a, b).ratio()
print("{} \n{} \n {}".format(a,b,ratio))

#SequenceMatcher compares sentences based on word use but not the location of each word

# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidfconverter = TfidfVectorizer(max_features=1500, min_df=1, max_df=0.7, stop_words=stopwords.words('english'))
# X = tfidfconverter.fit_transform(documents).toarray()
# print(X)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, text, test_size=0.2, random_state=0)
#
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
# classifier.fit(X_train, y_train)
#
# y_pred = classifier.predict(X_test)
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print(accuracy_score(y_test, y_pred))
