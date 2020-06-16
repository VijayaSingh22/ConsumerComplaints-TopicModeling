# Multi-class Classification and Clustering of Consumer Complaints
# Data: Consumer Complaints to CFPB, 555,957 rows x 18 columns, 171 MB (38 MB zipped)
# Source: https://www.kaggle.com/subhassing/exploring-consumer-complaint-data/data

# Import and preprocess data

import pandas as pd 
df = pd.read_csv('C:/Users/abhatt/Desktop/python/data/Consumer_Complaints.csv', encoding='latin-1')
df.shape                                                    # 555,957 x 18
df.dtypes
df = df[['product', 'company', 'consumer_complaint_narrative']]
df = df.rename(columns = {'consumer_complaint_narrative':'narrative'})
df = df[pd.notnull(df['narrative'])]
df['narrative'] = df['narrative'].str.replace('XXXX','')    # Redacted content
df.shape                                                    # 66,806 x 3
df.head()

import time
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

start_time = time.time()                                    # Takes 32.2 minutes!
clean_text = []
# for text in df['narrative']:
    words = regexp_tokenize(text.lower(), r'[A-Za-z]+')
    words = [w for w in words if len(w)>1 and w not in stopwords.words('english')]
    words = [lemmatizer.lemmatize(w) for w in words]
    clean_text.append(' '.join(words))
print('Elapsed clock time: ', (time.time() - start_time)/60, ' minutes')

len(clean_text)
df['clean_text'] = clean_text
df.head()

# Pickle file for later use

import pickle
with open('C:/Users/abhatt/Desktop/python/Consumer_Compliants_multiclass.pkl', 'wb') as pkl_file:
    pickle.dump(df, pkl_file) 

with open('C:/Users/abhatt/Desktop/python/Consumer_Compliants_multiclass.pkl', 'rb') as pkl_file:
    df_new = pickle.load(pkl_file) 
df_new.shape
df_new.dtypes
df = df_new

# Exploratory data analysis

df['category_id'] = df['product'].factorize()[0]
df.head()
df.groupby('product').narrative.count()

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
df.groupby('product').narrative.count().plot.bar(ylim=0)
plt.show()

# Split data into train and validation and create TF-IDF vectorizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

train_x, valid_x, train_y, valid_y = \
    train_test_split(df['clean_text'], df['product'], \
    test_size=0.2, random_state=42)
encoder = LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    # Regex: '\w{1,}' = 1+ ASCII char, digits, or underscore
tfidf_vect.fit(df['clean_text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# Logistic regression

from sklearn.linear_model import LogisticRegression
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
     penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
     verbose=0, warm_start=False)
model = LogisticRegression().fit(xtrain_tfidf, train_y)

# Compute model accuracy and confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(classification_report(valid_y, model.predict(xvalid_tfidf), \
    target_names=df['product'].unique()))

conf_matrix = confusion_matrix(valid_y, model.predict(xvalid_tfidf))
conf_matrix

# Plot confusion matrix as a heatmap using Seaborn
category_id_df = df[['product', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'product']].values)

import seaborn as sns
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="BuPu",
    xticklabels=category_id_df[['product']].values, 
    yticklabels=category_id_df[['product']].values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Prediction example
new_text = ["This company refuses to provide me verification and validation " + 
    "of debt per my right under the FDCPA. I do not believe this debt is mine."]
text_features = tfidf_vect.transform(new_text)
predictions = model.predict(text_features)
print('Prediction: ', id_to_category[predictions[0]])

# Other classifiers

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(xtrain_tfidf, train_y)

from sklearn.svm import LinearSVC, SVC
model = LinearSVC().fit(xtrain_tfidf, train_y)
# model = SVC().fit(xtrain_tfidf, train_y)                # Takes too long
# Note: LinearSVC uses One-vs-Rest multiclass reduction; SVC uses One-vs-One.
# LinearSVC fits N models (N = number of classes); SVC fits N*(N-1)/2 models.

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(xtrain_tfidf, train_y)


