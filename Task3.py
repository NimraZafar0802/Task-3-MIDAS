## IMPORTS


# Data Manipulation
import numpy as np
import pandas as pd

# Data dtypes
import json
import re
import string
from pandas.io.json import json_normalize


# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt


# Textual data manipulation
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer



df_original=pd.read_csv('C:/Users/Nimra and Ubaid/Desktop/flipkart_com-ecommerce_sample - flipkart_com-ecommerce_sample - flipkart_com-ecommerce_sample.csv')
df = df_original


## Data Understanding and manipulation
print(df.head())
print(df.dtypes)

# Checking the overall data info
print(df.info())


# Checking if we have any null values
print(df.isnull().any())

# Algorithms
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import metrics



# Checking the target variable size and details
print(df.groupby('product_category_tree').size())


# Understanding the data using describe()
print(df.describe())
print(df.head())



# Removing the unwanted variables
del df['uniq_id']
del df['crawl_timestamp']
del df['product_url']
del df['product_name']
del df['pid']
del df['retail_price']
del df['discounted_price']
del df['image']
del df['is_FK_Advantage_product']
del df['product_rating']
del df['overall_rating']
del df['brand']
del df['product_specifications']


print(df.head())


# As we have text data we will  use NLP techniques
# Label Encoding

df["product_category_tree"] = df["product_category_tree"].astype('category')
df["product_category_tree"] = df["product_category_tree"].cat.codes
print(df.dtypes)
print(df.head())

## Data Visualization

# Understanding data using pairplot
sns.pairplot(df)


## Implementing Machine Learning Model

## Splitting the data into test and train sets

# We will split the data into 70/30 split

from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.33)


print(test.head())
print(train.head())

length_train = train['description'].str.len()
length_test = test['description'].str.len()
plt.hist(length_train, label="train_payload")
plt.hist(length_test, label="test_payload")
plt.legend()
plt.show()



# Cleaning the text data

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    #text = text.lower()
    text = re.sub('\[.*?\]', '',str(text))
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text



# Applying the cleaning function to both test and train datasets
train['description'] =train['description'].apply(lambda x: clean_text(x))
test['description'] = test['description'].apply(lambda x: clean_text(x))

# updated text
print(train['description'].head())


# We will turn the sentence into smaller chuncks using tokenization to remove
# stopwords and lemmatization techniques

tokenizer=nltk.tokenize.RegexpTokenizer(r'\w+')
train['description'] = train['description'].apply(lambda x:tokenizer.tokenize(x))
test['description'] = test['description'].apply(lambda x:tokenizer.tokenize(x))
print(train['description'].head())


# removing stopwords
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words
train['description'] = train['description'].apply(lambda x : remove_stopwords(x))
test['description'] = test['description'].apply(lambda x : remove_stopwords(x))
print(test.head())


# lemmatization
lem = WordNetLemmatizer()
def lem_word(x):
    return [lem.lemmatize(w) for w in x]




train['description'] = train['description'].apply(lem_word)
test['description'] = test['description'].apply(lem_word)
# Combining all the text into one

def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

train['description'] = train['description'].apply(lambda x : combine_text(x))
test['description'] = test['description'].apply(lambda x : combine_text(x))
train['description']
train.head()



# Label encoding all the values of the text data using count vectorizer

count_vectorizer = CountVectorizer()
train_vector = count_vectorizer.fit_transform(train['description'])
test_vector = count_vectorizer.transform(test['description'])
print(train_vector[0].todense())

# Implementing Tf-IDF also onto our train data

tfidf = TfidfVectorizer(min_df = 2,max_df = 0.5,ngram_range = (1,2))
train_tfidf = tfidf.fit_transform(train['description'])
test_tfidf = tfidf.transform(test['description'])

# As the data is categorical and have 1000 rows we can use KNN to build our model
# Implementing knn and checking the accuracy on the train
# data using cross_val_score

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=2)

scores_vector = model_selection.cross_val_score(classifier, train_vector, train['description'], cv = 5, scoring = "f1_micro")
print("score:",scores_vector)
scores_tfidf = model_selection.cross_val_score(classifier, train_tfidf, train['description'], cv = 5, scoring = "f1_micro")
print("score of tfidf:",scores_tfidf)



# Predicting the test data using the model

classifier.fit(train_tfidf, train['product_category_tree'])
y_pred = classifier.predict(test_tfidf)
test['predict'] = y_pred



# Checking Precision, Recall, and giving the
# classification report for the prediction


print(classification_report(test['product_category_tree'], test['predict']))
print(confusion_matrix(test['product_category_tree'], test['predict']))
print(accuracy_score(test['product_category_tree'], test['predict']))
print("Precision:",metrics.precision_score(test['product_category_tree'], test['predict']))
print("Recall:",metrics.recall_score(test['product_category_tree'], test['predict']))


