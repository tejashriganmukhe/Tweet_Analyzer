# Importing Packages
import warnings
warnings.filterwarnings("ignore") 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
import urllib.request
import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import svm

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow import keras
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
#import pickel
# Reading Data
df = pd.read_csv(r'D://Python-Project-2022//Twitter_Sentiment_Analyasis//Book2.csv')

# Data Sample
a = df.sample(5)
print(a)

# Checking for NA Values
b = df.isnull().sum()
print(b)

# Distribution of different classes in sentiment
def count_values_in_column(data,feature):
    total=data.loc[:,feature].value_counts(dropna=False)
    percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
    return pd.concat([total,percentage],axis=1,keys=["Total","Percentage"])
c = count_values_in_column(df,"category")

print(c)

# Segrating based on different sentiments
df_negative = df[df["category"]==2]
df_positive = df[df["category"]==1]
df_neutral = df[df["category"]==0]

# create data for Pie Chart
plt.figure(figsize=(13, 8), dpi=80)
pichart = count_values_in_column(df,"category")
names= ["Positive","Neutral","Negative","Nan"]
size=pichart["Percentage"]
 
# Create a circle for the center of the plot
# my_circle=plt.Circle( (0,0), 0.5, color='white')
# plt.pie(size, labels=names, colors=['green','blue','red',"yellow"])
# p=plt.gcf()
# p.gca().add_artist(my_circle)
# plt.show()

# Function to Create Wordcloud
def create_wordcloud(text,path):
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",
    max_words=3000,
    stopwords=stopwords,
    random_state=42,
    width=900, height=500,
    repeat=True)
    wc.generate(str(text))
    wc.to_file(path)
    print("file Saved Successfully")
    # path=path
    # display(Image.open(path))


# Wordcloud for all tweets
plt.figure(figsize=(15, 8), dpi=80)
create_wordcloud(df['clean_text'].values,"all.png")

# Wordcloud for only positive tweets
plt.figure(figsize=(15, 8), dpi=80)
create_wordcloud(df_positive['clean_text'].values,"positive.png")

# Wordcloud for only negative tweets
plt.figure(figsize=(15, 8), dpi=80)
create_wordcloud(df_negative['clean_text'].values,"negative.png")

# Wordcloud for only neutral tweets
plt.figure(figsize=(15, 8), dpi=80)
create_wordcloud(df_neutral['clean_text'].values,"neutral.png")

#df['clean_text'].value_counts().plot.bar(color = 'pink', figsize = (6, 4))


df['category'].value_counts().plot.bar(color = 'blue', figsize = (6, 4))


# Stemming
ps = PorterStemmer()
# Initializing Lists
corpus = []
words = []
for i in range(0, len(df)):
    # Removing characters other than letters
    review = re.sub('[^a-zA-Z]', ' ', str(df["clean_text"][i]))
    # Lowering the case all the text
    review = review.lower()
    # Splitting into words
    review = review.split()
    # Applying Stemming
    stemmed = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    # Joining words
    review = ' '.join(stemmed)
    # Appending all tweets to a list after preprocessing
    corpus.append(review)
    # Appending all words for word embeddings
    words.append(stemmed)
# Corpus sample
corpus[1:10]

# Length 
print("Legth of Corpus:",len(corpus))

# Updating created corpus in our dataframe
df["clean_text"] = corpus
# Dropping NA Values and resetting index
df = df.dropna()
df = df.reset_index()
# Checking for NA Values after corpus updations
df.isna().sum()


# Exporting stemmed sentences
df[["clean_text","category"]].to_csv("stemmed.csv",index = False)
# Loading the stemmed sentences
df_stemmed = pd.read_csv("stemmed.csv")
# Extracting corpus 
corpus = list(df_stemmed["clean_text"])


# Applying TFIDF Vectorization
tfidf = TfidfVectorizer(max_features=5000,ngram_range=(1,3))
X_tfidf = tfidf.fit_transform(df["clean_text"]).toarray()
# Independent Variable
X = df_stemmed["clean_text"]
# Dependent Varible
Y = df_stemmed["category"]
df_tfidf = pd.DataFrame(X_tfidf,columns = tfidf.get_feature_names())
df_tfidf["output"] = Y
df_tfidf.head()




from textblob import TextBlob

## create function to subjectivity
def getSubjectivity(df_tfidf):
    return TextBlob(df_tfidf).sentiment.subjectivity
## create function to get polarity
def getPolarity(df_tfidf):
    return TextBlob(df_tfidf).sentiment.polarity
## create two new column
df['Subjectivity']=df['clean_text'].apply(getSubjectivity)
df['Polarity']=df['clean_text'].apply(getPolarity)
## show new Dataframe
df



## create function to compute positive, negative and neutral analysis
def getAnalysis(score):
    if score<0:
        return 2
    elif score==0:
        return 0
    else:
        return 1
df['category'] = df['Polarity'].apply(getAnalysis)
print(df)



for index,row in df.iterrows():
    if row['category']==1:
        plt.scatter(row['Polarity'],row['Subjectivity'],color='green')
    elif row['category']==2:
        plt.scatter(row['Polarity'],row['Subjectivity'],color='red')
    else:
        plt.scatter(row['Polarity'],row['Subjectivity'],color='blue')
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()



# from sklearn import svm as SVC


# # Train test Split
# X_train_tfidf,X_test_tfidf,Y_train_tfidf,Y_test_tfidf = train_test_split(X_tfidf,Y,test_size=0.33,random_state = 27)
# ######Multinomial Naive Bayes
# # Initializing Model
# classfier_tfidf = SVC(kernal = 'linear')
# # Fitting data
# classfier_tfidf.fit(X_train_tfidf,Y_train_tfidf)
# # Prediction on test data
# Y_pred_tfidf = classfier_tfidf.predict(X_test_tfidf)



# acc_tfidf = accuracy_score(Y_test_tfidf,Y_pred_tfidf)
# classification_tfidf = classification_report(Y_test_tfidf,Y_pred_tfidf)
# confusion_matrix_tfidf = confusion_matrix(Y_test_tfidf,Y_pred_tfidf)
# print("For SVM: \n")
# print(" \n Accuracy : ",acc_tfidf,"\n","Classification report \n",classification_tfidf,"\n","Confusion matrix \n",confusion_matrix_tfidf)
