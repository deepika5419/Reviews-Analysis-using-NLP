# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:33:13 2019

@author: Deepika
"""

#load data

import pandas as pd
import numpy as np

# for advanced visualizations
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)
import plotly.figure_factory as ff

# for basic visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


import io
import os
os.chdir("D:\\Python\\Data_Science\\Deep_Learning\\NLP_Learning\\Amazon_Alexa_Review_Comptition")

Alexa_train=pd.read_csv("D:\\Python\\Data_Science\\Deep_Learning\\NLP_Learning\\Amazon_Alexa_Review_Comptition\\amazon_alexa.tsv",sep='\t',header=0,encoding='latin1')
print(Alexa_train.head())
Alexa_train.shape

Alexa_train.describe()

### checking if there is any null data or not
Alexa_train.isnull().any().any()

##Describing the data according to the Length of the reviews
## adding a length column for analyzing the length of the reviews

Alexa_train['length']=Alexa_train['verified_reviews'].apply(len)
Alexa_train.groupby('length').describe().sample(10)


## Describing data according to the rating
Alexa_train.groupby('rating').describe()
## Describing data according to the Feedback

Alexa_train.groupby('feedback').describe()

### Data Visulization

## Distribution of ratings of Alexa

ratings=Alexa_train['rating'].value_counts()
label_rating=ratings.index
size_rating=ratings.values

Colors=['pink','lightblue','aqua','gold','crimson']
rating_Piechart=go.Pie(labels=label_rating,
                       values=size_rating,
                       marker=dict(colors=Colors),
                       name='Alexa',hole=0.3)

df=[rating_Piechart]
layout=go.Layout(title='Distribution of Ratings for Alexa')
fig=go.Figure(data=df,layout=layout)
#fig=go.Figure(data=[go.Pie(labels=label_rating,values=size_rating,marker=dict(colors=Colors),name='Alexa',hole=0.3)],layout=layout)

py.offline.plot(fig,filename='Pie_Chart1.html')

### BAR PLOT
#linespace

color=plt.cm.copper(np.linspace(0,1,15))
Alexa_train['variation'].value_counts().plot.bar(color=color,figsize=(15,9))
plt.title('Distribution variation in ALexa',fontsize=20)
plt.xlabel('variations')
plt.ylabel('count')
plt.show()


###Distribution of Feedbacks for Alexa

feedbacks=Alexa_train['feedback'].value_counts()
labels_feedback=feedbacks.index
size_feedback=feedbacks.values

colors=['yellow','lightgreen']
feedback_piechart=go.Pie(labels=labels_feedback,
                         values=size_feedback,
                         marker=dict(colors=colors),
                         name='Alexa',hole=0.3)
df2=[feedback_piechart]
layout=go.Layout(title='Distribition Of Feedback for Alexa')
fig=go.Figure(data=df2,layout=layout)

py.offline.plot(fig,filename='Pie_Feedback.html')

Alexa_train['length'].value_counts().plot.hist(color='skyblue',figsize=(15,5),bin=50)
plt.title('Distribution of Length in Reviews')
plt.xlabel('lengths')
plt.ylabel('count')
plt.show()


##Check some of the reviews according to thier lengths

Alexa_train[Alexa_train['length']==1]['verified_reviews'].iloc[0]
Alexa_train[Alexa_train['length']==21]['verified_reviews'].iloc[0]
Alexa_train[Alexa_train['length']==50]['verified_reviews'].iloc[0]
Alexa_train[Alexa_train['length']==150]['verified_reviews'].iloc[0]

## Variations Vs Ratings
plt.rcParams['figure.figsize']=(15,9)
plt.style.use('fivethirtyeight')
sns.boxenplot(Alexa_train['variation'],Alexa_train['rating'],paltte='spring')
plt.title("Variation vs Ratings")
plt.xticks(rotation=90)
plt.show()

plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('fivethirtyeight')
sns.swarmplot(Alexa_train['variation'], Alexa_train['length'], palette = 'deep')
plt.title("Variation vs Length of Ratings")
plt.xticks(rotation = 90)
plt.show()

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize']=(12,7)
plt.style.use('fivethirtyeight')
sns.violinplot(Alexa_train['feedback'], Alexa_train['rating'], palette = 'cool')
plt.title("feedback wise Mean Ratings")
plt.show()


warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 7)
plt.style.use('fivethirtyeight')

sns.boxplot(Alexa_train['rating'], Alexa_train['length'], palette = 'Blues')
plt.title("Length vs Ratings")
plt.show()

##The Bar plot represents the most frequnt words in the reviews so that we can get a rough idea about the reviews and what people think of the product.
from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(Alexa_train.verified_reviews)
sum_words = words.sum(axis=0)


words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

plt.style.use('fivethirtyeight')
color = plt.cm.ocean(np.linspace(0, 1, 20))
frequency.head(20).plot(x='word', y='freq', kind='bar', figsize=(15, 6), color = color)
plt.title("Most Frequently Occuring Words - Top 20")
plt.show()

###Plotting a wordscloud for the Words to see all the words, The Larger the words the larger is the frequency for that word.

from wordcloud import WordCloud

wordcloud = WordCloud(background_color = 'lightcyan', width = 2000, height = 2000).generate_from_frequencies(dict(words_freq))

plt.style.use('fivethirtyeight')
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(wordcloud)
plt.title("Vocabulary from Reviews", fontsize = 20)
plt.show()

###
plt.rcParams['figure.figsize']=(12,7)
plt.style.use('fivethirtyeight')
sns.stripplot(Alexa_train['feedback'],Alexa_train['length'],palette='Reds')
plt.title("Feedback Vs Length")
plt.show()

##Ratings vs Length vs Variation

trace=go.Scatter3d(x=Alexa_train['length'],
                   y=Alexa_train['rating'],
                   z=Alexa_train['variation'],
                   name='Amezon Alexa',
                   mode='markers',
                   marker=dict(size=10,color=Alexa_train['rating'],colorscale='Viridis',))

df3=[trace]

layout=go.Layout(title='Length vs Variation vs Ratings',margin=dict(l=0,r=0,b=0,t=0))
fig=go.Figure(data=df3,layout=layout)
#iplot(fig)
py.offline.plot(fig,filename='Variation.html')

### Spacy

import spacy
nlp=spacy.load('C:\\Users\Deepika\\Anaconda3\\Lib\\site-packages\\en_core_web_sm\\en_core_web_sm-2.2.0')

### If SPCY is not WOrking in that case run below
#!pip install -U spacy download en_core_web_sm
#!pip install -U spacy download en_core_web_sm

def explain_text_entities(text):
    doc=nlp(text)
    for ent in doc.ents:
        print(f'Entity:{ent},Label:{ent.label_},{spacy.explain(ent.label_)}')
        
for i in range(15,50):
    one_sentence=Alexa_train['verified_reviews'][1]
    doc=nlp(one_sentence)
    spacy.displacy.render(doc,style='ent',jupyter=True)
    
# cleaning the texts
# importing the libraries for Natural Language Processing
    
    
import re
import nltk
from nltk.corpus import stopwords   
from nltk.stem.porter import PorterStemmer

corpus=[]
for i in range(0,3150):
    review=re.sub('[a-zA-Z]','',Alexa_train['verified_reviews'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word)for word in review if not word in set(stopwords.words('english'))]
    review=''.join(review)
    corpus.append(review)
    
### Creating bag of word

from sklearn.feature_extraction.text import TfidfVectorizer

cv=CountVectorizer(max_features=2500)
x=cv.fit_transform(corpus).toarray()
y=Alexa_train.iloc[:,4].values
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=15)


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
x_train=mm.fit_transform(x_train)
x_test=mm.transform(x_test)



### Modelling

##Random Forest


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

print("Training Accuracy :",model.score(x_train,y_train))
print("Testing Accuracy :",model.score(x_test,y_test))

cm=confusion_matrix(y_test,y_pred)
print(cm)


### Applying K fold cross validation

from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=model,X=x_train,y=y_train,cv=10)

print("Accuracy :",accuracies.mean())
print("Standard Variance :",accuracies.std())


params = {
    'bootstrap': [True],
    'max_depth': [80, 100],
    'min_samples_split': [8, 12],
    'n_estimators': [100, 300]
}


# applying grid search with stratified folds

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

cv_object=StratifiedKFold(n_splits=2)
grid=GridSearchCV(estimator=model,param_grid=params,cv=cv_object,verbose=0,return_train_score=True)
grid.fit(x_train,y_train.ravel())
print("Best Parameter Combination :{}".format(grid.best_params_))
print("Mean Cross Validation Accuracy-Train Set :{}".format(grid.cv_results_['mean_train_score'].mean()*100))
print("Mean Cross Validation Accuracy - Validation Set:{}".format(grid.cv_results_['mean_test_score'].mean()*100))

from sklearn.metrics import accuracy_score
print("Accuracy Score for test Set :",accuracy_score(y_test,y_pred))


















    




    
        
















