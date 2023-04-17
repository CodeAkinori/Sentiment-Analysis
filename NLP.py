import numpy as np # Linear algebra
import pandas as pd # Data processing
import nltk
import re # For Regex
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
import pickle

# Import DataSet from Pandas
data = pd.read_csv(r'dataset/IMDB-Dataset.csv')


# Remove HTML Tags and other with Regex rule
def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', text)



def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ''
        return rem



def to_lower(text):
    return text.lower()


nltk.download('stopwords')
nltk.download('punkt')

def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]


def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])


# Creating the model BOW (Bag of Words)

X = np.array(data.iloc[:,0].values)
y = np.array(data.sentiment.values)
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(data.review).toarray()
print("X.shape = ",X.shape)
print("y.shape = ",y.shape)


# Train test split (ML)

trainx, testx, trainy, testy = train_test_split(X,y,test_size=0.2, random_state=9)
print('Train shapes : X = {}, y = {}'.format(trainx.shape, trainy.shape))
print('Test shapes : X = {}, y = {}'.format(testx.shape, testy.shape))

# Defining the models and Training them

gnb,mnb,bnb = GaussianNB(),MultinomialNB(alpha=1.0, fit_prior=True),BernoulliNB(alpha=1.0, fit_prior=True)
gnb.fit(trainx,trainy)
mnb.fit(trainx,trainy)
bnb.fit(trainx,trainy)

# Predicion and accuracy metrics to choose best model

ypg = gnb.predict(testx)
ypm = gnb.predict(testx)
ypb = gnb.predict(testx)

print('Gaussian = ', accuracy_score(testy,ypg))
print('Multinomial = ',accuracy_score(testy,ypm))
print('Bernoulli = ',accuracy_score(testy,ypb))

pickle.dump(bnb,open('modell.pkl', 'wb'))


# Utilizing the predictive machine for analise of things


