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
data = pd.read_csv('IMDB-Dataset.csv')


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


# Train test split (ML)

trainx, testx, trainy, testy = train_test_split(X,y,test_size=0.2, random_state=9)

# Defining the models and Training them

gnb,mnb,bnb = GaussianNB(),MultinomialNB(alpha=1.0, fit_prior=True),BernoulliNB(alpha=1.0, fit_prior=True)
gnb.fit(trainx,trainy)
mnb.fit(trainx,trainy)
bnb.fit(trainx,trainy)

# Predicion and accuracy metrics to choose best model

ypg = gnb.predict(testx)
ypm = gnb.predict(testx)
ypb = gnb.predict(testx)

pickle.dump(bnb,open('modell.pkl', 'wb'))


# Utilizing inputs

rev = 'A wonderful little production. The filming technique is very unassuming- very old-time-BBC fashion and gives a comforting, and sometimes discomforting, sense of realism to the entire piece. The actors are extremely well chosen- Michael Sheen not only "has got all the polari" but he has all the voices down pat too! You can truly see the seamless editing guided by the references to Williams diary entries, not only is it well worth the watching but it is a terrificly written and performed piece. A masterful production about one of the great masters of comedy and his life. The realism really comes home with the little things: the fantasy of the guard which, rather than use the traditional dream techniques remains solid then disappears. It plays on our knowledge and our senses, particularly with the scenes concerning Orton and Halliwell and the sets particularly of their flat with Halliwells murals decorating every surface are terribly well done.'

f1 = clean(rev)
f2 = is_special(f1)
f3 = to_lower(f2)
f4 = rem_stopwords(f3)
f5 = stem_txt(f4)

bow,words = [], word_tokenize(f5)

for word in words:
    bow.append(words.count(word))

word_dict = cv.vocabulary_
pickle.dump(word_dict, open('bow.pkl', 'wb'))

inp = []
for i in word_dict:
    inp.append(f5.count(i[0]))
y_pred = bnb.predict(np.array(inp).reshape(1, 1000))

print(y_pred)