# Atiwat Onsuwan 1802514 and Fazal Jamadar 1802447
# Submitted as part of CE807-7-SP: Text Analytics assignment 2
# Acknowledgement : https://www.kaggle.com/dundee2002/rotten-tomatoes-movie-reviews-w-glove-lstm
import numpy as np
import pandas as pd
import warnings
from nltk import re
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.metrics import accuracy_score, classification_report
import keras.backend as K
from keras.preprocessing import sequence

warnings.filterwarnings("ignore")

# Load training and test set
df = pd.read_csv('train.tsv', delimiter='\t')
test_df = pd.read_csv('test.tsv', delimiter='\t')

## Features extraction
# store column Phrase from training set for training purpose
X = df['Phrase']
# store column Phrase from test set for prediction
test = test_df['Phrase']
# store column Sentiment from training set (Classes) for training purpose
Y = to_categorical(df['Sentiment'])

# Get the number of all classes from training set
num_classes = df['Sentiment'].nunique()
seed = 128  # fix random seed for reproducing for train and test splitting
np.random.seed(seed)

# Spilt train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.20,
                                                    stratify=Y,
                                                    random_state=seed)

### Data pre-processing
# After we check the word cloud in our features we decided not to remove stopwords
# Because some of the word in stopwords are useful in this sentiment problem

# Tokenize features of training and test sets
X_train = X_train.str.lower()
X_train = X_train.apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
X_test = X_test.str.lower()
X_test = X_test.apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
test = test.str.lower()
test = test.apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

# define max features for tokenizer (max number of word in training set after tokenize)
max_features = 13733
# define tokenizer object
tokenizer = Tokenizer(num_words=max_features)
# fit the features in divided set using tokenizer object above
tokenizer.fit_on_texts(list(X_train))
# turn features to numerical form
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
test = tokenizer.texts_to_sequences(test)

# get the max length of the word in phrase column
max_lenght = max([len(s.split()) for s in df['Phrase']])

# pad the features sequence to the form which LSTM can understand
X_train = sequence.pad_sequences(X_train, maxlen=max_lenght)
X_test = sequence.pad_sequences(X_test, maxlen=max_lenght)
test = sequence.pad_sequences(test, maxlen=max_lenght)

# define the batch size
batch_size = 128
# define number of epoch (training loop)
epochs = 10
# define embedded dim number
embed_dim = 300


# add model setting (classifier and hidden layer)
def model_setting(num_features, dim):
    np.random.seed(seed)
    K.clear_session()
    model = Sequential()
    model.add(Embedding(num_features, dim, input_length=max_lenght))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    # model.add(LSTM(56, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # display model summary
    print(model.summary())
    return model


# train the model using model_setting function above
def train(model):
    early_stopping = EarlyStopping(min_delta=0.001, mode='max', monitor='val_acc', patience=2)
    callback = [early_stopping] # prevent the model from being prune
    model.fit(X_train, y_train,
              epochs=epochs, batch_size=batch_size, verbose=2, callbacks=callback)


# evaluate the performance of the trained over divided training set and display classification report
def evaluate():
    y_pred_test = model.predict_classes(X_test, batch_size=batch_size, verbose=2)
    print('Accuracy:\t{:0.1f}%'.format(accuracy_score(np.argmax(y_test, axis=1), y_pred_test) * 100))
    print('\n')
    print(classification_report(np.argmax(y_test, axis=1), y_pred_test))


# use our trained model to predict class of test set then write into csv file
def predict_test_set():
    pred_test = model.predict_classes(test, batch_size=batch_size, verbose=2)
    # Write prediction if test set in csv file for Kaggle submission
    with open('submission.csv', 'w') as csvfile:
        csvfile.write('PhraseId,Sentiment\n')
        for i, j in zip(test_df['PhraseId'], pred_test):
            csvfile.write('{}, {}\n'.format(i, int(j)))


# call all functions
model = model_setting(max_features, embed_dim)
train(model)
evaluate()
predict_test_set()
