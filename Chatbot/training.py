import random
import json
import pickle
import numpy as np

import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer= WordNetLemmatizer()# call the constryctor and learn pickle serialization

intents=json.loads(open("intents.json").read())

words=[]
classes=[]
documents=[]
ignore_letters=["?","!",".",","]

for intent in intents['intents']:
    for patterns in intent["patterns"]:
        word_list=nltk.word_tokenize(patterns)
        words.extend(word_list)#extend instead of append since extend means taking the content and appending to a list but appending mean taking a list and appending to a list
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
#print(documents)
           
words=[lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words=sorted(set(words))

classes=sorted(set(classes))

print(classes)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training=[]
output_empty=[0]*len(classes)

#preparing training data
for document in documents:
    bag=[]
    word_patterns=document[0]
    word_patterns=[lemmatizer.lemmatize(word.lower()) for word in word_patterns ]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row=list(output_empty)#copying not type casting
    output_row[classes.index(document[1])]=1
    training.append(bag+output_row)#imp

random.shuffle(training)
training=np.array(training)

train_x=training[:,:len(words)]
train_y=training[:,len(words):]

#building neural network
model=Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),),activation='relu'))#rectified linear unit /(check capital letter very important)
model.add(Dropout(0.5))#overfitting
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation="softmax"))

sgd=SGD(lr=0.01,decay=1e-6, momentum=0.9,nesterov=True)#lr=learing rate
model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=['accuracy'])

hist=model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save('chatbot_model.h5',hist)#h5 file meaning
print('done')