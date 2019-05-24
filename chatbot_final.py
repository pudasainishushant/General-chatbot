
# coding: utf-8

# In[36]:


import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import pandas as pd
import pickle
import random
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


# In[37]:


with open('intents.json') as json_data:
    intents = json.load(json_data)


# In[38]:


words = []
classes = []
documents = []
ignore_words = ['?']
responsess = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
    for response in intent['responses']:
        responsess.extend(response)

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# sort classes
classes = sorted(list(set(classes)))


# In[39]:


# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word - create base word, in attempt to represent related words
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])


# In[40]:


#Build model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fit model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)


# In[41]:


pickle.dump(model, open("katana-assistant-model.pkl", "wb"))


# In[42]:


pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y, 'responses':responsess}, open( "katana-assistant-data.pkl", "wb" ) )


# In[43]:


data = pickle.load( open( "katana-assistant-data.pkl", "rb" ) )
words = data['words']
classes = data['classes']
responsess = data['responses']


# In[44]:


global graph
graph = tf.get_default_graph()

with open(f'katana-assistant-model.pkl', 'rb') as f:
    model = pickle.load(f)


# In[45]:


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1

    return(np.array(bag))
def classify_local(sentence):
    ERROR_THRESHOLD = 0.25
    
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    # return tuple of intent and probability
    
    return return_list


	# In[73]:
for i in range(20):
	user_query = input("User query : ")
	result = classify_local(user_query)
	intent = result[0][0]


	# In[78]:


	for i in range(len(intents['intents'])):
	    if intents['intents'][i]['tag']==intent:
	        print("Response: ")
	        print(intents['intents'][i]['responses'][0])

