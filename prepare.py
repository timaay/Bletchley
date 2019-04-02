import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
#nltk.download('punkt')
import pandas as pd
import numpy as np
#nltk.download('wordnet')
import itertools
import string

ps = PorterStemmer()

example_text = "Hello my name is Oscar steal stole stealing"
stop_words = set(stopwords.words("english"))

words = words_tokenize(example_text)

filtered_sentence = [w for w in words if not w in stop_words]
print(filtered_sentence)

for w in filtered_sentence:
  print(ps.stem(w))
  
  
################ Dataframe ####################
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()


text1 = "Hello, my name is Oscar steal stole stealing"
text2 = "This is an example text"
text3 = "We need to do text analysis on these texts"
stop_words = set(stopwords.words("english"))


df = pd.DataFrame(data = [text1, text2, text3], columns=["review"])

##### create a new column "proc_words" ############
df["proc_words"] = df["review"].str.lower() # make to lower
df["proc_words"] = df["proc_words"].apply(str.split) # split the string Note: tokenize doesn't have the punctuation as different element
df["proc_words"] = df["proc_words"].apply(lambda x: [item.translate(str.maketrans("","", string.punctuation)) for item in x]) # remove punctuation
df["proc_words"] = df["proc_words"].apply(lambda x: [item for item in x if item not in stop_words]) # Remove stopwords
df["proc_words"] = [" ".join(x) for x in df["proc_words"]] # Join the words (Used to tokenize them later, maybe it's not needed as tokenize
# seems to work the same as str.split()
df["proc_words"] = df["proc_words"].apply(word_tokenize) # Tokenize the words
df["proc_words"] = df["proc_words"].apply(lambda x: [ps.stem(item) for item in x]) # Stem the words

###### a dictionary to for words that we need to count #########
# function to take only the unique and stemmed elements from the dictionary values
def dict_stem(text_list):
    return(list(set([ps.stem(word) for word in text_list])))

# Dictionary of group of words that we need to count how many times they appear on the text
wdict =    {
 "group1": dict_stem(["climate", "environment", "sustainable", "sustainability", "renewable"]),
 "group2": dict_stem(["roosendaal"]),
 "group3": dict_stem(["rotterdam", "rotterdam"])
}


##### import actual files ########
import os
import re

path = 'D:\\filings_clean_withscore' # Change this to your directory where the files are

files = []
# r=root, d=directories, f = files
# for loop to take the file names
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))
scores = []
names = []
# For loop to extract the ESG scores and name
for f in files:
    scores.append(re.findall("_\d_\d_\d_\d", f))
    names.append(re.findall("text_10k_(.*)(?=.txt)", f))

# Make the names into a list (previously it was a list of lists)
names_joined = [' '.join(x) for x in names]

text_dict = {}

# Create a dictionary with keys as the file names (only the unique identifier for each company)
# and as values, the text
for file, name in zip(files, names_joined):
    with open(file, "r") as myfile:
        text_dict[name] = myfile.read().replace("\n", " ")

df_text = pd.DataFrame(text_dict.items(), columns=["id", "text"])


@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

import tensorflow as tf 
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# fix random seed for reproducibility
np.random.seed(7)

# create model
model = tf.keras.modelsSequential()
model.add(Dense(12, input_dim=100, activation='relu')) # 100 12 8 4 model
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='sigmoid'))

# model.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# model.fit(x_train, y_train, epochs = 3, batch_size = 8)

# Define custom loss
def custom_loss(layer):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        correct = 0
        for i in range(2):
            if (y_pred[i] == y_true[i):
                                    correct = correct + 1
            if (correct == 0):
                                    correct = 1
        return (y_pred[3] == y_true[3])*(correct)
   
    # Return a function
    return loss
    
# Compile the model
model.compile(optimizer='adam',
              loss=custom_loss(layer), # Call the loss function with the selected layer
              metrics=['accuracy'])


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save('oscar>stelios.model')

new_model = tf.keras.models.load_model('oscar>stelios.model')

predictions = new_model.predict([x_test])

print(predictions)
print (np.argmax(predictions[0]))

