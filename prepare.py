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
