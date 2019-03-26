from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

example_text = "Hello my name is Oscar steal stole stealing"
stop_words = set(stopwords.words("english"))

words = words_tokenize(example_text)

filtered_sentence = [w for w in words if not w in stop_words]
print(filtered_sentence) 
