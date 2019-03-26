from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()

example_text = "Hello my name is Oscar steal stole stealing"
stop_words = set(stopwords.words("english"))

words = words_tokenize(example_text)

filtered_sentence = [w for w in words if not w in stop_words]
print(filtered_sentence)

for w in filtered_sentence:
  print(ps.stem(w))

