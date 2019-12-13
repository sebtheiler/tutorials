# Load data
import pandas as pd

col_names = ['sentiment','id','date','query_string','user','text']
data_path = 'training.1600000.processed.noemoticon.csv'

tweet_data = pd.read_csv(data_path, header=None, names=col_names, encoding="ISO-8859-1").sample(frac=1) # .sample(frac=1) shuffles the data
tweet_data = tweet_data[['sentiment', 'text']] # Disregard other columns
print(tweet_data.head())

# Preprocess function
import re
allowed_chars = ' AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789~`!@#$%^&*()-=_+[]{}|;:",./<>?'
punct = '!?,.@#'
maxlen = 280

def preprocess(text):
    return ''.join([' ' + char + ' ' if char in punct else char for char in [char for char in re.sub(r'http\S+', 'http', text, flags=re.MULTILINE) if char in allowed_chars]])[:maxlen]

# Apply preprocessing
tweet_data['text'] = tweet_data['text'].apply(preprocess)

# Put __label__ in front of each sentiment
tweet_data['sentiment'] = '__label__' + tweet_data['sentiment'].astype(str)

# Save data
import os

# Create directory for saving data if it does not already exist
data_dir = './processed-data'
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

# Save a percentage of the data (you could also only load a fraction of the data instead)
amount = 0.125

tweet_data.iloc[0:int(len(tweet_data)*0.8*amount)].to_csv(data_dir + '/train.csv', sep='\t', index=False, header=False)
tweet_data.iloc[int(len(tweet_data)*0.8*amount):int(len(tweet_data)*0.9*amount)].to_csv(data_dir + '/test.csv', sep='\t', index=False, header=False)
tweet_data.iloc[int(len(tweet_data)*0.9*amount):int(len(tweet_data)*1.0*amount)].to_csv(data_dir + '/dev.csv', sep='\t', index=False, header=False)

# Memory management
del tweet_data
import gc; gc.collect()

# Load the data into Corpus format
from flair.data_fetcher import NLPTaskDataFetcher
from pathlib import Path

corpus = NLPTaskDataFetcher.load_classification_corpus(Path(data_dir), test_file='test.csv', dev_file='dev.csv', train_file='train.csv')

# Make label dictionary
label_dict = corpus.make_label_dictionary()

# Load embeddings
from flair.embeddings import WordEmbeddings, FlairEmbeddings

word_embeddings = [WordEmbeddings('glove'),
#                    FlairEmbeddings('news-forward'),
#                    FlairEmbeddings('news-backward')
                  ]

# Initialize embeddings
from flair.embeddings import DocumentRNNEmbeddings

document_embeddings = DocumentRNNEmbeddings(word_embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256)

# Create model
from flair.models import TextClassifier

classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

# Create model trainer
from flair.trainers import ModelTrainer

trainer = ModelTrainer(classifier, corpus)

# Train the model
trainer.train('model-saves',
              learning_rate=0.1,
              mini_batch_size=32,
              anneal_factor=0.5,
              patience=8,
              max_epochs=200)

# Load the model and make predictions
from flair.data import Sentence

classifier = TextClassifier.load('model-saves/final-model.pt')

pos_sentence = Sentence(preprocess('I love Python!'))
neg_sentence = Sentence(preprocess('Python is the worst!'))

classifier.predict(pos_sentence)
classifier.predict(neg_sentence)

print(pos_sentence.labels, neg_sentence.labels)