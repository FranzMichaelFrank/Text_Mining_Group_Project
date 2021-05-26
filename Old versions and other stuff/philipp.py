from word_mover_distance import model
import torchtext.vocab as vocab
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from scipy.stats import pearsonr
from scipy.stats import kendalltau
import re
import matplotlib.pyplot as plt
from nltk import download
from nltk.corpus import stopwords
import torch
import bert_score
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

path_corpus = "/Users/philippmetzger/OneDrive/PHILIPP/NOVA IMS/2nd Semester/06 Text Mining 4 ECTS/00 Project/corpus/"
cs_en = pd.read_csv(path_corpus + "cs-en/scores.csv")
de_en = pd.read_csv(path_corpus + "de-en/scores.csv")
en_fi = pd.read_csv(path_corpus + "en-fi/scores.csv")
en_zh = pd.read_csv(path_corpus + "en-zh/scores.csv")
ru_en = pd.read_csv(path_corpus + "ru-en/scores.csv")
zh_en = pd.read_csv(path_corpus + "zh-en/scores.csv")

data_list = [cs_en, de_en, en_fi, en_zh, ru_en, zh_en]
names_list = ['cs_en', 'de_en', 'en_fi', 'en_zh', 'ru_en', 'zh_en']
text_columns_list = ['source', 'reference', 'translation']

tokenized_corpus = []
for column in range(3):
    for data in data_list:
        for i in range(data.shape[0]):
            string = data.iloc[i,column].lower()
            string = string.replace('.', ' .')
            string = string.replace('?', ' .')
            string = string.replace('!', ' .')
            string = string.replace('(', '')
            string = string.replace(')', '')
            # Remove all numbers
            string = re.sub(r'[0-9]', '', string)
            # If string contains chinese
            if re.search(u'[\u4e00-\u9fff]', string):
                string = list(string)
                string = ' '.join(string)
            split_entry = string.split()
            tokenized_corpus.append(split_entry)

vocabulary = {word for doc in tokenized_corpus for word in doc}
word2idx = {w:idx for (idx, w) in enumerate(vocabulary)}

def build_training(tokenized_corpus, word2idx, window_size=2):
    window_size = 2
    idx_pairs = []
    
    # for each sentence
    for sentence in tqdm(tokenized_corpus):
        indices = [word2idx[word] for word in sentence]
        # for each word, threated as center word
        for center_word_pos in range(len(indices)):
            # for each window position
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                # make soure not jump out sentence
                if  context_word_pos < 0 or \
                    context_word_pos >= len(indices) or \
                    center_word_pos == context_word_pos:
                    continue  
                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))
    return np.array(idx_pairs)

training_pairs = build_training(tokenized_corpus, word2idx)

def get_onehot_vector(word_idx, vocabulary):
    x = torch.zeros(len(vocabulary)).float()
    x[word_idx] = 1.0
    return x

def Skip_Gram(training_pairs, vocabulary, embedding_dims=5, learning_rate=0.001, epochs=10):
    torch.manual_seed(3)
    W1 = Variable(torch.randn(embedding_dims, len(vocabulary)).float(), requires_grad=True)
    losses = []
    for epo in tqdm(range(epochs)):
        loss_val = 0
        for input_word, target in training_pairs:
            x = Variable(get_onehot_vector(input_word, vocabulary)).float()
            y_true = Variable(torch.from_numpy(np.array([target])).long())

            # Matrix multiplication to obtain the input word embedding
            z1 = torch.matmul(W1, x)
    
            # Matrix multiplication to obtain the z score for each word
            z2 = torch.matmul(torch.transpose(W1, 0, 1), z1)
    
            # Apply Log and softmax functions
            log_softmax = F.log_softmax(z2, dim=0)
            # Compute the negative-log-likelihood loss
            loss = F.nll_loss(log_softmax.view(1,-1), y_true)
            loss_val += loss.item()
            
            # compute the gradient in function of the error
            loss.backward() 
            
            # Update your embeddings
            W1.data -= learning_rate * W1.grad.data
            #W2.data -= learning_rate * W2.grad.data

            W1.grad.data.zero_()
            #W2.grad.data.zero_()
        
        losses.append(loss_val/len(training_pairs))
    
    return W1, losses

def plot_loss(loss):
    x_axis = [epoch+1 for epoch in range(len(loss))]
    plt.plot(x_axis, loss, '-g', linewidth=1, label='Train')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

W1, losses = Skip_Gram(training_pairs, word2idx, epochs=1000)