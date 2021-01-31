#!/usr/bin/env python3
"""
student.py
Student name:
Chen Zikang:z5272654
Liu Jizhou:z5222381
groupID: g023486
UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.

Question:Briefly describe how your program works, and explain any design and training decisions you made along the way.

Answer:
For the preprocessing part, we use the stopwords in the nltk library as our stopwords. Then we get rid of all the stopwords
in the sentences. We also use re library to remove all the junk character. For postprocessing and tokenise, we don't think any
processing step is required, and there is no need to modify codes in tokenise part. We choose word dimension of 200,
we found that in our program, word dimension 200 has the best performance among 50,100,200,300.

For the model structure part, we initially choose GRU as our RNN model, because we know that both GRU and LSTM
are good model from language classification. But the highest scores for me using GRU is around 83. Then, I choose LSTM instead.
We used a LSTM to handle the varying input size as well as the long-term dependencies of the product reviews. Because we use
bidirectional LSTM, so we take the first and the last output in the LSTM network and concat them together. We construct a linear
layer after the LSTM layer, following by relu activation function. Here, I also try different function like tanh and leakyrelu, but
relu shows the best performance. Next, we build two fully-connected layers, one is for predicting the rating(vetcor size 2), another one
is for predicting the category(vector size 5).

In loss function part, we use the cross-entropy function to compute the loss for rating and category. After many times experiment,
I finally choose all the dataset as my training set, and iteration time is 10 and let batchsize = 32. I use adam optimiser
because LSTM does not converge with SGD in my model. In testing part, I use argmax to take the largest probability in rating
and category vectors to be my model prediction.

This assignment give us much knowledge about the usage of LSTM and GRU. These two RNN-based models can handle greatly long-term sequence
of words. We finally get around 85 scores at the end. We also try BERT on pytorch and this implementation shows very good performance in the task,
but we know BERT is not allowed in this work.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
import re
from torchtext.vocab import GloVe
import numpy as np
import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # remove illegal characters in sample
    """
        Called after tokenising but before numericalising.
        """
    afdata = []
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
                 "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
                 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
                 'themselves',
                 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
                 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
                 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
                 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
                 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
                 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
                 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
                 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
                 "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                 "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
                 "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    junk_reg = '[#@_$%\s\w\d]'
    stop_words = set(stopwords)
    for i in range(len(sample)):
        sentence = sample[i]
        sentence = sentence.lower()
        sentence = re.findall(junk_reg, sentence, re.S)
        sentence = "".join(sentence)
        word_tokens = sentence.split(' ')
        filtered_w = [w for w in word_tokens if w not in stop_words]
        sc = " ".join(filtered_w)
        afdata.append(sc)
    return afdata

def postprocessing(batch, vocab):

    return batch

stopWords = {}
wordVectors = GloVe(name='6B', dim=200)  #(50, 100, 200 or 300)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    ratingOutput = ratingOutput.argmax(dim=1, keepdim=True)
    categoryOutput = categoryOutput.argmax(dim=1, keepdim=True)
    return ratingOutput, categoryOutput

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        self.lstm = tnn.LSTM(input_size=200,hidden_size=50,num_layers=2,bias=True,dropout=0.5,batch_first=True,bidirectional=True)
        self.linear1 = tnn.Linear(in_features=200,out_features=50,bias=True)
        self.FC1 = tnn.Linear(in_features=50, out_features=2, bias=True)
        self.FC2 = tnn.Linear(in_features=50, out_features=5, bias=True)
        self.tanh = tnn.Tanh()
        self.lrelu = tnn.LeakyReLU()
        self.relu = tnn.ReLU()

    def forward(self, input, length):
        output, _ = self.lstm(input)
        output1 = torch.cat((output[:, -1, :], output[:, 0, :]), dim=1)
        output2 = self.linear1(output1)
        output3 = self.relu(output2)
        ratingOutput = self.FC1(output3)
        categoryOutput = self.FC2(output3)
        return ratingOutput, categoryOutput

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.loss = tnn.CrossEntropyLoss()
    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        rating_loss = self.loss(ratingOutput,ratingTarget)
        category_loss = self.loss(categoryOutput,categoryTarget)
        return torch.mean(rating_loss)+torch.mean(category_loss)

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 1.0
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters())
