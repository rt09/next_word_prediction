# importing necessary libraries

import re
import nltk
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt


# loading the dataset which is a book
corpus=open("dataset_book.txt",errors="ignore").read()

# converting text to lowercase so that America and america will treat as same
corpus=corpus.lower()

# removing unnecessary symbols basically removing non alphanumeric characters
clean_corpus=re.sub('[^a-z0-9]+',' ', corpus)

# downloading punkt tokenizer
nltk.download('punkt')

#tokenizing the text into words
tokens = word_tokenize(clean_corpus)

#length of the sequence to train
train_len = 3

#creating sequences 
text_sequences = []
for i in range(train_len,len(tokens)+1):
  seq = tokens[i-train_len:i]
  text_sequences.append(seq)

#converting the texts into integer sequence for better analysis
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)

# creating numpy array
sequences=np.asarray(sequences)


#vocabulary size is the number of unique words
vocabulary_size = len(tokenizer.word_counts)+1
vocabulary_size

# trianing input data
train_inputs=sequences[:,:-1]

#input sequence length 
seq_length=train_inputs.shape[1]

# training output data
train_targets=sequences[:,-1]

# converting each word index(position of word in the vocabualry) into binary number this is called one hot encoding
train_targets = to_categorical(train_targets, num_classes=vocabulary_size)

# designing lstm model
class lstm(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        #simple lookup table that stores embeddings of a fixed dictionary and size.
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        #lstm 
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=2, bidirectional=False)
        
        #fully connected layer
        self.linear = nn.Linear(hidden_size*seq_length,vocab_size)
    
    def forward(self, input_word):
        #input sequence to embeddings
        embedded = self.embed(input_word)
        
        #passing the embedding to lstm model
        output, hidden = self.lstm(embedded)
        
        #reshaping
        output=output.view(output.size(0), -1)
        
        #fully connected layer
        output = self.linear(output)
        return output,hidden
    
    
model=lstm(vocab_size=vocabulary_size,embed_size=128, hidden_size=256)

#Adam optimizer
optimizer= Adam(model.parameters(), lr=0.07)

#loss
criterion = nn.BCEWithLogitsLoss()

#training the model
def train(epoch):
    #set the model to train
    model.train()
    tr_loss=0    
    
    #clearing the Gradients 
    optimizer.zero_grad()
    
    #predict the output
    y_pred, (state_h, state_c) = model(torch.from_numpy(train_inputs))
    
    #compute the loss
    loss=criterion(y_pred,torch.from_numpy(train_targets))
    detach_tnsor=loss.detach().numpy()
    losses.append(detach_tnsor)
    
    #backpropagate
    loss.backward()

    #update the parameters
    optimizer.step()
    tr_loss = loss.item()

    print("Epoch : ",epoch,"loss : ",loss)


#number of epoch
no_epoch=10
losses=[]

# training the model for multiple epochs to expose training set to model properly
for epoch in range(1,no_epoch+1):
    train(epoch)
     

print(losses)
#plotting the loss, loss is decreasing for each epoch
plt.plot(losses, label='Training loss')
plt.show()
     

def predict_next_word(text):
    #set the model to evaluation
    model.eval()

    #preprocess
    text = text.lower().strip()
    
    #converting the text to word tokens
    input_tokens = word_tokenize(text)
    
    #converting the tokens to integer sequence
    sequences = tokenizer.texts_to_sequences([input_tokens])
    
    #converting to array
    sequences=np.asarray(sequences)
    with torch.no_grad():
        #converting to tensor
        sequences=torch.from_numpy(sequences)
        #predicting the output
        predict,(hidden,cell)=model(sequences)
    
    #applying the softmax layer
    softmax = torch.exp(predict)
    prob = list(softmax.numpy()[0])
    
    #index of the predict word
    indices = np.argsort(prob)[-3:]

    #converting the sequence back to word
    next_word=tokenizer.sequences_to_texts([[index] for index in indices])
    return next_word
    

#we trained our model with sequence length of 2
input_text=input("Enter your sentence with sequence lenght of 2 :")




print("Possible next word will be:", predict_next_word(input_text))
     
