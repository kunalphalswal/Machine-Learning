from helper import CTransformer
import os
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
import pandas as pd

def handle_data(file_path,batch_size=10,seq_length=512):
      dataset = pd.read_csv(file_path)

      #reshuffle the data
      dataset = dataset.sample(frac=1).reset_index(drop=True)

      #making the size of dataset divisible by 100 so that suitable batches can be made
      while(len(dataset)%100!=0):
          dataset=dataset[:-1]

        #converting reviews to string datatype so that bert encoder can be applied on it.
      reviews = dataset['review'].astype(str).tolist()

      # Map the sentiment labels to numeric values
      label_mapping = {
          'positive': 1,
          'negative': 0,
          'neutral': 2
      }
      dataset['sentiment'] = dataset['sentiment'].map(label_mapping)
      device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
      )
      labels = torch.tensor(dataset['sentiment'])
      #Convert into vectors
      tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
      vocab_size = tokenizer.vocab_size

      #converts sentences of words to vector of integers.
      encoded_reviews = tokenizer(
          reviews,
          return_tensors="pt", #retuning tensors are compatible with pytorch
          padding=True,  #TODO: incorporate attention mask
          truncation=True,
          max_length=seq_length
      )

      # Define the split ratio 
      train_size = int(0.5 * len(dataset))
      test_size = len(dataset) - train_size
      
      #divide data into train, test and then batches
      train_reviews =encoded_reviews['input_ids'][:train_size].view(train_size//batch_size,batch_size,seq_length)
      test_reviews =encoded_reviews['input_ids'][train_size:].view(test_size//batch_size,batch_size,seq_length)
      train_labels = labels[:train_size].view(train_size//batch_size,batch_size)
      test_labels = labels[train_size:].view(test_size//batch_size,batch_size)
      return (train_reviews.to(device), test_reviews.to(device),
        train_labels.to(device), test_labels.to(device),
        vocab_size)

def train(model,x_train,x_labels,epochs=20,learning_rate=0.0001,lr_warmup=10000):
  #set the model to training mode
  model.train(True)

  #TODO:read about optimizer and learning rate scheduler.
  opt = torch.optim.Adam(lr=learning_rate, params=model.parameters())
  sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (lr_warmup / x_train.shape[1]), 1.0))

  #start the training in epochs, batches
  for epoch in range(epochs):
    for batch_reviews,batch_labels in zip(x_train,x_labels):
      opt.zero_grad()
      out = model.forward(batch_reviews)
      loss = F.nll_loss(out,batch_labels) #TODO:negative log likelihood loss calculation
      #log the loss
      print(f"Epoch {epoch}, loss: {loss}")
      loss.backward()
      #update parameters
      opt.step()
      #update learning rate
      sch.step()
    #TODO:saving the model after each epoch to stop training without losing progress

def test(model,x_test,x_labels):
    #set model to evaluation mode
    model.eval()
    correct = 0
    for batch_reviews,batch_labels in zip(x_test,x_labels):
        out = model.forward(batch_reviews)
        for i,prob in enumerate(out):
            if(torch.argmax(out[i])==batch_labels[i]):
                correct+=1
    print(f"Accuracy: {(correct/(x_test.shape[0]*x_test.shape[1]))*100}%")

if __name__=="__main__":
    """
        1. Load the datasets
        2. Pre-process the data:
            2a. Tokenization: Convert words to numbers in the input.
            2b. Length of each input(no of words) should be equal to seq_length.
        3. Split into train and test
        4. Decide batch size for training
        5. Perform Training
        6. Perform Testing
    """
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    #handle data
    file_path="helper/IMDB.csv"
    embedding_dim = 128
    batch_size=5
    seq_length=256
    train_reviews,test_reviews,train_labels,test_labels,vocab_size=handle_data(file_path,batch_size=10,seq_length=seq_length)

    if(os.path.exists("IMDBSentiment.pth")==False):
        #train and save the model
        #define the model 
        model = CTransformer(k=embedding_dim,heads=8,depth=6,seq_length=seq_length,num_tokens=vocab_size,num_classes=2).to(device)
        #start the training
        train(model,x_train=train_reviews,x_labels=train_labels,epochs=80,learning_rate=0.0001,lr_warmup=10000)
        torch.save(model.state_dict(),"IMDBSentiment.pth")
    else:
        #load the saved model
        model = CTransformer(embedding_dim,8,6,seq_length,vocab_size,2).to(device)
        model.load_state_dict(torch.load("IMDBSentiment.pth"))
    test(model,test_reviews,test_labels)