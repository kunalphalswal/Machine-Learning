import helper
from helper import SelfAttention, TransformerBlock, CTransformer
import os
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
import pandas as pd

def handle_data(file_path,embedding_dim = 768,batch_size=10,seq_length=512):
    # Load the IMDB dataset
        dataset = pd.read_csv(file_path)
        #TODO: use the full data
        dataset = dataset.sample(frac=0.01).reset_index(drop=True)
        #making the length of df divisible by 10 so that suitable batches can be made
        while(len(dataset)%100!=0):
            dataset=dataset[:-1]
        reviews = dataset['review'].astype(str).tolist()
        label_mapping = {
            'positive': 1,
            'negative': 0,
            'neutral': 2
        }

        # Map the sentiment labels to numeric values
        dataset['sentiment'] = dataset['sentiment'].map(label_mapping)
        labels = torch.tensor(dataset['sentiment'])
        #Convert into vectors
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        vocab_size = tokenizer.vocab_size
        # print(reviews.dtype)
        encoded_reviews = tokenizer(
            reviews,
            return_tensors="pt", #retuning tensors are compatible with pytorch
            padding=True,
            truncation=True,
            max_length=embedding_dim
        )
        # Define the split ratio (e.g., 80% train, 20% test)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        #divide data into train, test and then batches
        batch_size=10
        seq_length=512
        train_reviews =encoded_reviews['input_ids'][:train_size].view(train_size//batch_size,batch_size,embedding_dim)
        test_reviews =encoded_reviews['input_ids'][train_size:].view(test_size//batch_size,batch_size,embedding_dim)
        train_labels = labels[:train_size].view(train_size//batch_size,batch_size)
        test_labels = labels[train_size:].view(test_size//batch_size,batch_size)
        return train_reviews,test_reviews,train_labels,test_labels,vocab_size

def train(model,x_train,x_labels,epochs):
    for epoch in epochs:
        out = model.forward(x_train)
        loss = F.nll_loss(out,x_labels)
        loss.backward()
        #update the parameters
        #log the loss
if __name__=="__main__":
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    if(os.path.exists("IMDBSentiment.pth")==False):
        #train and save the model
        """
        1. Load the datasets
        2. Pre-process the data:
            2a. Tokenization+ vectorization: Convert words to numbers in the input and labels to indices in the output.
            2b. Length of each input(no of words) should be equal to seq_length.
        3. Split into train and test
        4. Decide batch size for training
        5. Perform Training
        6. Perform Testing
        """
        #handle data
        file_path="helper/IMDB.csv"
        embedding_dim = 768
        batch_size=10
        seq_length=512
        train_reviews,test_reviews,train_labels,test_labels,vocab_size=handle_data(file_path,embedding_dim=768,batch_size=10,seq_length=512)

        #define the model 
        model = CTransformer(k=embedding_dim,heads=8,depth=6,seq_length=seq_length,num_tokens=vocab_size,num_classes=2)
        
        #start the training


    else:
        #load the saved model
        model = CTransformer().to(device)
        model.load_state_dict(torch.load("IMDBSentiment.pth"))