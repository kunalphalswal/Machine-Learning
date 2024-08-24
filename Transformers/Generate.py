from helper import GTransformer
import os
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
import pandas as pd

def sample_sequence(encoded_data,seq_length):
    total_length = encoded_data.shape[0]
    start_idx = torch.randint(0, total_length - seq_length, (1,)).item()
    sampled_sequence = encoded_data[start_idx:start_idx + seq_length]
    sampled_labels = encoded_data[start_idx+1:start_idx+seq_length+1]
    return (sampled_sequence,sampled_labels)

def handle_data(file_path,batch_size=10,seq_length=256):
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )

    # Specify the path to your text file
    file_path = 'helper/enwik8.txt'

# Open the file and read its contents
    with open(file_path, 'r', encoding='utf-8') as file:
        text_data = file.read()  # Read the entire file as a string

    #Convert into vectors
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    #converts sentences of words to vector of integers.
    encoded_data = tokenizer(
        text_data,
        return_tensors="pt", #retuning tensors are compatible with pytorch
        return_attention_mask=False
    )
    # Squeeze the first dimension to get a 1D tensor
    encoded_data = encoded_data['input_ids'].squeeze(0)

    # Define the number of batches and initialize lists to hold training and testing data
    num_batches = 1000
    x_train, labels_train = [], []
    x_test, labels_test = [], []

    # Generate training data by sampling sequences and their labels
    for batch in range(num_batches):
        for seq in range(batch_size):
            seq, label = sample_sequence(encoded_data, seq_length)
            # Convert seq and label to lists before appending to the arrays
            x_train.append(seq.tolist())
            labels_train.append(label.tolist())

    # Generate testing data by sampling sequences and their labels
    for batch in range(num_batches):
        for seq in range(batch_size):
            seq, label = sample_sequence(encoded_data, seq_length)
            # Convert seq and label to lists before appending to the arrays
            x_test.append(seq.tolist())
            labels_test.append(label.tolist())

    # Convert lists to tensors for PyTorch compatibility
    x_train = torch.tensor(x_train, dtype=torch.long).view(num_batches, batch_size, seq_length)
    labels_train = torch.tensor(labels_train, dtype=torch.long).view(num_batches, batch_size, seq_length)
    x_test = torch.tensor(x_test, dtype=torch.long).view(num_batches, batch_size, seq_length)
    labels_test = torch.tensor(labels_test, dtype=torch.long).view(num_batches, batch_size, seq_length)

    # Transfer tensors to the appropriate device (e.g., GPU if available)
    x_train = x_train.to(device)
    labels_train = labels_train.to(device)
    x_test = x_test.to(device)
    labels_test = labels_test.to(device)

    # Return the tensors along with the vocabulary size
    return x_train, labels_train, x_test, labels_test, vocab_size


def train(model, x_train, x_labels, epochs=20, learning_rate=0.0001, lr_warmup=10000):
    # Set the model to training mode
    model.train(True)
    
    # Initialize optimizer and learning rate scheduler
    opt = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (lr_warmup / x_train.shape[1]), 1.0))
    
    # Start training loop
    for epoch in range(epochs):
        epoch_loss = 0 
        for batch_reviews, batch_labels in zip(x_train, x_labels):
            opt.zero_grad()
            
            # Forward pass
            out = model.forward(batch_reviews)
            
            # Apply softmax to get probabilities
            out = F.softmax(out, dim=-1)
            
            # Flatten the output and labels
            out = out.view(-1, out.size(-1))  # Shape: (t, vocab_size)
            batch_labels = batch_labels.view(-1)  # Shape: (t,)
            
            # Calculate loss only for non-padded tokens: there will be no padding during training or testing.
            loss = F.nll_loss(torch.log(out), batch_labels)
            loss.backward()
            
            epoch_loss += loss.item()
            
            # Update parameters
            opt.step()
            # Update learning rate
            sch.step()

            print(f"Batch loss: {loss.item()}")
        
        # Log the loss
        print(f"Epoch {epoch}, loss: {epoch_loss:.4f}")


    #TODO:saving the model after each epoch to stop training without losing progress

def test(model,x_test,x_labels):
    #set model to evaluation mode
    model.eval()
    correct = 0
    for batch_reviews,batch_labels in zip(x_test,x_labels):
        out = model.forward(batch_reviews)
        for seq,seq_labels in zip(out,batch_labels):
            predicted_tokens = torch.argmax(seq, dim=1)
            for i,token in enumerate(predicted_tokens):
                if(token == seq_labels[i]):
                    correct+=1
    print(f"Accuracy: {(correct/(x_test.shape[0]*x_test.shape[1]))*100}%")

def tokensToSentence(token_ids):
    tokenizer  = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    text = tokenizer.convert_tokens_to_string(tokens)
    return text

def inference(model,seq,padding):
    #make it for sequence by sequence and modify the shape of 1 sequence accordingly to pass it to forward.
    model.eval()
    seq = seq.unsqueeze(0)
    padding = padding.unsqueeze(0)
    out = model.forward(seq,padding)
    #batch_padding is of shape: b,t.
    out = out.squeeze(0)
    padding = padding.squeeze(0)
    # Step 3: Identify non-padded positions
    non_padded_positions = padding.nonzero(as_tuple=True)[0]
    
    # Step 4: Extract the logits corresponding to non-padded tokens
    out = out[non_padded_positions]

    result = []
    #incorporate temperature here.
    temperature = 0.5
    out = out/temperature
    top_k = 5
    for token in out:
        #sample here
        top_k_probs, top_k_indices = torch.topk(token, top_k)
        top_k_probs = top_k_probs / top_k_probs.sum()  # Re-normalize
        next_token = top_k_indices[torch.multinomial(top_k_probs, 1)].item()
        result.append((next_token))
    return tokensToSentence(result)

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
    file_path="helper/enwik8.txt"
    embedding_dim = 256
    batch_size=10
    seq_length=256
    train_reviews,test_reviews,train_labels,test_labels,vocab_size=handle_data(file_path,batch_size=10,seq_length=seq_length)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    if(os.path.exists("WikiGen.pth")==False):
        #train and save the model
        #define the model 
        model = GTransformer(k=embedding_dim,heads=8,depth=6,seq_length=seq_length,num_tokens=tokenizer.vocab_size).to(device)
        #start the training
        train(model,x_train=train_reviews,x_labels=train_labels,epochs=20,learning_rate=0.0001,lr_warmup=10000)
        torch.save(model.state_dict(),"WikiGen.pth")
    else:
        #load the saved model
        model = GTransformer(embedding_dim,8,6,seq_length,tokenizer.vocab_size).to(device)
        model.load_state_dict(torch.load("WikiGen.pth",map_location=device))
    test(model,test_reviews,test_labels)
    sentence = "We are the watchers on the wall"
   
    encoded = tokenizer(sentence,padding=True,max_length=seq_length)
    seq = torch.tensor(encoded['input_ids'])
    padding = torch.tensor(encoded['attention_mask'])
    print(f"These are the predicted tokens:\n{inference(model,seq,padding)}")