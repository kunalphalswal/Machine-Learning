from helper import CTransformer
import os
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
import pandas as pd

def handle_data(file_path, batch_size=10, seq_length=512):
    dataset = pd.read_csv(file_path)

    # Reshuffle the data
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    # Making the size of dataset divisible by 100 so that suitable batches can be made
    while len(dataset) % 100 != 0:
        dataset = dataset[:-1]

    # Converting reviews to string datatype so that BERT encoder can be applied on it
    reviews = dataset['review'].astype(str).tolist()

    # Map the sentiment labels to numeric values
    label_mapping = {
        'positive': 1,
        'negative': 0,
        'neutral': 2
    }
    dataset['sentiment'] = dataset['sentiment'].map(label_mapping)

    # Device handling
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    labels = torch.tensor(dataset['sentiment']).to(device)

    # Convert reviews into vectors
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    # Converts sentences of words to vector of integers
    encoded_reviews = tokenizer(
        reviews,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seq_length
    ).to(device)

    padding = encoded_reviews['attention_mask'].to(device)

    # Define the split ratio
    train_size = int(0.5 * len(dataset))
    test_size = len(dataset) - train_size

    # Divide data into train, test, and then batches
    train_reviews = encoded_reviews['input_ids'][:train_size].view(train_size // batch_size, batch_size, seq_length)
    test_reviews = encoded_reviews['input_ids'][train_size:].view(test_size // batch_size, batch_size, seq_length)
    train_labels = labels[:train_size].view(train_size // batch_size, batch_size)
    test_labels = labels[train_size:].view(test_size // batch_size, batch_size)
    train_padding = padding[:train_size].view(train_size // batch_size, batch_size, seq_length)
    test_padding = padding[train_size:].view(test_size // batch_size, batch_size, seq_length)

    return (train_reviews, test_reviews, train_labels, test_labels, train_padding, test_padding, vocab_size)

def train(model, x_train, x_labels, padding, epochs=20, learning_rate=0.0001, lr_warmup=10000):
    # Set the model to training mode
    model.train(True)
    
    # TODO: Read about optimizer and learning rate scheduler
    opt = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (lr_warmup / x_train.shape[1]), 1.0))

    # Start the training in epochs, batches
    for epoch in range(epochs):
        epoch_loss = 0 
        for batch_reviews, batch_labels, batch_padding in zip(x_train, x_labels, padding):
            opt.zero_grad()
            out = model(batch_reviews, batch_padding)
            loss = F.nll_loss(out, batch_labels)  # TODO: Read about negative log likelihood loss calculation
            loss.backward()
            epoch_loss += loss.item()  # Use `.item()` to extract the loss value
            opt.step()
            sch.step()
        
        # Log the loss
        print(f"Epoch {epoch + 1}, loss: {epoch_loss:.4f}")

        # TODO: Save the model after each epoch to stop training without losing progress

def test(model, x_test, x_labels, padding):
    # Set the model to evaluation mode
    model.eval()
    correct = 0
    
    with torch.no_grad():
        for batch_reviews, batch_labels, batch_padding in zip(x_test, x_labels, padding):
            out = model(batch_reviews, batch_padding)
            correct += (torch.argmax(out, dim=1) == batch_labels).sum().item()

    # Calculate and print accuracy
    accuracy = (correct / (x_test.size(0) * x_test.size(1))) * 100
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    """
        1. Load the datasets
        2. Pre-process the data:
            2a. Tokenization: Convert words to numbers in the input.
            2b. Length of each input (number of words) should be equal to seq_length.
        3. Split into train and test
        4. Decide batch size for training
        5. Perform Training
        6. Perform Testing
    """
    # Get CPU, GPU, or MPS device for training.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Handle data
    file_path = r"C:\Users\Rohan\Desktop\Machine-Learning-main\Transformers\helper\IMDB.csv"
    embedding_dim = 128
    seq_length = 256
    batch_size = 5
    (train_reviews, test_reviews, train_labels, test_labels, train_padding, test_padding, vocab_size) = handle_data(file_path, batch_size=batch_size, seq_length=seq_length)

    if not os.path.exists("IMDBSentiment.pth"):
        # Train and save the model
        # Define the model
        model = CTransformer(k=embedding_dim, heads=8, depth=6, seq_length=seq_length, num_tokens=vocab_size, num_classes=2).to(device)
        # Start the training
        train(model, x_train=train_reviews, x_labels=train_labels, padding=train_padding, epochs=80, learning_rate=0.0001, lr_warmup=10000)
        torch.save(model.state_dict(), "IMDBSentiment.pth")
    else:
        # Load the saved model
        model = CTransformer(k=embedding_dim, heads=8, depth=6, seq_length=seq_length, num_tokens=vocab_size, num_classes=2).to(device)
        model.load_state_dict(torch.load("IMDBSentiment.pth", map_location=device))
    
    # Perform testing
    test(model, x_test=test_reviews, x_labels=test_labels, padding=test_padding)
