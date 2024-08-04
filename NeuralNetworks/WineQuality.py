import pandas as pd
import numpy as np
import keras as kr
import os
#import data
df=pd.read_csv("WineQT.csv")
# shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df=df.drop('Id',axis=1)
#split data into training and testing: ideally 80:20 ratio
x_train=df[:1000].drop(['quality'],axis=1)
x_test=df[1000:].drop(['quality'],axis=1)
y_train=df[:1000]['quality']
y_test = df[1000:]['quality']
if(os.path.exists("WineQualityModel.keras")==False):
    #sequential model consisting of layers.
    #defining the content of layers
    model=kr.models.Sequential([
        kr.Input(shape=(x_train.shape[1],)),#no of features in the training data.
        kr.layers.Dense(units=32,activation=kr.activations.relu,name="L1"),
        kr.layers.Dense(units=16,activation=kr.activations.relu,name="L2"),
        kr.layers.Dense(units=10,activation=kr.activations.linear,name="L3")
    ])
    #defining the loss calculation and the optimizer(the way we modify the weights.)

    model.compile(
        optimizer=kr.optimizers.Adam(learning_rate=0.001),
        loss=kr.losses.SparseCategoricalCrossentropy(from_logits=True)
        # the max label is 7, so when the loss function converts each label to a one-hot encoded vector: it does like 3->[0,0,0,1,0,...]
    )

    trained_model=model.fit(
        x_train,y_train,epochs=30,validation_split=0.1
    )
    model.save("WineQualityModel.keras")
    
model_path = 'WineQualityModel.keras'
if os.path.exists(model_path):
    # Load the model
    loaded_model = kr.models.load_model(model_path)
else:
    print("Model file not found. Please train the model first.")
#what does validation split do?
predictions=loaded_model.predict(df.drop(['quality'],axis=1))
index=0
correct=0
# because the neurons correspond to [0,10), and the labels are [3,8], the encoding corresponds to: 3->neuron 4, 4->neuron 5, 5-> neuron 6, 6->neuron 7 and.. 8->neuron 9.
#proof below: Nope, that is not a good visualizer because data was heavily skewed towards 5,6. hence every prediction is 5 or 6. OVERFITTING
category8=[]
for prediction in predictions:
    if(df['quality'].iloc[index]==5):
        category8.append(np.argmax(prediction))
    index+=1
category8=np.array(category8)
unique_values, counts = np.unique(category8, return_counts=True)

# Create a dictionary with unique values and their counts
value_counts = dict(zip(unique_values, counts))

print(value_counts)
# how to print loss after each epoch during the model training.
