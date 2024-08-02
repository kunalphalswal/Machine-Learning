import pandas as pd
import numpy as np
import keras as kr
import os
#import data
df=pd.read_csv("CaliHousing.csv")
# shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_size=int(0.8*df.shape[0])
# Will need to handle categorical input.
encoded_proximities = pd.get_dummies(df['ocean_proximity'],prefix='ocean_proximity')
#other ways to use categorical data: https://towardsdatascience.com/an-overview-of-categorical-input-handling-for-neural-networks-c172ba552dee
df=pd.concat([df,encoded_proximities],axis=1)
df=df.drop('ocean_proximity',axis=1)
# split data into training and testing: ideally 80:20 ratio
x_train=df[:train_size].drop(['median_house_value'],axis=1)
x_test=df[train_size:].drop(['median_house_value'],axis=1)
y_train=df[:train_size]['median_house_value']
y_test = df[train_size:]['median_house_value']
avg_price=y_test.mean()
if(os.path.exists("CaliHousingModel.keras")==False):
    #sequential model consisting of layers.
    #defining the content of layers
    model=kr.models.Sequential([
        kr.Input(shape=(x_train.shape[1],)),#no of features in the training data., so that parameters are initialized
        kr.layers.Dense(units=64,activation=kr.activations.relu,name="L1"),
        kr.layers.Dense(units=32,activation=kr.activations.relu,name="L2"),
        kr.layers.Dense(units=16,activation=kr.activations.relu,name="L3"),
        kr.layers.Dense(units=8,activation=kr.activations.relu,name="L4"),
        kr.layers.Dense(units=1,activation=kr.activations.linear,name="L5")
    ])
    #defining the loss calculation and the optimizer(the way we modify the weights.)

    model.compile(
        optimizer=kr.optimizers.Adam(learning_rate=0.001),
        loss=kr.losses.MeanSquaredError()
    )

    trained_model=model.fit(
        x_train,y_train,epochs=20
    )
    model.save("CaliHousingModel.keras")

model_path = 'CaliHousingModel.keras'
if os.path.exists(model_path):
    # Load the model
    loaded_model = kr.models.load_model(model_path)
else:
    print("Model file not found. Please train the model first.")
mae=kr.losses.MeanAbsoluteError()
predictions=loaded_model.predict(x_test)
error = mae(predictions,y_test)
print(f"MAE is:{error}, ratio of MAE over mean house price{error/avg_price}")
# what does validation split do?
# how to evaluate the model on testing data directly
# how to print loss after each epoch during the model training.
