if(os.path.exists("wine_quality_model.h5")==False):
#     #sequential model consisting of layers.
#     #defining the content of layers
#     model=kr.models.Sequential([
#         kr.Input(shape=(x_train.shape[1],)),#no of features in the training data.
#         kr.layers.Dense(units=32,activation=kr.activations.relu,name="L1"),
#         kr.layers.Dense(units=16,activation=kr.activations.relu,name="L2"),
#         kr.layers.Dense(units=10,activation=kr.activations.linear,name="L3")
#     ])
#     #defining the loss calculation and the optimizer(the way we modify the weights.)

#     model.compile(
#         optimizer=kr.optimizers.Adam(learning_rate=0.001),
#         loss=kr.losses.SparseCategoricalCrossentropy(from_logits=True)
#         # the max label is 7, so when the loss function converts each label to a one-hot encoded vector: it does like 3->[0,0,0,1,0,...]
#     )

#     trained_model=model.fit(
#         x_train,y_train,epochs=30,validation_split=0.3
#     )
#     model.save("wine_quality_model.h5")

# model_path = 'wine_quality_model.h5'
# if os.path.exists(model_path):
#     # Load the model
#     loaded_model = kr.models.load_model(model_path)
# else:
#     print("Model file not found. Please train the model first.")
# #what does validation split do?
# predictions=loaded_model.predict(x_test)
# index=0
# correct=0
# # create mapping from label to index of neuron.
# for prediction in predictions:
#     if(np.argmax(prediction)==y_test.iloc[index]):
#         correct+=1
#     index+=1
# print(f"Accuracy:{correct/x_test.shape[0]} in {x_test.shape[0]} rows")
# # how to print loss after each epoch during the model training.
