import numpy as np
import pandas as pd
def sigmoidHelper(num):
    # can do this directly with np only
    exponential= np.exp(num)
    return exponential/(1+exponential)
 
class Activation:
    def __init__(self,function):
        self.function=function
    def produceActivation(self,inputArray,W,B):
        if(self.function=='relu'):
            return self.relu(inputArray,W,B)
        elif(self.function=='linear'):
            return self.linear(inputArray,W,B)
        elif(self.function=='sigmoid'):
            return self.sigmoid(inputArray,W,B)
    def addB(self,output,B):
        for row in output:
            row=row+B
        return output
    def relu(self,inputArray,W,B):
        output = np.matmul(inputArray,W.T)
        output=self.addB(output,B)
        return np.maximum(0,output)
    def linear(self,inputArray,W,B):
        output = np.matmul(inputArray,W.T)
        output=self.addB(output,B)
        return output
    def sigmoid(self,inputArray,W,B):
        output = np.matmul(inputArray,W.T)
        output=self.addB(output,B)
        return sigmoidHelper(output)# see if it works.
        activation=[]
        for row in output:
            newRow=[]
            for scalar in row:
                newRow.append(sigmoidHelper(scalar))
            activation.append(newRow)
        return np.array(activation)
    
#establish relationships between these classes so that certain functions can only be used by classes.
class Layer:
    #will need to handle optimizer too here
    #there should be a type of layer too like Dense, etc.
    def __init__(self,units,activation,name):
        self.units=units
        self.activation=Activation(activation)
        self.name=name
    def initializeParameters(self,inputRows,inputCols):
        #there might be some bias in initialization of weights
        self.W=np.random.uniform(low=-0.5,high=0.5,size=(self.units,inputCols))
        self.B=np.random.uniform(low=-0.5,high=0.5,size=(self.units))
    def produceOutput(self,inputArray):
        #input array has to be horizontal 1 row
        self.lastOutput=self.activation.produceActivation(inputArray,self.W,self.B)
        return self.lastOutput
    def get_weights(self):
        return [self.W,self.B]
class Model:
    #verify dimensions of input before training or predicting
    #need to handle model type too: like Sequential or anything else
    def __init__(self,inputShape,*layers):
        self.layers=[]
        features=inputShape[1]
        inputRows=inputShape[0]
        for layer in layers:
            # sequential automatically assigns feature size to subsequent layers
            # to first layer: cols of input data are the features but to subsequent layers, each unit of previous layer denotes a new feature.
            layer.initializeParameters(inputRows,features)#hope this makes changes to existing object only
            self.layers.append(layer)
            features=layer.units
    def compile(self,optimizer,lossFunction):
        self.optimizer=optimizer
        self.lossFunction=Loss(lossFunction)
    def fit(self,x_train,y_train,epochs):
        #incorporate batch_size, validation, etc later
        for i in range(epochs):
            
            self.train(0,x_train,y_train,self.optimizer.learning_rate)
            predictions=self.predict(x_train)
            cost=0
            avg_price=y_train.mean()
            for j in range(x_train.shape[0]):
                cost+=abs(predictions[j]-y_train[j])
            print(f"Epoch :{i}, loss:{cost}")
    def train(self,layer_ind,input,labels,learning_rate):
        #Calc loss if last layer else get loss from next layer
        if(layer_ind==len(self.layers)):
            prevLoss=[]
            #there may be a better function in numpy
            for j in range(input.shape[1]):#means each activation
                dJ_daj=0
                for i in range(input.shape[0]):
                    dJ_daj+=input[i][j]-labels[i]
                dJ_daj/=len(labels)
                prevLoss.append(dJ_daj)
            return prevLoss
        output=self.layers[layer_ind].produceOutput(input)
        loss=self.train(layer_ind+1,output,labels,learning_rate)
        for unitLoss in loss:
            unitLoss=self.calc_derivative(unitLoss,self.layers[layer_ind].activation)
        #produce loss for previous layer
        prevLoss=[]
        if(layer_ind>0):
            for prevUnit in range(self.layers[layer_ind].W.shape[1]):
                dJ_daprevUnit=0
                for curUnit in range(self.layers[layer_ind].W.shape[0]):
                    dJ_daprevUnit+=loss[curUnit]*self.layers[layer_ind].W[curUnit][prevUnit]
                prevLoss.append(dJ_daprevUnit)
        # update parameters.
        for unit in range(self.layers[layer_ind].W.shape[0]):
            for param in range(self.layers[layer_ind].W.shape[1]):
                da_dwparam=0
                for example in range(input.shape[0]):
                    da_dwparam+=input[example][param]
                da_dwparam/=input.shape[0]
                dj_dwparam=da_dwparam*loss[unit]
                self.layers[layer_ind].W[unit][param]-=(dj_dwparam*self.layers[layer_ind].W[unit][param])
            dj_db=loss[unit]
            B=self.layers[layer_ind].B
            B[unit]-=B[unit]*learning_rate
        #return prev layer loss
        return prevLoss
    def calc_derivative(self,unitLoss,activation):
        if(activation=="linear"):
            return unitLoss
        elif(activation=="relu"):
            if(unitLoss>0):
                return unitLoss
            else:
                return 0
        elif(activation=="sigmoid"):
            return unitLoss*(1-unitLoss)
        else:
            return unitLoss
    def predict(self,inputArray):
        #use model type here maybe: like sequential or ...
        #keep chanign dimensions of the final output array according to layers
        outputArray=np.zeros(inputArray.shape[0])
        for i in range(inputArray.shape[0]):
            input=inputArray[i]
            for col in self.layers:
                input=col.produceOutput(input)
            outputArray[i]=input[0]
        return (outputArray)
    def calculateCost(self,output,y_train):
        cost=0
        for value,label in zip(output,y_train):
            cost+=self.lossFunction.calculateLoss(value,label)
        return cost/output.shape[0]

class Optimizer:
    # how does backpropagation work in the first place and how is calculation of very first derivative compatible with all kinds of losses.
# The updation of weights and biases ie:
# w-=learning_rate*(dL_dw)
#here we want to slowly make changes to w and dL_dw is just a term to signify whether the changes should be smaller or larger because if
# making small change in a parameter substantially decreases the loss then we can afford to make a bigger change in the weight and vice versa 
# for small changes.
    #For updating the values in backprop, we can either store the previous values or recalculate them.
    def __init__(self,optimizer, learning_rate):
        self.optimizer=optimizer
        self.learning_rate=learning_rate

class Loss: #ideally there should be a separate class for each function, or atleast what I'm doing is not correct.
    def __init__(self,lossFunction):
        self.lossFunction=lossFunction
    def calculateLoss(self,value1,value2):
        if(self.lossFunction=="MeanSquaredError"):
            return self.meanSquaredError(value1,value2)
        elif(self.lossFunction=="MeanAbsoluteError"):
            return self.meanAbsoluteError(value1,value2)
    def meanSquaredError(value1,value2):
        return (value1-value2)**2
    def meanAbsoluteError(value1,value2):
        return abs(value1-value2)
def z_score(arr):
    if(np.ndim(arr)>1):
        for attr in range(arr.shape[1]):
            sum=0
            for i in range(arr.shape[0]):
                    sum+=arr[i][attr]
            sum/=arr.shape[0]
            var=0
            for i in range(arr.shape[0]):
                    var+=(sum-arr[i][attr])**2
            var/=arr.shape[0]
            var=np.sqrt(var)
            for i in range(arr.shape[0]):
                    arr[i][attr]=((arr[i][attr]-sum)/var)
    else:
        sum=0
        for i in range(arr.shape[0]):
                sum+=arr[i]
        sum/=arr.shape[0]
        var=0
        for i in range(arr.shape[0]):
                var+=(sum-arr[i])**2
        var/=arr.shape[0]
        var=np.sqrt(var)
        for i in range(arr.shape[0]):
            arr[i]=((arr[i]-sum)/var)
    return arr

if __name__ == "__main__":
    
    # do it with real data after defining loss calculator and optimizer.
    #import data
    df=pd.read_csv("CaliHousing.csv")
    # shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size=int(0.8*df.shape[0])
    # Will need to handle categorical input.
    # encoded_proximities = pd.get_dummies(df['ocean_proximity'],prefix='ocean_proximity')
    # #other ways to use categorical data: https://towardsdatascience.com/an-overview-of-categorical-input-handling-for-neural-networks-c172ba552dee
    # df=pd.concat([df,encoded_proximities],axis=1)
    encoded_proximities = pd.get_dummies(df['ocean_proximity'], prefix='ocean_proximity')

    # Convert boolean values to integers (0 and 1)
    encoded_proximities = encoded_proximities.astype(int)

    # Concatenate the original DataFrame with the new binary columns
    df = pd.concat([df, encoded_proximities], axis=1)
    df=df.drop('ocean_proximity',axis=1)
    df = df.fillna(df.mean())
    # split data into training and testing: ideally 80:20 ratio
    x_train=z_score((df[:train_size].drop(['median_house_value'],axis=1)).to_numpy())
    x_test=z_score(df[train_size:].drop(['median_house_value'],axis=1).to_numpy())
    y_train=z_score(df[:train_size]['median_house_value'].to_numpy())
    y_test = z_score(df[train_size:]['median_house_value'].to_numpy())
    model=Model(x_train.shape,
        Layer(16,"relu","L1"),#define activations separately
        Layer(8,"relu","L2"),
        Layer(4,"relu","L3"),
        Layer(1,"linear","L4")
    )
    model.compile(
        optimizer=Optimizer("Adam",learning_rate=0.01),
        lossFunction="MeanSquaredError"
    )
    trained_model=model.fit(
        x_train,y_train,epochs=5
    )
    predictions=model.predict(x_test)
    cost=0
    avg_price=y_test.mean()
    for i in range(x_test.shape[0]):
        cost+=abs(predictions[i]-y_test[i])
        # if(i%10==0):
        #     print(cost)
    print((cost/x_test.shape[0])/avg_price)
    #avg price is negative because of normalization

