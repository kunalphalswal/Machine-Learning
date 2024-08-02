import numpy as np

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
    def relu(self,inputArray,W,B):
        output = np.matmul(inputArray,W)
        output+=B
        return np.maximum(0,output)
    def linear(self,inputArray,W,B):
        output = np.matmul(inputArray,W)
        output+=B
        return output
    def sigmoid(self,inputArray,W,B):
        output = np.matmul(inputArray,W)
        output+=B
        activation=[]
        for scalar in output:
            activation.append(sigmoidHelper(scalar))
        return np.array(activation)
    
#establish relationships between these classes so that certain functions can only be used by classes.
class Layer:
    #will need to handle optimizer too here
    #there should be a type of layer too like Dense, etc.
    def __init__(self,units,activation,name):
        self.units=units
        self.activation=Activation(activation)
        self.name=name
    def initializeParameters(self,inputCols):
        #there might be some bias in initialization of weights
        self.W=np.random.rand(inputCols,self.units)
        self.B=np.random.rand(self.units)
    def produceOutput(self,inputArray):
        #input array has to be horizontal 1 row
        self.lastOutput=self.activation.produceActivation(inputArray,self.W,self.B)
        return self.lastOutput
    def get_weights(self):
        return [self.W,self.B]
class Model:
    #verify dimensions of input before training or predicting
    #need to handle model type too: like Sequential or anything else
    def __init__(self,inputFeatures,*layers):
        self.layers=[]
        features=inputFeatures
        for layer in layers:
            # sequential automatically assigns feature size to subsequent layers
            # to first layer: cols of input data are the features but to subsequent layers, each unit of previous layer denotes a new feature.
            layer.initializeParameters(features)#hope this makes changes to existing object only
            self.layers.append(layer)
            features=layer.units
    def compile(self,lossFunction,optimizer):
        self.optimizer=optimizer
        self.lossFunction=Loss(lossFunction)
    def fit(self,x_train,y_train,epochs):
        #incorporate batch_size, validation, etc later
        for i in range(epochs):
            output=self.predict(x_train)
            cost=self.lossFunction(output,y_train)
            print(f"Epoch {i+1}: Loss: {cost}")
            self.optimizer.backProp(self,output,y_train)
        #function to produce output
        #function to calculate loss
        #function to propagate back the loss and update the weights in the layers.
        #repeat for each epoch
    def predict(self,inputArray):
        #use model type here maybe
        #keep chanign dimensions of the final output array according to layers
        for layer in self.layers:
            output=[]
            for row in inputArray:
                output.append(layer.produceOutput(row))
            inputArray=np.array(output)
        return inputArray
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
    def backProp(self,model,output,y_train):
        n = len(model.layers)
        #calculate derivate of last layer here, so that rest layers get ready to consume values.
        #here, we store the derivative of loss wrt upcoming layer.
        DL_DA = output-y_train
        #assuming mean squared error.
        for i in range(n-1,-1,-1):
            DA_DZ=self.calcDerivative(self,model.layers[i].activation,model.layers[i].lastOutput)
            for j in range(0,model.layers[i].units):
                DZ_DW=
                model.layers[i].W-self.learning_rate*DZ_DW

class Loss: #ideally there should be a separate class for each function, or atleast what I'm doing is not correct.
    def __init__(self,lossFunction):
        self.lossFunction=lossFunction
    def calculateLoss(self,value1,value2):
        if(self.lossFunction=="MeanSquaredError"):
            return self.meanSquaredError(value1,value2)
    def meanSquaredError(value1,value2):
        return (value1-value2)**2

if __name__ == "__main__":
    model=Model(4,
        Layer(5,"relu","L1"),#define activations separately
        Layer(3,"relu","L2"),
        Layer(1,"linear","L3")
    )
    # do it with real data after defining loss calculator and optimizer.
    model.fit()
    print(model.predict(np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])))
