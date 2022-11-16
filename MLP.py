import numpy as np
import pandas as pd
import DataSpilt as ds
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement



class MLP:

    def __init__(self, data, nodes, weights = None, step_size=0.001, momentum=0.5):
        self.data = pd.read_csv("Data/" + data, index_col=0, header=0)  # initializes entire dataset to dataframe
        self.name = data
        self.samples = ds.getSamples(data)  # gets split data
        self.train = pd.concat(self.samples[:9])  # creates training data
        self.test = self.samples[9]  # creates test data
        self.weights = weights
        self.step_size = step_size  # initilizes step size
        self.momentum = momentum  # initializes momentum
        self.bias = 1


        #self.data = pd.DataFrame(np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]]))
        #self.test = self.data
        #self.train = self.data



        if 'class' in self.train.columns:  # checks if the dataset is classification or regression
            self.classification = True
            self.classes = self.data['class'].unique()
            self.output = len(self.classes)
            if self.output == 2:
                self.output = 1  # binary classification
        else:
            self.classification = False
            self.output = 1  # if regression one single output

        self.input = self.data.shape[1]-1  # an input for each feature that is not the output
        self.layers = len(nodes)  # uses list of nodes to determine layers
        self.nodes = nodes  # list of nodes per hidden layer

        # initializes 3 dim array of weights either -0.01, or 0.01. Can be jagged
        # [input layer (input x nodes), [hidden nodes (prev layer x current x layers)], output layer (nodes x output)]

        if not self.weights:
            if nodes[0] == 0:  # case for no hidden layers
                self.weights = [np.random.uniform(-0.01, 1/(self.input+self.output), size=(self.input, self.output))]
                self.bweights = [np.random.uniform(-0.01, 0.01, size=(1, self.output))]
            else: # case for one or more hidden layers
                self.weights = [np.random.uniform(0,1/(self.input+self.nodes[0]), size=(self.input, self.nodes[0]))]  # first hidden layer
                [self.weights.append(np.random.uniform(0,1/(self.nodes[i]+self.nodes[i+1]), size=(self.nodes[i],self.nodes[i+1]))) for i in range(len(self.nodes)-1)],
                self.weights.append(np.random.uniform(0,1/(self.nodes[-1]+self.output), size=(self.nodes[-1], self.output)))  # output layer

                # bias nodes adds a node for every layer that has no inputs, insures activation of sigmoids
                self.bweights = [np.random.uniform(-0.01, 0.01, size=(1, self.nodes[i])) for i in range(len(self.nodes))]
                self.bweights.append(np.random.uniform(-0.01, 0.01, size=(1, self.output)))

        self.deltas = [np.zeros_like(i) for i in self.weights]  # empty set for deltas/gradients of prev back prop

        # make copies of original state of MLP for cross-validation

    # train
    # ------
    # While not converged, feeds forward and does backpropagation for each point in the training set

    def Train(self,video=False, Graph=False, gradient=False, history=False):
        # creating training set
        trainV = self.train.to_numpy()  # initializes dataframe to numpy

        # while training on the training set still improves by some amount (can change later)
        np.random.shuffle(trainV)  # shuffles training data

        # actual and predicted vectors to calculate F1 scores or MSE
        actual = []
        prediction = []

        # iterating through each element in the training set
        for x in trainV:
            t = x[-1]  # sets target output
            x = x[:-1]  # drops class catagory
            actual = np.append(actual, t)  # adds target to actual

            # gets result from output nodes
            A = self.feedForward(x)
            O = A[-1][-1]  # sets output
            #print("actual",t,"\npredicted",O)

            if self.output > 1:  # multi-classification takes the index of the largest output of softmax
                O = self.classes[np.argmax(O)]
            elif self.classification:  # binary classification rounds to index either 0 or 1 of classes
                O = self.classes[int(O.round()[0])]
            # regression is just the output

            prediction = np.append(prediction, O)  # adds prediction to list

        performance = self.performance(prediction, actual)  # adds epoch performance to list

        return performance

    def feedForward(self, x):

        a = x[:,np.newaxis].T   # sets the first input
        A = [a]  # A is array of each activation from the input of each node
        weights = self.weights
        bweights = self.bweights
        # ADDS BIAS WEIGHTS TO WEIGHTS
        for idx,w in enumerate(weights[:-1]): # goes through the network
            z = np.dot(a,w) + bweights[idx] # gets input to activation function from prev output/current input times wieght
            A.append(z)
            a = np.tanh(z)  # gets hyperbolic tan output for the layer
            A[-1] = (A[-1],a)   # weighted input to node and output of node to list

        w = weights[-1]  # initialize final weight
        z = np.dot(a,w) + bweights[-1] # getting inputs to final node
        A.append(z)
        if self.output > 2:  # multi classification
            y = self.softMax(z)
        elif self.classification:  # binary classification
            y = np.tanh(z)
        else:  # regression
            y = z

        A[-1] = (A[-1], y) # adds final output to output list

        #print("Weights\n", self.weights, "\n")
        #print("Output\n", A,"\n\n\n")

        return A  # returns outputs

    # Sigmoid
    # ----------
    # input can be scaler or 1d array
    # follows: F(x) = 1/(1+e^(z))
    # z is the input from: sum for all i of: (w_i*x_i + Bias)
    def sigmoid(self, z):
        np.seterr(over="warn")
        val = 1 / (1 + np.exp(-z))
        np.seterr(over="warn")
        return val


    # Softmax
    # For Classification with >2 classes
    def softMax(self, o):
        val = np.exp(o)/np.sum(np.exp(o))
        return val

    # performance
    # -----------------
    # decides which performance metric to use to get performance
    # MSE for regression F1 for classification

    def performance(self, prediction, actual):
        np.seterr(invalid="ignore")
        if(self.classification):
            performance = f1_score(prediction, actual)
        else:
            performance = np.sum((np.array(prediction)-np.array(actual))**2) /len(prediction)
        np.seterr(invalid="warn")
        return performance

    def getWeights(self):
        return self.weights

# F1 score
# -------------
# gets F1 score for classification

def f1_score(prediction, actual):
    np.seterr(divide='ignore', invalid='ignore')
    m = pd.crosstab(prediction, actual)  # confusion matrix
    i = m.index.union(m.columns)
    m = m.reindex(index=i, columns=i, fill_value=0)  # makes the matrix square
    m.fillna(0)  # fills na values w/ 0
    P = precision(m)  # calculates precision
    R = recall(m)  # calcaultes recall
    f1 = 2*(R * P) / (P + R)  # calculates f1
    f1 = f1[~np.isnan(f1)]  # drops na values
    f1 = np.sum(f1) / f1.shape[0]  # gets average of all f1 scores
    np.seterr(divide='warn', invalid='warn')
    return f1

def precision(m):
    M = m.to_numpy()
    diag = np.diag(M)  # true positives
    p = diag / np.sum(M, axis= 0)  # true positives / TP + false positives
    return p

def recall(m):

    M = m.to_numpy()
    diag = np.diag(M)  # true positives
    r = diag / np.sum(M, axis= 1)  # true positives / TP + false negatives
    return r

# tune
# --------------
# given a data set, gets performance on test set for all combinations of nodes and layers and given step sizes

def tune():  # to tune things
    data = ["forestfires.data"] # files to tune
    performance = {}  # to save results

    for a in data:
        print(a)
        test = MLP(a,[4,4],500, step_size = 0.05,  momentum = .1)
        nodes = [list(combinations_with_replacement([test.input-i for i in range(test.input-test.output+1)], i)) for i in [0,1,2]]  # gets combinations of layers to tun, between inpout nodes and output nodes
        stepsize = [0.000001, 0.000005,0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.01]  # stepsizes to test
        performance[a] = {}

        for ii in reversed(nodes):
            for i in reversed(ii):
                layers = str(i)
                test = MLP(a,i, 500,step_size = 0.05,  momentum = .5)
                performance[a][layers] = {}
                for l in stepsize:
                    print(test.nodes, l)
                    test.step_size = l
                    result = test.crossValidate(1)  # running test
                    performance[a][layers][l] = result
                    print(result)
        print(performance[a])

    print(performance)

# Video
# ------------
# to fulfill all video requirements

def video():

    print("One test fold sample outputs for glass and machine, 500 epochs:\n")
    layers = [([0],0.01),([7],0.01),([7,7],0.01)]
    print("Classification: Glass\n")
    for i in layers:
        test = MLP("glass.data", i[0], i[1], 0.1)
        result = test.crossValidate(1)
        print("Nodes in Layer(s):", i[0],"\nPerformance (F1 score):", result)
    layers2 = [([0], 0.0001), ([6], 0.0001), ([6, 6], 0.00001)]
    print("\nRegression: Machine\n")
    for i in layers2:
        test2 = MLP("machine.data", i[0], i[1], 0.5)
        result = test2.crossValidate(1)
        print("Nodes in Layer(s):", i[0], "\nPerformance (MSE):", result)


    print("\nSample models for no hidden layer, one hidden layer, and two hidden "
          "layers\n-------------------------------------")
    print("Inputs are size of the Y axis (rows), output is size of the X axis (columns)")
    print("Inputs are dot multiplied with the weights\n"
          "then the output (1d vector) is put through tanh activation function\n------------------------------")
    list = ["no hidden layers", "1 hidden layer", "2 hidden layers"]
    layers = [([0], 0.01), ([6], 0.01), ([6,6], 0.01)]
    for idx, i in enumerate(layers):
        print(list[idx])
        test = MLP("glass.data", i[0], i[1], 0.1, 250 )
        print(test.weights)


    print("\nForward Propagation with tanh activation function (machine):\n")  # other print statements in train
    test2.Train(video=True)


    print("\n\nOutput gradient calculation and weight updates:")  # other print statements in backpropagate
    print("classification:\n")
    test.Train(gradient=True)
    print("regression:\n")
    test2.Train(gradient=True)

    layers = [([0],0.01),([7],0.01),([7,7],0.01)]
    print("\n10 fold CrossValidation for glass:")
    for i in layers:
        test = MLP("glass.data", i[0], i[1], 0.1)
        result = test.crossValidate(10)
        print("Nodes in Layer(s):", i[0],"\nPerformance (F1 score):", result)

# convergenceExperiment
# ------------------------
# runs experiment to get data to create graph of average convergence over 10 folds for a variety of step sizes
def convergenceExperiment():
    data = [["machine.data",[6,6],[ 0.000005,0.00001,0.00005,0.0001,0.0005],0.5, 500], ["glass.data", [7,7] ,[0.001,0.005,0.01,0.05, 0.1],0.1, 500]]  # sets and steps
    results = {}
    for d in data:
        print(d[0])
        results[d[0]] = {}
        for step in d[2]:
            print(step)
            results[d[0]][step] = {}
            test = MLP(d[0],d[1],step,d[3],d[4])
            result = test.crossValidate(10, history=True)  # test
            print(result)
            results[d[0]][step] = result

    print(results)
