import numpy as np
import pandas as pd
import DataSpilt as ds
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement



class MLP:

    def __init__(self, data, nodes, step_size=0.001, momentum=0.5, epochs = 500, weights = None):
        self.data = pd.read_csv("Data/" + data, index_col=0, header=0)  # initializes entire dataset to dataframe
        self.name = data[:-5]
        self.samples = ds.getSamples(self.data)  # gets split data
        self.train = pd.concat(self.samples[:9])  # creates training data
        self.test = self.samples[9]  # creates test data
        self.weights = weights
        self.step_size = step_size  # initilizes step size
        self.momentum = momentum  # initializes momentum
        self.bias = 1
        self.epochs = epochs


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


        if len(nodes) == 0 or nodes[0] == 0:  # case for no hidden layers
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
        self.weightsCopy = self.weights.copy()
        self.bweightsCopy = self.bweights.copy()
        self.deltasCopy = self.deltas.copy()


    # train
    # ------
    # While not converged, feeds forward and does backpropagation for each point in the training set

    def Train(self,video=False, Graph=False, gradient=False, history=False):
        # creating training set
        trainV = self.train.to_numpy()  # initializes dataframe to numpy

        # helper variables for calculating convergence
        performance = [0]
        converged = False
        in_row = 0

        times = [0]

        # for live graph
        if Graph:
            plt.ion()
            graph = plt.plot(times, performance)[0]
            plt.xlim([0, 1000])
            if self.classification:
                plt.ylim([0, 1])
            else:
                plt.ylim([0, 100])

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

    def Test(self):
        actual = []  # for performance metrics
        prediction = []
        testV = self.test.to_numpy()  # init test data to numpy
        for x in testV:  # going through each test vector
            t = x[-1]  # sets target output
            x = x[:-1]  # drops class catagory
            actual.append(t)
            A = self.feedForward(x)  # feedsforward test vector
            y = A[-1][0]  # sets output of feed forward
            if self.classification:  # finds predicition
                if self.output > 2:
                    y = A[-1][-1]
                    prediction.append(self.classes[np.argmax(y)])  # softmax
                else:
                    y = A[-1][-1]
                    prediction.append(self.classes[int(y.round().item())])  # binary classification
            else:
                prediction.append(round(y[0][0],1))  # regression
        #print('Predictions:',prediction)
        #print('Actual:     ',actual)
        performance = self.performance(prediction, actual)  # gets performance for test set
        #print("Test:",performance)
        return performance

    # CrossValidate
    # ---------------
    # performs cross validation, resets weights and rotates train test data

    def crossValidate(self, times=10, history=False):
        metric_vec = np.zeros(times)  # stores result for each fold
        timeseries = np.zeros(self.epochs)  # for convergence experiment
        for i in range(times):  # going throught amount of folds

            print('Fold',i)
            #reset MLP to original state
            self.weights = self.weightsCopy  # reinit beginning parameters
            self.bweights = self.bweightsCopy
            self.deltas = self.deltasCopy

            timeseries += self.Train(history=history)  # trains data
            metric_vec[i] = self.Test()  # adds test result to result array
            self.samples.append(self.samples.pop(0))  # rotation of folds
            self.train = pd.concat(self.samples[0:9])
            self.test = self.samples[9]
        return metric_vec.mean()

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

glass = MLP("glass.data",[6,6])
glass.Train()



