import numpy as np
import pandas as pd
from MLP import MLP
import DataSpilt as ds
import random

class Dataset:
    def __init__(self, data):
        if ".data" in data:
            self.name = data[:-5]
        else:
            self.name = data

        self.path = "Data/" + self.name + ".data"

        self.data = pd.read_csv(self.path, index_col=0, header=0)
        self.samples = ds.getSamples(self.data)  # gets split data
        self.train = pd.concat(self.samples[:9])  # creates training data
        self.test = self.samples[9]  # creates test data

        if 'class' in self.train.columns:  # checks if the dataset is classification or regression
            self.classification = True
            self.classes = self.data['class'].unique()
            self.output = len(self.classes)
            if self.output == 2:
                self.output = 1  # binary classification
        else:
            self.classification = False
            self.output = 1  # if regression one single output

        if self.output > 2:  # multi classification
            self.outputF = softMax
        elif self.classification:  # binary classification
            self.outputF = np.tanh
        else:  # regression
            self.outputF = identity

        self.networks = {i: Network(self, i) for i in [0, 1, 2]}


    def shuffle(self):
        self.samples.append(self.samples.pop(0))  # rotation of folds
        self.train = pd.concat(self.samples[0:9])
        self.test = self.samples[9]


class Network:
    def __init__(self, dataset, layers):
        self.dataset = dataset
        w, bw = np.load('Weights/' + self.dataset.name + "/" + str(layers) + "/" + 'weights.npz'), np.load(
            'Weights/' + self.dataset.name + "/" + str(layers) + "/" + 'bweights.npz')
        self.weights = [w[key] for key in w]
        self.bweights = [np.swapaxes(bw[key], 1, 0) for key in bw]
        self.layers = layers
        self.weightsCopy = self.weights.copy()
        self.bweightsCopy = self.bweights.copy()

    # pushes all training vectors through the population network in one pass
    def evaluate(self):
        train = self.dataset.train.to_numpy()
        solutions = train[:, -1]
        a = train[:, :-1]
        for idx, i in enumerate(self.weights[:-1]):
            try:
                z = np.dot(a, i) + self.bweights[idx]
                if z.ndim > 3:
                    raise Exception
            except:
                z = np.einsum('ijk,jkl -> ijl', a, i) + self.bweights[idx]
            a = np.tanh(z)

        w = self.weights[-1]
        try:
            z = np.dot(a, w) + self.bweights[-1]
            if z.ndim > 3:
                raise Exception
        except:
            z = np.einsum('ijk,jkl -> ijl', a, w) + self.bweights[-1]

        y = self.dataset.outputF(z).squeeze()
        performances = self.performance(y, solutions)
        return performances

        # todo: assign performances to weight index of weights in list
        # ie: 4.2 mse is from weight set 5 in the 3d weight list
        # weight set 5 is the the 5/10 on the 3rd dimension
        # for each layer of weights

    def performance(self, prediction, actual):
        np.seterr(invalid="ignore")
        if self.dataset.classification:
            performance = f1_score(prediction, actual)
        else:
            performance = np.sum((prediction - actual[:, np.newaxis]) ** 2, axis=0) / prediction.shape[0]
        np.seterr(invalid="warn")
        return performance

    # reset weights of network object after performing algorithm
    def reset(self):
        self.weights = self.weightsCopy
        self.bweights = self.bweightsCopy

    # performs genetic algorithm
    def geneticAlgorithm(self, prob=.05, SD=.01):

        def select(fitnesses, x = 5):
            # should select x number of pairs weighting selection odds by fitness
            pSelection = fitnesses / np.sum(fitnesses)
            pairs = [np.random.choice(np.where(fitnesses)[0], p=pSelection, replace=False, size=2) for i in range(x)]
            return pairs

        def crossover(pairs):

            nextGenW = []
            nextGenB = []
            #iterate through each pair in the selected set of pairs
            for pair in pairs:
                parent1W = [i[pair[0],:,:] for i in self.weights]
                parent1B = [i[0,pair[0],:] for i in self.bweights]

                parent2W = [i[pair[1], :, :] for i in self.weights]
                parent2B = [i[0,pair[1],:] for i in self.bweights]

                #randomly select genes that will be crossed over between the two parents
                xoverW = [np.random.choice([0,1], p=[.5, .5], size=i.shape) for i in parent1W]
                xoverB = [np.random.choice([0,1], p=[.5, .5], size=i.shape) for i in parent1B]

                child1W = parent1W
                child1B = parent1B
                child2W = parent2W
                child2B = parent2B

                #crossover the two children at the selected indices
                for i in range(len(child1W)):
                    child1W[i][np.where(xoverW[i] == 1)] = parent2W[i][np.where(xoverW[i] == 1)]
                    child2W[i][np.where(xoverW[i] == 1)] = parent1W[i][np.where(xoverW[i] == 1)]
                    child1B[i][np.where(xoverB[i] == 1)] = parent2B[i][np.where(xoverB[i] == 1)]
                    child2B[i][np.where(xoverB[i] == 1)] = parent1B[i][np.where(xoverB[i] == 1)]

                child1W = Mutate(child1W, .05, .01)
                child1B = Mutate(child1B, .05, .01)
                child2W = Mutate(child2W, .05, .01)
                child2B = Mutate(child2B, .05, .01)
                nextGenW.append(child1W)
                nextGenW.append(child2W)
                nextGenB.append(child1B)
                nextGenB.append(child2B)
            pass
        # randomly alter some genes in a given solution's chromosome, with fixed probability
        def Mutate(child, mutationProb, mutationSD):
            mutationBinaries = [np.random.choice([0,1], p=[1-mutationProb, mutationProb], size=i.shape) for i in child]
            mutationTerms = [np.random.normal(0, mutationSD, size=i.shape) for i in child]
            mutatedChild = child
            for i in range(len(child)):
                mutatedChild[i] = child[i] + mutationBinaries[i] * mutationTerms[i]
            return mutatedChild

        def run():

            # evaluate weights and assign performance
            fitness = self.evaluate()
            pairs = select(fitness, int(len(fitness)/2))
            crossover(pairs)


            return np.max(fitness)

        done = False
        lastGenBestFitness = np.max(self.evaluate())
        while not done:
            bestGenFitness = run()
            if abs(lastGenBestFitness - bestGenFitness) < .01*lastGenBestFitness: done = True

    def diffEvo(self, xoP):
            # xoP = crossover probabiliy
        def mutate():
            trialV = "mutated weights"
            return trialV

        def crossover(trialV):
            children = "crossed over trial vectors with original vectors"
            return children

        def pick():
            best = "picks which is best of child or parent"
            return best

            # mutates then crosses over weights
        def evolve():
            self.weights = [pick(crossover(mutate(i))) for i in self.weights]
            self.bweights = [pick(crossover(mutate(i))) for i in self.bweights]


    def SBO(self):
        pass

def identity(x):
    return x

def softMax(o):
    val = np.exp(o) / np.sum(np.exp(o))
    return val

def f1_score(prediction, actual):
    np.seterr(divide='ignore', invalid='ignore')
    m = pd.crosstab(prediction, actual)  # confusion matrix
    i = m.index.union(m.columns)
    m = m.reindex(index=i, columns=i, fill_value=0)  # makes the matrix square
    m.fillna(0)  # fills na values w/ 0
    P = precision(m)  # calculates precision
    R = recall(m)  # calcaultes recall
    f1 = 2 * (R * P) / (P + R)  # calculates f1
    f1 = f1[~np.isnan(f1)]  # drops na values
    f1 = np.sum(f1) / f1.shape[0]  # gets average of all f1 scores
    np.seterr(divide='warn', invalid='warn')
    return f1

def precision(m):
    M = m.to_numpy()
    diag = np.diag(M)  # true positives
    p = diag / np.sum(M, axis=0)  # true positives / TP + false positives
    return p

def recall(m):
    M = m.to_numpy()
    diag = np.diag(M)  # true positives
    r = diag / np.sum(M, axis=1)  # true positives / TP + false negatives
    return r

abalone = Dataset("abalone")
abalone.networks[2].geneticAlgorithm()