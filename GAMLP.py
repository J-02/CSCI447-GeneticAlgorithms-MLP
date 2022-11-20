import numpy as np
import pandas as pd
from MLP import MLP
import DataSpilt as ds
import random


# GA Experimentation Pseudocode; Trains a population of MLPs and uses a GA to select the best one

class GAModel:
    # initializes with a population of solutions, a probability of crossover, a probability of mutation, and a mutation SD
    def __init__(self, data, nodes, step_size=.001, momentum=.5, solutionPopulationWeights=[], probOfCrossover=.1,
                 probOfMutation=.05, mutationSD=.01):
        self.solutionPopulation = solutionPopulationWeights
        self.populationSize = len(solutionPopulationWeights)
        self.populationFitnesses = []
        self.probOfCrossover = probOfCrossover
        self.probOfMutation = probOfMutation
        self.mutationSD = mutationSD
        self.data = data
        self.nodes = nodes
        self.step_size = step_size
        self.momentum = momentum
        for model in self.solutionPopulation:
            # calculate fitnesses of all models in the solution population
            MLP_model = MLP(self.data, self.nodes, None, self.step_size, self.momentum)
            self.populationFitnesses.append(MLP_model.Train())

    # training function, iterates through generations until the best model in a given generation doesn't improve sufficiently
    # from previous generation
    def Train(self):
        converged = False
        lastBestFitness = 0
        generations = 0
        timesWithoutChange = 0
        while not converged:
            print('Generation', generations + 1)
            newSolutionPopulation = []
            newPopulationFitnesses = []
            # performs n/5 reproductions, will probably need to figure out how to decide this later
            for i in range(int(self.populationSize / 5)):
                # easier to return the indices in the list of the parents selected
                parent1, parent2 = self.selectParents()

                # crossover the 2 selected parents
                child1, child2 = self.crossParents(self.solutionPopulation[parent1], self.solutionPopulation[parent2])

                # randomly mutate some genes in each child
                child1 = self.Mutate(child1)
                child2 = self.Mutate(child2)

                # using generational replacement, can adjust to steady state later if we want

                # remove parents from original population so not selected multiple times (should they be able to be
                # selected multiple times?)
                parent_indices = [parent1, parent2]
                self.solutionPopulation = [i for j, i in enumerate(self.solutionPopulation) if j not in parent_indices]
                self.populationFitnesses = [i for j, i in enumerate(self.solutionPopulation) if j not in parent_indices]

                # add children to the "next generation"
                newSolutionPopulation.append(child1)
                newSolutionPopulation.append(child2)

                # add children performances to the "next generation"
                newPopulationFitnesses.append(MLP(self.data, self.nodes, child1, self.step_size, self.momentum).Train())
                newPopulationFitnesses.append(MLP(self.data, self.nodes, child2, self.step_size, self.momentum).Train())

            # set the current generation population to the newly generated population of children
            self.solutionPopulation.append(newSolutionPopulation)
            self.populationFitnesses.append(newPopulationFitnesses)

            # continue until the best solution in the children population doesn't improve that much, and return
            # that solution
            maxSolutionFitness = max(self.populationFitnesses)
            bestSolution = self.solutionPopulation[np.argmax(self.populationFitnesses)]

            # convergence conditions: if 50 generations have passed, or if 10 generations in a row have passed without
            # a .1% change in best population fitness, converge
            if generations == 50: converged = True
            if abs(maxSolutionFitness - lastBestFitness) < .001 * lastBestFitness:
                timesWithoutChange = timesWithoutChange + 1
                if timesWithoutChange == 10: converged = True
            else:
                timesWithoutChange = 0
            lastBestFitness = maxSolutionFitness
            print('This generation best fitness: ', maxSolutionFitness)

        return bestSolution, maxSolutionFitness

    # select 2 parents to reproduce and create 2 children
    def selectParents(self):
        # fitness proportionate
        # sum up all fitnesses across entire solution population
        fitnessSum = np.sum(self.populationFitnesses)
        # calcuate probabilities of each solution being selected for reproduction
        selectionProbabilities = self.populationFitnesses / fitnessSum

        # rank proportionate
        # sum up all ranks across entire solution population
        # rankSum = np.sum(range(self.populationSize+1))
        # find ranks of all models in population
        # sortedRanks = np.argsort(self.populationFitnesses)
        # define selection probabilities by these ranks
        # selectionProbabilities = sortedRanks+1/rankSum

        # choosing 2 parents based on probabilities
        parent1, parent2 = np.random.choice(range(self.populationSize), size=2, p=selectionProbabilities)
        return parent1, parent2

    # cross parents over, uniform crossover (with 10% probability) implemented but could switch to one point or two point
    def crossParents(self, parent1, parent2):

        # need to flatten weight matrices into 1d arrays to get into "chromosome" form
        # not sure syntactically the best way to do this, can think of some shitty ways though
        parentChromosome1 = parent1
        parentChromosome2 = parent2

        childChromosome1 = parentChromosome1
        childChromosome2 = parentChromosome2

        # uniform crossover
        # swap the genes from the two parents in the children with some fixed probability
        # iterate through all genes in the chromosomes
        crossoverBinaries = np.random.choice([0, 1], p=[1 - self.probOfCrossover, self.probOfCrossover],
                                             size=len(parentChromosome1))
        childChromosome1 = np.choose(crossoverBinaries, [parentChromosome1, parentChromosome2])
        childChromosome2 = np.choose(crossoverBinaries, [parentChromosome2, parentChromosome1])

        # one point crossover
        # crossoverPoint = random.randint(0, len(parentChromosome1))
        # childChromosome1[0:crossoverPoint] = parentChromosome2[0:crossoverPoint]
        # childChromosome2[crossoverPoint+1:len(parentChromosome1)-1] = parentChromosome1[crossoverPoint+1:len(parentChromosome1)-1]

        # two point crossover
        # point1 = random.randint(0, len(parentChromosome1))
        # point2 = random.randint(0, len(parentChromosome1))
        # childChromosome1[min(point1, point2):max(point1, point2)] = parentChromosome2[min(point1, point2):max(point1, point2)]
        # childChromosome2[min(point1, point2):max(point1, point2)] = parentChromosome1[min(point1, point2):max(point1, point2)]

        # return children as new solutions with child chromosomes as weight matrix, all else held the same
        child1 = childChromosome1
        child2 = childChromosome2
        return child1, child2

    # randomly alter some genes in a given solution's chromosome, with fixed probability
    def Mutate(self, solution):

        # again, probably an optimal way to get to this 1d array, or might not even be necessary
        chromosome = solution

        # iterate through all genes, mutate some % with a term from a normal distribution with fixed SD (need to tune)
        mutationBinaries = np.random.choice([0, 1], p=[1 - self.probOfMutation, self.probOfMutation],
                                            size=len(chromosome))
        mutationTerms = np.random.normal(0, self.mutationSD, size=len(chromosome))
        chromosome = chromosome + mutationBinaries * mutationTerms

        mutatedSolution = chromosome

        return mutatedSolution


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
        self.yolo = 0

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
        self.performance(y, solutions)

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

        def select(array):
            # dunno what to do here since population is from 10 fold CV
            return array

        def crossover(array):
            xover = np.random.choice([0, 1], p=[0.5, 0.5], size=(array.shape[0] // 2, array.shape[1], array.shape[2]))
            xover2 = (~xover.astype(bool)).astype(int)
            children1 = np.array([np.choose(xover[i], array[i::5]) for i in range(5)])
            children2 = np.array([np.choose(xover2[i], array[i::5]) for i in range(5)])
            children = np.vstack([children1, children2])
            return children

        # randomly alter some genes in a given solution's chromosome, with fixed probability
        def Mutate(children):
            # iterate through all genes, mutate some % with a term from a normal distribution with fixed SD (need to tune)
            mutationBinaries = np.random.choice([0, 1], p=[1 - prob, prob], size=children.size())
            mutationTerms = np.random.normal(0, SD, size=children)
            mutatedChildren = children + mutationBinaries * mutationTerms

            return mutatedChildren

        # crosses over then mutates weights
        def evolve():
            self.weights = [Mutate(crossover(i)) for i in self.weights]
            self.bweights = [Mutate(crossover(i)) for i in self.bweights]
        def run(prob=0.05, SD=0.01):
            self.pM = prob
            self.mSD = SD
            performance = []
            done = False
            while not done:
                pass

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
abalone.networks[2].evaluate()