import numpy as np
from MLP import MLP
import random
#GA Experimentation Pseudocode; Trains a population of MLPs and uses a GA to select the best one

class GAModel:
    #initializes with a population of solutions, a probability of crossover, a probability of mutation, and a mutation SD
    def __init__(self, solutionPopulation = [], probOfCrossover = .1, probOfMutation = .05, mutationSD = .01):
        self.solutionPopulation = solutionPopulation
        self.populationSize = len(solutionPopulation)
        self.populationFitnesses = []
        self.probOfCrossover = probOfCrossover
        self.probOfMutation = probOfMutation
        self.mutationSD = mutationSD
        for model in self.solutionPopulation:
            # calculate fitnesses of all models in population
            MLP_model = MLP(weights = model)
            self.populationFitnesses.append(MLP_model.Train())

    #training function, iterates through generations until the best model in a given generation doesn't improve sufficiently
    #from previous generation
    def Train(self):
        converged = False
        maxSolutionFitness = 0
        lastBestFitness = 0
        generations = 0
        timesWithoutChange = 0
        while not converged:
            newSolutionPopulation = []
            newPopulationFitnesses = []
            #performs n/5 reproductions, will probably need to figure out how to decide this later
            for i in range(int(self.populationSize/5)):

                #easier to return the indices in the list of the parents selected
                parent1, parent2 = self.selectParents()

                #crossover the 2 selected parents
                child1, child2 = self.crossParents(self.solutionPopulation[parent1], self.solutionPopulation[parent2])

                #randomly mutate some genes in each child
                child1 = self.Mutate(child1)
                child2 = self.Mutate(child2)

                #using generational replacement, can adjust to steady state later if we want

                #remove parents from original population so not selected multiple times (should they be able to be
                # selected multiple times?)
                parent_indices = [parent1, parent2]
                self.solutionPopulation = [i for j, i in enumerate(self.solutionPopulation) if j not in parent_indices]
                self.populationFitnesses = [i for j, i in enumerate(self.solutionPopulation) if j not in parent_indices]

                #add children to the "next generation"
                newSolutionPopulation.append(child1)
                newSolutionPopulation.append(child2)

                #add children performances to the "next generation"
                newPopulationFitnesses.append(MLP(weights=child1).Train())
                newPopulationFitnesses.append(MLP(weights=child2).Train())

            #set the current generation population to the newly generated population of children
            self.solutionPopulation.append(newSolutionPopulation)
            self.populationFitnesses.append(newPopulationFitnesses)

            #continue until the best solution in the children population doesn't improve that much, and return
            #that solution
            maxSolutionFitness = max(self.populationFitnesses)
            bestSolution = self.solutionPopulation[np.argmax(self.populationFitnesses)]

            #convergence conditions: if 50 generations have passed, or if 10 generations in a row have passed without
            #a .1% change in best population fitness, converge
            if generations == 50: converged = True
            if abs(maxSolutionFitness-lastBestFitness) < .001*lastBestFitness:
                timesWithoutChange = timesWithoutChange+1
                if timesWithoutChange == 10: converged = True
            else: timesWithoutChange = 0
            lastBestFitness = maxSolutionFitness

        return bestSolution, maxSolutionFitness

    #select 2 parents to reproduce and create 2 children
    def selectParents(self):
        #sum up all fitnesses across entire solution population
        fitnessSum = np.sum(self.populationFitnesses)

        #calcuate probabilities of each solution being selected for reproduction
        selectionProbabilities = self.populationFitnesses/fitnessSum

        #choosing 2 parents based on probabilities
        parent1, parent2 = np.random.choice(range(self.populationSize), size=2, p=selectionProbabilities)
        return parent1, parent2

    #cross parents over, uniform crossover (with 10% probability) implemented but could switch to one point or two point
    def crossParents(self, parent1, parent2):

        #need to flatten weight matrices into 1d arrays to get into "chromosome" form
        #not sure syntactically the best way to do this, can think of some shitty ways though
        parentChromosome1 = np.flatten(parent1.weights)
        parentChromosome2 = np.flatten(parent2.weights)

        childChromosome1 = parentChromosome1
        childChromosome2 = parentChromosome2

        # uniform crossover
        # swap the genes from the two parents in the children with some fixed probability
        #iterate through all genes in the chromosomes
        for i in len(parentChromosome1):
            chanceOfCrossover = np.random.uniform(0,1)
            if chanceOfCrossover < self.probOfCrossover:
                childChromosome1[i] = parentChromosome2[i]
                childChromosome2[i] = parentChromosome1[i]

        #one point crossover
        #crossoverPoint = random.randint(0, len(parentChromosome1))
        #childChromosome1[0:crossoverPoint] = parentChromosome2[0:crossoverPoint]
        #childChromosome2[crossoverPoint+1:len(parentChromosome1)-1] = parentChromosome1[crossoverPoint+1:len(parentChromosome1)-1]

        #two point crossover
        #point1 = random.randint(0, len(parentChromosome1))
        #point2 = random.randint(0, len(parentChromosome1))
        #childChromosome1[min(point1, point2):max(point1, point2)] = parentChromosome2[min(point1, point2):max(point1, point2)]
        #childChromosome2[min(point1, point2):max(point1, point2)] = parentChromosome1[min(point1, point2):max(point1, point2)]

        #return children as new solutions with child chromosomes as weight matrix, all else held the same
        child1 = childChromosome1
        child2 = childChromosome2
        return child1, child2


    #randomly alter some genes in a given solution's chromosome, with fixed probability
    def Mutate(self, solution):

        #again, probably an optimal way to get to this 1d array, or might not even be necessary
        chromosome = solution

        #iterate through all genes, mutate some % with a term from a normal distribution with fixed SD (need to tune)
        for i in len(chromosome):
            chanceOfMutation = np.random.uniform(0,1)
            if chanceOfMutation < self.probOfMutation:
                chromosome[i] = chromosome[i] + np.random.normal(0, self.mutationSD)

        mutatedSolution = chromosome

        return mutatedSolution


