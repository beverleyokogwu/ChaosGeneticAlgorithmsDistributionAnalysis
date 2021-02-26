#import statements
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import copy
import inspect
import sys
# generate random Gaussian values
from numpy.random import seed
from numpy.random import randn

#genarrayav=[]
genarraymin=[]
#mutd_values_CGA =[] #store the mutated values
#mutd_values_GA =[] #store the mutated values

#Histogram plotting function
def plotHistogram(map_array, name):


    n, bins, patches = plt.hist(x=map_array, bins=20, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(name)

    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

# Logistic Class
class LM:
    def __init__(self, x_0, r, shift, scale):
        self.x = x_0
        self.r = r
        self.shift= shift
        self.scale = scale

    def next_val(self):
        self.x = self.r * self.x *(1.0-self.x)
        return self.x

    def shift_scale_next(self):
        return (self.next_val() + self.shift) * self.scale

# Gaussian class - for GA
class Gauss:
    def __init__(self, shift, scale):
        self.shift= shift
        self.scale = scale

    def next_val(self):
        return np.random.normal()

    def shift_scale_next(self):
        return (self.next_val() + self.shift) * self.scale

def logisticMap(r, x_n):
    return r * x_n *(1.0-x_n)

def Rosenbrock(array):
    #summation, sigma
    sigma=0
    # length of the array
    d = len(array)

    for elem in range(1,d-1):
        sigma+=(100*((array[elem+1]-(array[elem])**2)**2)+(array[elem]-1)**2)
    return sigma

def Griewank(array):

    #summation, sigma
    sigma=0
    #product, pi
    pi=1
    # length of the array
    d = len(array)

    for val in range(1,d):
        sigma+=math.pow(array[val],2)
        pi*=math.cos(float(array[val])/math.sqrt(val))

    return (float(sigma)/4000)-float(pi)+1


def Rastrigin(array):
    #summation, sigma
    sigma=0
    # length of the array
    d = len(array)

    for el in range(1,d):
        sigma+=((array[el]**2)-(10*math.cos(2*math.pi*array[el])))

    return (10*d)+sigma

def generatePopulation(obj, pop_size, indiv_size):
    population =[]
    for i in range(pop_size):
        values=[]
        for i in range(indiv_size):
            values.append(obj.shift_scale_next())
        population.append([values,math.inf])
    return population



#.           [0 ]                       [1]
#       0                    1        0     1
#[[[array of 100 LM values],inf], [[array],inf]...]
def evaluate_fitness(population,benchMark):

    fitarray=[]

    for index in range(len(population)):


        if benchMark == Rastrigin:
            bm_value = Rastrigin(population[index][0])
        elif benchMark == Rosenbrock:
            bm_value = Rosenbrock(population[index][0])
        else:
            bm_value = Griewank(population[index][0])

        fitarray.append(bm_value)

        # changes the fitness value
        population[index][1]=bm_value
    #average = sum(fitarray)/len(fitarray)#average fitness of individuals at some generation
    #genarrayav.append(average)
    genarraymin.append(min(fitarray))
    #genarraymax.append(max(fitarray))
    #print(len(genarrayav))


def find_fittest(populat):
    min = populat[0][1]
    winnerIndividual=populat[0]

    for item in range(len(populat)):
        if populat[item][1]< min:
            min=populat[item][1]
            winnerIndividual=populat[item]

    return winnerIndividual

def get_parent(pop):
    #print("In pick Parent")
    lengthOfPop = len(pop)
    #print("The length is {}".format(lengthOfPop))
    randomItem1 = random.randint(0,lengthOfPop-1)
    parentA=pop[randomItem1][0]
    #print("Parent A is {}".format(parentA))
    Afit = pop[randomItem1][1]
    #print("Parent A's fitness is {}".format(Afit))

    randomItem2 = random.randint(0,lengthOfPop-1)
    parentB=pop[randomItem2][0]
    #print("Parent B is {}".format(parentB))
    Bfit = pop[randomItem2][1]
    #print("Parent B's fitness is {}".format(Afit))

    if Afit < Bfit:
        #print("{} is < than {}, so return {}".format(Afit,Bfit,Afit))
        return parentA
    else:
        #print("{} is < than {}, so return {}".format(Bfit,Afit,Bfit))
        return parentB

def crossover(p1,p2):
    parentLen = len(p1)
    crossPoint= random.randint(0,parentLen)
    newKid1= p1[:crossPoint]
    newKid2= p2[crossPoint:]
    newKid = newKid1+newKid2
    return newKid


def mutate(individual,probability,algorithm_object):


    #for each individual's genes, get a random number between 0 and 1
    for gene in range(len(individual)):
        num = random.uniform(0.0,1.0)
        if num < probability:
            n_val=algorithm_object.shift_scale_next()
            individual[gene]+=n_val

    return individual


# THE GA DRIVER

#generate the initial population  [[[array of 100 LM values],inf], [,]...] //total = 200
'''
This is a simple function/code that calculates the average and std. dev
for n trails of running the GA/CGA
'''




#Use variables & make code more flexible


#map = lm
probability = 0.05
gen_size = 500
default_fitness= math.inf
num_trails=50
population_size=50
individual_size=20

def EA(map,gen_size,probability,default_fitness,pop,bm):


    #evaluate the fitness
    evaluate_fitness(pop,bm)

    #get the fittest individual
    fittest = find_fittest(pop)

    #this is the first (zeroth) generation
    gen = 0

    # so far within constraints
    while fittest[1]>0 and gen <gen_size:
        # the best is the new pop -> may/may not do this | elitism
        new_population = [fittest]

        #add to the new population:
        for i in range(len(pop)-1):
            #pick 2 parents
            p1 = get_parent(pop)
            p2 = get_parent(pop)
            #crossover
            potential_child = crossover(p1,p2)
            #mutation
            child = mutate(potential_child,probability,map)
            new_population.append([child, default_fitness])

        #make the new-population the next pop to work with
        pop = new_population

        #generation increases by 1
        gen+=1

        #evaluate the fitness of this population
        evaluate_fitness(pop,bm)

        #make the best
        fittest = find_fittest(pop)
        #print("Generation {}: Fittest: {}".format(gen,fittest))
    print("Ran Successfully!")

def getMinFitness(xZero,lambda_val,l_min_fitness_array,benchmark):
    genarraymin.clear()
    lm = LM(xZero, lambda_val,-0.5,2)
    rdm = Gauss(0,0.5)
    pop = generatePopulation(rdm, population_size, individual_size)
    EA(lm,gen_size,probability,default_fitness,pop,benchmark)
    mn = genarraymin.copy()
    l_min_fitness_array.append(mn)

def plotMapParameters(l1,l2,l3,l4,l5,benchmark):
    #list containing proposed x0s
    xZeros=[0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4,0.497]

    '''
    Run with each of these values and run it with those results for 10-20 trails.
    Similar to the GA2.py file.
    '''

    '''
    Run with average and pick a random x0 each time to plot it. (Like GA2.py for every trail.)

    Then run the CGA VS GA with it.
    EXPECTATION: Should be the same.
    '''




    #store the values to plot
    l1_minFitness =[]
    l2_minFitness =[]
    l3_minFitness =[]
    l4_minFitness =[]
    l5_minFitness =[]

    #run 10 instances...
    for i in range(len(xZeros)):

        getMinFitness(xZeros[i],l1,l1_minFitness,benchmark)
        getMinFitness(xZeros[i],l2,l2_minFitness,benchmark)
        getMinFitness(xZeros[i],l3,l3_minFitness,benchmark)
        getMinFitness(xZeros[i],l4,l4_minFitness,benchmark)
        getMinFitness(xZeros[i],l5,l5_minFitness,benchmark)




    #convert to np ARRAYS
    l1_minFitness = np.array(l1_minFitness)
    l2_minFitness = np.array(l2_minFitness)
    l3_minFitness = np.array(l3_minFitness)
    l4_minFitness = np.array(l4_minFitness)
    l5_minFitness = np.array(l5_minFitness)

    """
    for index in range(len(l1_minFitness)):
        print(index)
        print(l3_minFitness[index])
    """
    #Let's plot
    x = [i for i in range(gen_size+1)]


    plt.plot(x,l1_minFitness[0],color="red",label=l1)
    plt.plot(x,l2_minFitness[0],color="orange",label=l2)
    plt.plot(x,l3_minFitness[0],color="green",label=l3)
    plt.plot(x,l4_minFitness[0],color="blue",label=l4)
    plt.plot(x,l5_minFitness[0],color="purple",label=l5)

    for index in range(1,len(l3_minFitness)):
        plt.plot(x,l1_minFitness[index],color="red")
        plt.plot(x,l2_minFitness[index],color="orange")
        plt.plot(x,l3_minFitness[index],color="green")
        plt.plot(x,l4_minFitness[index],color="blue")
        plt.plot(x,l5_minFitness[index],color="purple")


    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title('Logistic Map on {} Fitness Function'.format(benchmark.__name__))

    # show a legend on the plot
    plt.legend()
    plt.show()





plotMapParameters(3.6,3.7,3.8,3.9,4.0,Rastrigin)
#plotMapParameters(3.8,3.9,4.0,Rosenbrock)
#plotMapParameters(3.8,3.9,4.0,Griewank)
