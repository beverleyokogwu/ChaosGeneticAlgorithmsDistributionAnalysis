'''
This is a test experiment to know whether self-adaptation is possible.


Comments are deliberately not cut out to allow for users to explore
different combinations.

Author: Beverley-Claire Okogwu
'''
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import copy
from scipy.stats import norm
import seaborn as sns

# generate random Gaussian values
from numpy.random import seed
from numpy.random import randn

genarrayav=[]
genarraymin=[]
mutd_values_CGA =[] #store the mutated values
mutd_values_CGA2 =[] #store the mutated values of 2nd r param
mutd_values_GA =[] #store the mutated values

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
    #plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.ylim([0,5000])
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

    #Shift affects the mean, and scale affects the standard deviation!
    def shift_scale_next(self):
        return (self.next_val() + self.shift) * self.scale

# Cubic Map class
class Cubic:
    def __init__(self,x_0, r, shift, scale):
        self.x = x_0
        self.r = r
        self.shift= shift
        self.scale = scale

    def next_val(self):
        self.x = (self.r * self.x *((self.x**2)-1.0))+self.x
        return self.x

    def shift_scale_next(self):
        return (self.next_val() + self.shift) * self.scale

def logisticMap(r, x_n):
    return r * x_n *(1.0-x_n)

#unimodal
def rosenbrock(array):
    #summation, sigma
    sigma=0
    # length of the array
    d = len(array)

    for elem in range(1,d-1):
        sigma+=(100*((array[elem+1]-(array[elem])**2)**2)+(array[elem]-1)**2)
    return sigma

def griewank(array):

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


#multimodal
def rastrigin(array):
    #summation, sigma
    #optimum value
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
def evaluate_fitness(population):

    fitarray=[]

    for index in range(len(population)):

        #bm_value = rosenbrock(population[index][0])
        bm_value = rastrigin(population[index][0])
        fitarray.append(bm_value)

        # changes the fitness value
        population[index][1]=bm_value
    average = sum(fitarray)/len(fitarray)#average fitness of individuals at some generation
    genarrayav.append(average)
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

def crossover(p1,p2,probability):
    #print("In Crossover Function... \n parent1 = {}\n parent2 ={}".format(p1,p2))

    parentLen = len(p1)
    num = random.uniform(0.0,1.0)
    if num < probability:
        crossPoint= random.randint(0,parentLen)
        #print("crosspoint={}".format(crossPoint))
        newKid1= p1[:crossPoint]
        newKid2= p2[crossPoint:]
        newKid = newKid1+newKid2
        #print("The new Kid is {}".format(newKid))
        return newKid
    else:
        #print("No crossover, so returning p1 = {}".format(p1))
        return copy.deepcopy(p1)


def mutate(individual,probability,algorithm_object,generation_num,self_adapt_cutoff):
    #print("IN MUTATE......")
    #print("GENERATION NUMBER PASSED : {}".format(generation_num))

    #for each individual's genes, get a random number between 0 and 1
    for gene in range(len(individual)):
        num = random.uniform(0.0,1.0)
        if num < probability:
            n_val=algorithm_object.shift_scale_next()

            if generation_num <= self_adapt_cutoff:
                #print("The generation is {}, so we add {}, param: {} to mutd_values_CGA".format(generation_num,n_val, algorithm_object.r))
                mutd_values_CGA.append(n_val)
            else:
                #print("The generation is GREATER THAN 250: {}, so we add {} from LM parameter {} to mutd_values_CGA2".format(generation_num,n_val,algorithm_object.r))
                mutd_values_CGA2.append(n_val)

            individual[gene]+=n_val

    return individual



def EA(map,gen_size,probabilitym,default_fitness,pop,probabilityc,self_adapt_cutoff):


    #evaluate the fitness
    evaluate_fitness(pop)

    #get the fittest individual
    fittest = find_fittest(pop)

    #this is the first (zeroth) generation
    gen = 0

    # so far within constraints
    while fittest[1]>0 and gen <gen_size:

        if gen > self_adapt_cutoff:
            #print("The generation {} is greater than 2!!!............".format(gen))

            #just change r
            map.r = 4.0

        # the best is the new pop -> may/may not do this | elitism
        #print("Getting the fittest individual to start a new population")
        new_population = [fittest]

        #[g,g,g,g,g,g]
        #add to the new population:
        for i in range(len(pop)-1):
            #pick 2 parents
            #print("Let's pick two parents: \t")
            p1 = get_parent(pop)
            p2 = get_parent(pop)
            #print(p1)
            #print("\t")
            #print(p2)
            #crossover
            #print("Let's crossover")
            potential_child = crossover(p1,p2,probabilityc)
            #potential_child = copy.deepcopy(p1)
            #mutation
            #print("\t\tpotential child is: {}".format(potential_child))
            #print("Let's mutate")
            child = mutate(potential_child,probabilitym,map,gen,self_adapt_cutoff)
            new_population.append([child, default_fitness])

        #make the new-population the next pop to work with
        pop = new_population

        #generation increases by 1
        gen+=1

        #evaluate the fitness of this population
        evaluate_fitness(pop)

        #make the best
        fittest = find_fittest(pop)
        #print("Generation {}: Fittest: {}".format(gen,fittest))
    print("Ran Successfully!")



def self_adaptation_test_1():

    #Run first with one r value
    for i in range(num_trails):
        #print("\nTRAIL NUMBER {}".format(i))
        pop = generatePopulation(rdm, population_size, individual_size)
        pop_CGA = copy.deepcopy(pop)
        EA(lm,gen_size,probabilitym,default_fitness,pop_CGA,probabilityc,self_adapt_cutoff)# Run the CGA

    #print("Our mutation values are: ")
    CGA_mutd_values_first_r = mutd_values_CGA.copy()
    CGA_mutd_values_second_r = mutd_values_CGA2.copy()
    #print("CGA_mutd_values_first_r --> {}".format(CGA_mutd_values_first_r))
    #print("\n\nCGA_mutd_values_second_r --> {}".format(CGA_mutd_values_second_r))
    title1= 'Shift-Scale Distributions for CGA with r = 3.7'
    title2= 'Shift-Scale Distributions for CGA with r = 4.0'

    print("NOW PLOTTING HISTOGRAMS!!!!!!!!")
    plotHistogram(CGA_mutd_values_first_r,title1)
    plotHistogram(CGA_mutd_values_second_r,title2)



def self_adaptation_test_2():

    #Run first with one r value

    #print("\nTRAIL NUMBER {}".format(i))
    pop = generatePopulation(rdm, population_size, individual_size)
    pop_CGA = copy.deepcopy(pop)
    EA(lm,gen_size,probabilitym,default_fitness,pop_CGA,probabilityc,self_adapt_cutoff)# Run the CGA

    #print("Our mutation values are: ")
    CGA_mutd_values_first_r = mutd_values_CGA.copy()
    CGA_mutd_values_second_r = mutd_values_CGA2.copy()
    #print("CGA_mutd_values_first_r --> {}".format(CGA_mutd_values_first_r))
    #print("\n\nCGA_mutd_values_second_r --> {}".format(CGA_mutd_values_second_r))
    title1= 'Shift-Scale Distributions for CGA with r = 3.7'
    title2= 'Shift-Scale Distributions for CGA with r = 4.0'

    print("NOW PLOTTING HISTOGRAMS!!!!!!!!")
    plotHistogram(CGA_mutd_values_first_r,title1)
    plotHistogram(CGA_mutd_values_second_r,title2)




'''
DRIVER
'''

#Use variables & make code more flexible
initial_x_0 = 0.02
cm_parameter = 3.7
lm_shift= -0.5
lm_scale = 2
rdm_shift = 0
rdm_scale = 0.5
lm = LM(initial_x_0, cm_parameter,lm_shift, lm_scale)
rdm = Gauss(rdm_shift,rdm_scale)
probabilitym = 0.05
probabilityc = 0.8
gen_size = 500
self_adapt_cutoff = 250
default_fitness= math.inf
num_trails=50
population_size=50
individual_size=20


self_adaptation_test_2()
