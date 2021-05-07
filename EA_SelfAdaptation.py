'''
Python Implementations of the Self-Adaptation Experiment from thesis
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
mutd_values_GA =[] #store the mutated values
gen_r_values = [] #stores the  r values for each generation
all_r_vals_avg = []
r_dist = []
r_dist_0=[]


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
# [[[1,2,3,3.8],inf],[[3,4,5,4.0], inf]]
def rastrigin(array):
    #print("IN RASTRIGIN. RECEIVED LAST VALUE AS:")
    #print(array[-1])
    #print(min(array))
    #print(max(array))
    #print(" ")

    #summation, sigma
    #optimum value
    sigma=0
    # length of the array
    d = len(array)

    for el in range(1,d):
        sigma+=((array[el]**2)-(10*math.cos(2*math.pi*array[el])))

    return (10*d)+sigma

#Histogram plotting function
def plotHistogram(map_array, name, ylimit):


    n, bins, patches = plt.hist(x=map_array, bins=20, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(name)

    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    #plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.ylim([0,ylimit])
    plt.xlim([3.5,4.0])
    plt.show()

def generatePopulation(obj, pop_size, indiv_size):
    population =[]
    # [[[1,2,3,r],inf],[[3,4,5,r2], inf]]

    for i in range(pop_size):
        gene_values=[]

        for i in range(indiv_size):
            gene_values.append(obj.shift_scale_next())
        #add a random r value
        possible_r_value = random.uniform(3.6,4.0)

        gene_values.append(possible_r_value)
        r_dist_0.append(possible_r_value)
        population.append([gene_values,math.inf])
    return population



#.           [0 ]                       [1]
#       0                    1        0     1
#[[[array of 100 LM values,r],inf], [[array],inf]...]
def evaluate_fitness(population):

    fitarray=[]

    for index in range(len(population)):

        #bm_value = rosenbrock(population[index][0])
        #print(population[index][0][:-1])
        #print(len(population[index][0][:-1]))
        #print(population[index][0][-1])

        # [[[1,2,3]],[[3,4,5]]]
        bm_value = rastrigin(population[index][0][:-1])#don't calcutate fitness for r value
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




# With Self-Adaptation
def mutate(individual,probability,algorithm_object,start,stop,gen):

    #for each individual's genes, get a random number between 0 and 1
    for gene in range(len(individual)-1):#excluding the r
        num = random.uniform(0.0,1.0)
        if num < probability:
            r_val = individual[-1] #get the r value for that gene

            if algorithm_object == lm:#CGA

                algorithm_object.r = r_val
                n_val=algorithm_object.shift_scale_next()
                #CV  = r * x *(1.0-x)
                mutd_values_CGA.append(n_val)

            elif algorithm_object == rdm:#GA

                n_val=algorithm_object.shift_scale_next()
                mutd_values_GA.append(n_val)

            individual[gene]+=n_val
    #make a smaller standard deviation
    rdm.scale= 0.01
    sigma_r=rdm.shift_scale_next() # adding the sigma to the r

    # make sure it's not out of range
    new_r = individual[-1]+sigma_r

    if new_r > 4.0:
        new_r = 4.0
    elif new_r < 3.57:
        new_r = 3.57

    if gen >= start and gen <= stop:
        r_dist.append(new_r)# this piece deals with the distributions of r values.

    individual[-1]=new_r

    return individual



# THE GA/CGA DRIVER

#generate the initial population  [[[array of 100 LM values],inf], [,]...] //total = 200

def EA(map,gen_size,probabilitym,default_fitness,pop,probabilityc,start):


    #evaluate the fitness
    evaluate_fitness(pop)



    #get the fittest individual
    fittest = find_fittest(pop)

    #this is the first (zeroth) generation
    gen = 0

    # so far within constraints
    while fittest[1]>0 and gen <gen_size:
        # the best is the new pop -> may/may not do this | elitism
        new_population = [fittest]

        #[g,g,g,g,g,g]
        #add to the new population:
        for i in range(len(pop)-1):
            #pick 2 parents
            p1 = get_parent(pop)
            p2 = get_parent(pop)
            #crossover
            potential_child = crossover(p1,p2,probabilityc)
            #potential_child = copy.deepcopy(p1)
            #mutation
            #print("\t\tpotential child is: {}".format(potential_child))
            child = mutate(potential_child,probabilitym,map,start,gen_size,gen)
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
    print("Ran Successfully! End of EA")



#Compute the average mutation gen_size
def avg_mutation_size_computation():
    #Run the CGA
    mut_size_total = 0.0
    for _ in range(num_trails):

        pop = generatePopulation(rdm, population_size, individual_size)
        EA(lm,gen_size,probabilitym,default_fitness,pop,probabilityc)

    mut_size = mutd_values_CGA.copy()
    for mutation in mut_size:
        mut_size_total += mutation

    avg_mutation_size = mut_size_total/ len(mut_size)
    print("BIAS for r = {} = {}".format(cm_parameter,avg_mutation_size))


def r_analysis():
    avg_r_all_trails = []

    for _ in range(num_trails):

        gen_r_values.clear()
        pop = generatePopulation(rdm, population_size, individual_size)
        pop_CGA = copy.deepcopy(pop)
        EA(lm,gen_size,probabilitym,default_fitness,pop_CGA,probabilityc)
        r_values = copy.deepcopy(all_r_vals_avg)
        avg_r_all_trails.append(r_values)

    print("r_values length: {}".format(len(r_values)))
    print("avg_r_all_trails length {}".format(len(avg_r_all_trails)))

    avg_r_all_trails_np = np.array(avg_r_all_trails)
    print("avg_r_all_trails_np length: {}".format(len(avg_r_all_trails_np)))

    avg_r = np.mean(avg_r_all_trails_np,axis=0)
    print("avg_r length: {}".format(len(avg_r)))

    #plot
    gens = [i for i in range(gen_size+1)]
    plt.plot(gens, avg_r, label = "Average r")
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title('Average of R values per Generation')
    plt.legend()
    plt.show()

#what do the distributions look like at any given point in time
def r_hist(generation):
    for _ in range(num_trails):
        if generation == 0:
            pop = generatePopulation(rdm, population_size, individual_size)
        else:
            pop = generatePopulation(rdm, population_size, individual_size)
            pop_CGA = copy.deepcopy(pop)
            EA(lm,generation,probabilitym,default_fitness,pop_CGA,probabilityc)
    if generation == 0:
        plotHistogram(r_dist_0,'R Distributions for Generation: {}'.format(generation),ylimit)
    else:
        plotHistogram(r_dist,'R Distributions for {} Generations'.format(generation),ylimit)


def r_intervals(start, stop):
    for _ in range(num_trails):
        pop = generatePopulation(rdm, population_size, individual_size)
        pop_CGA = copy.deepcopy(pop)
        EA(lm,stop,probabilitym,default_fitness,pop_CGA,probabilityc,start)
        plotHistogram(r_dist,'R Distributions for the interval [{} , {}] Generations'.format(start, stop),ylimit)




def plots():
    #mean and standard deviation plots
    avgs2D_CGA=[]
    mins2D_CGA=[]

    avgs2D_GA=[]
    mins2D_GA=[]


    #DEALS WITH CGA & GA (merged ^^)
    for i in range(num_trails):

        #reset
        genarrayav.clear()
        genarraymin.clear()


        pop = generatePopulation(rdm, population_size, individual_size)

        #make copies--> adjustment for approach#1. the arrays are independent of each other.
        pop_GA = copy.deepcopy(pop)
        pop_CGA = copy.deepcopy(pop)

        EA(lm,gen_size,probabilitym,default_fitness,pop_CGA,probabilityc)# Run the CGA

        av = genarrayav.copy()
        mn = genarraymin.copy()

        genarrayav.clear()
        genarraymin.clear()

        EA(rdm,gen_size,probabilitym,default_fitness,pop_GA,probabilityc)# Run the GA

        avGA = genarrayav.copy()
        mnGA = genarraymin.copy()

        avgs2D_CGA.append(av) # add the average fitness across generations
        mins2D_CGA.append(mn)# add the min fitness across generations to the 2D array


        #Append for GA:
        avgs2D_GA.append(avGA) # add the average fitness across generations
        mins2D_GA.append(mnGA)# add the min fitness across generations to the 2D array


    CGA_mutd_values = mutd_values_CGA.copy()
    GA_mutd_values = mutd_values_GA.copy()


    #at the end, should have two 2D arrays
    #use the np mean and std dev to compute them in an array
    avgs2D_CGA = np.array(avgs2D_CGA)
    mins2D_CGA = np.array(mins2D_CGA)
    avgs2D_GA = np.array(avgs2D_GA)
    mins2D_GA = np.array(mins2D_GA)



    std_avgs_CGA = np.std(avgs2D_CGA,axis=0)
    std_mins_CGA = np.std(mins2D_CGA,axis=0)
    avg_avgs_CGA = np.mean(avgs2D_CGA,axis=0)
    avg_mins_CGA = np.mean(mins2D_CGA,axis=0)


    #GA
    std_avgs_GA = np.std(avgs2D_GA,axis=0)
    std_mins_GA = np.std(mins2D_GA,axis=0)
    avg_avgs_GA = np.mean(avgs2D_GA,axis=0)
    avg_mins_GA = np.mean(mins2D_GA,axis=0)

    #plot for averages
    x = [i for i in range(gen_size+1)]

    """
    plt.errorbar(x, avg_avgs_CGA, std_avgs_CGA, label = "average-fitness of averages (CGA)")
    plt.errorbar(x, avg_mins_CGA, std_mins_CGA,  label = "average-fitness of mins (CGA)")
    plt.errorbar(x, avg_avgs_GA, std_avgs_GA,  label = "average-fitness of averages (GA)")
    plt.errorbar(x, avg_mins_GA, std_mins_GA,  label = "average-fitness of mins (GA)")
    """

    #REGULAR PLOTS AVG-AVG


    plt.plot(x, avg_avgs_CGA,  label = "average-fitness of averages (CGA)")
    plt.plot(x, avg_avgs_GA,  label = "average-fitness of averages (GA)")
    # naming the x axis
    plt.xlabel('generation')
    # naming the y axis
    plt.ylabel('fitness')
    # giving a title to my graph
    plt.title('Average of Average Fitness per Generation')

    # show a legend on the plot
    plt.legend()
    plt.show()

    #REGULAR PLOTS AVG-MINS
    plt.plot(x, avg_mins_CGA,   label = "average-fitness of mins (CGA)")
    plt.plot(x, avg_mins_GA,  label = "average-fitness of mins (GA)")

    # naming the x axis
    plt.xlabel('generation')
    # naming the y axis
    plt.ylabel('fitness')
    # giving a title to my graph
    plt.title('Average of Min Fitness per Generation')

    # show a legend on the plot
    plt.legend()
    plt.show()


    '''
    #SUBPLOTS FOR ANG-AVG EAs
    fig,ax = plt.subplots(2,2)
    ax[0,0].errorbar(x, avg_avgs_GA,yerr=std_avgs_CGA )
    ax[0,0].set_title('Average of Average Fitness per Genaration (GA)')
    ax[0,0].grid('on')

    ax[1,0].errorbar(x, avg_avgs_CGA,yerr=std_avgs_GA,color='green')
    ax[1,0].set_title('Average of Average Fitness per Genaration (CGA)')
    ax[1,0].grid('on')

    ax[0,1].errorbar(x, avg_mins_GA,yerr=std_mins_CGA,color='orange')
    ax[0,1].set_title('Average of Min Fitness per Genaration (GA)')
    ax[0,1].grid('on')

    ax[1,1].errorbar(x, avg_mins_CGA,yerr=std_mins_GA,color='red')
    ax[1,1].set_title('Average of Min Fitness per Genaration (CGA)')
    ax[1,1].grid('on')
    #plt.show()

    #SUBPLOTS FOR DISTRIBUTIONS
    '''


    #plotHistogram(CGA_mutd_values,'Shift-Scale Distributions for CGA')
    #plotHistogram(GA_mutd_values,'Shift-Scale Distributions for GA')

    plt.show()


#Use variables & make code more flexible
initial_x_0 = 0.02
cm_parameter = 3.6 #dummy value; will change
lm_shift= -0.5
lm_scale = 2
rdm_shift = 0
rdm_scale = 0.5
lm = LM(initial_x_0, cm_parameter,lm_shift, lm_scale)
rdm = Gauss(rdm_shift,rdm_scale)
probabilitym = 0.05
probabilityc = 0.8
gen_size = 500
default_fitness= math.inf
num_trails=1
population_size=50
individual_size=20
ylimit = 1500




#r_analysis()
#plots()
#avg_mutation_size_computation()
for i in range(0,1000,100):
    r_intervals(i,i+100)
    r_dist.clear()






#EA(lm,gen_size,probability,default_fitness)
'''
        #add new r value to external list
        # 1 r value per indiv
        # 49 indivs per gen
        # 500 gen
        # need avg r for each gen
        #print("LEN OF GEN_R_VALUES: {}".format(len(gen_r_values)))
        if len(gen_r_values) < 50:

            #covers the 49 indivs per gen
            gen_r_values.append(new_r)

        if len(gen_r_values)==49:

            avg_r_per_gen = sum(gen_r_values)/49
            all_r_vals_avg.append(avg_r_per_gen) #add avg r per gen
            #print("all r array is {}".format(all_r_vals_avg))
            gen_r_values.clear() # clear for the next gen

'''
