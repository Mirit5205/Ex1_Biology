import subprocess
import sys

import pkg_resources

requierd = {'matplotlib', 'numpy'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = requierd - installed
if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout = subprocess.DEVNULL)
    
import random

from matplotlib.animation import FuncAnimation

import numpy as np
import matplotlib.pyplot as plt

INTERACTIVE_PLOT = 0

'''
N - population size

D - percentage of sicks

R - percentage of fast creatures

RECOVERY_TIME - number of generations until creature that get sick
                recover.

INFECTION_PROBABILITY_ABOVE_THRESHOLD - if there is above of T % of sicks
                                        the probability to get infected decrease
                                        (the creature become more careful)
                                        
INFECTION_PROBABILITY_BELOW_THRESHOLD - if there is below of T % of sicks
                                        the probability to get infected increase

THRESOLD - when the percentage of sicks is greater than
           THRESOLD precent of the population,
           people start to watch out and
           obey the instructions

'''

N = 2000
D = 1
R = 5
RECOVERY_TIME = 7
INFECTION_PROBABILITY_ABOVE_THRESHOLD = 0.01
INFECTION_PROBABILITY_BELOW_THRESHOLD = 0.1
THRESOLD = 10

'''
GENERATION - counting the generations, every generation is 
             a frame of the plot
                
NUM_OF_SICKS - counting the number of sicks, appear in the plot
'''

GENERATION = 0
NUM_OF_SICKS = 0


# plot use this matrix data:
# there is 1 where there is a creature
# 2 where there is a sick creature
# 0 when the cell is empty

matrix = np.zeros(shape=(200, 200), dtype=int)

# Creates a list containing 200 lists, each of 200 items, all set to 0
w, h = 200, 200
creatures_matrix = [[0 for x in range(w)] for y in range(h)]

class Creature:

    # symbol:
    # 1 - healthy
    # 5 - sick
    # fast, sick and recover are booleans
    # disease_onset:
    # the generation when the creature got infected
    # recovery_from_illness:
    # the generation when the creature should recover
    symbol = 1
    fast = 0
    sick = 0
    recover = 0
    disease_onset = -1
    recovery_from_illness = - 1

    def __init__(self, location):
        self.location = location
        self.neighbors = []

    def move(self):
        # check if the creature should recover
        # in this generation
        if self.recovery_from_illness == GENERATION:
            self.sick = 0
            self.recover = 1
            self.disease_onset = -1
            self.recovery_from_illness = -1
            self.symbol = 1

            # decrease number of sicks
            global NUM_OF_SICKS
            NUM_OF_SICKS -= 1

        # if the creature sick
        # there is a chance to get infected neighbors
        if self.sick == 1:
            infect(self)

        '''
        this is the BEGINNING of the part in the code that make sure
        that no two creatures will place in the same cell
        '''
        # get neighbors locations:
        # if it's simple creature only the creatures that are 1 step from him
        # if it's fast creature only the creatures that are 10 step from him
        location_possibilities = get_neighbors_locations(self.location, self.fast)

        # get three optional locations
        locations = random.sample(location_possibilities, 3)

        # try to move to the first location.
        # if it's taken by another creature try to move to the next location
        # if all the 3 locations are taken, stay in place.
        for l in locations:

            # the creature choose to stay in place
            if l[0] == self.location[0] and l[1] == self.location[1]:
                break
            new_i = l[0]
            new_j = l[1]
            if creatures_matrix[new_i][new_j] == 0:
                creatures_matrix[self.location[0]][self.location[1]] = 0
                self.location[0] = new_i
                self.location[1] = new_j
                creatures_matrix[self.location[0]][self.location[1]] = self
                self.update_neighbors()
                break
        '''
        this is the END of the part in the code that make sure
        that no two creatures will place in the same cell
        '''

    # after the creature move, probably his neighbors changes too
    def update_neighbors(self):
        neighbors_locations = get_neighbors_locations(self.location, self.fast)
        for n in neighbors_locations:
            i = n[0]
            j = n[1]
            if n == self.location:
                continue
            if creatures_matrix[i][j] != 0:
                self.neighbors.append(creatures_matrix[i][j])

    def __str__(self):
        return f"i: {self.location[0]}, j: {self.location[1]} " \
               f"fast: {self.fast}, sick: {self.sick}"

# create matrix for the plot based on the creatures
# locations and values (sick, healthy)
def creatures_locations_to_matrix(creatures):
    matrix = np.zeros(shape=(200, 200), dtype=int)

    for c in creatures:
        i = c.location[0]
        j = c.location[1]
        matrix[i][j] = c.symbol

    return matrix

def initial_num_of_creatures(N):
    # inital number of sicks according to user input
    n = 0

    creatures = []
    while True:
        if n == N:
            break
        i = random.randint(0, 199)
        j = random.randint(0, 199)
        if creatures_matrix[i][j] == 0:
            c = Creature([i, j])
            creatures_matrix[i][j] = c
            creatures.append(c)
            n += 1

    return creatures

def initial_num_of_sicks(creatures, D):
    # for example given num_of_persons = 100, percentage_of_sicks = 10
    # num_of_sicks = 100 * (10 / 100) = 10
    num_of_creatures = len(creatures)
    num_of_sicks = int(num_of_creatures * (D / 100))

    n = 0

    while True:
        if n == num_of_sicks:
            break

        k = random.randint(0, num_of_creatures - 1)

        if creatures[k].symbol != 5:
            creatures[k].symbol = 5
            creatures[k].sick = 1
            creatures[k].disease_onset = 0
            creatures[k].recovery_from_illness = 0 + RECOVERY_TIME
            global NUM_OF_SICKS
            NUM_OF_SICKS += 1
            n += 1

def initial_num_of_fast_creatures(creatures, R):
    # for example given num_of_persons = 100, percentage_of_sicks = 10
    # num_of_sicks = 100 * (10 / 100) = 10
    num_of_creatures = len(creatures)
    num_of_fast_creatures = int(num_of_creatures * (R / 100))

    n = 0

    while True:
        if n == num_of_fast_creatures:
            break

        k = random.randint(0, num_of_creatures - 1)

        if creatures[k].fast == 0:
            creatures[k].fast = 1
            n += 1

# N = num of creatures - default value 2000
# D = percentage of sick creatures default value 10%
# R = percentage of fast creatures default value 1%
def initial_population(N = 2000, D = 10, R = 1):

    creatures = initial_num_of_creatures(N)
    for c in creatures:
        c.update_neighbors()
    initial_num_of_sicks(creatures, D)
    initial_num_of_fast_creatures(creatures, R)

    return creatures

# get neighbors locations of simple creature:
# only the neighbors that are 1 step from him
def get_neighbors_locations_of_simple_creature(i, j):

    locations = [[i, j], [i, j + 1], [i, j - 1], [i + 1, j], [i - 1, j],
                 [i + 1, j + 1], [i - 1, j - 1], [i + 1, j - 1], [i - 1, j + 1]]

    for l in locations:
        if l[0] == -1:
            l[0] = 199
        elif l[0] == 200:
            l[0] = 0
        if l[1] == -1:
            l[1] == 199
        if l[1] == 200:
            l[1] = 0

    return locations

# get neighbors locations of fast creature:
# only the neighbors that are 10 step from him
def get_neighbors_locations_of_fast_creature(i, j):
    locations = [[i, j], [i, j + 10], [i, j - 10], [i + 10, j], [i - 10, j],
                 [i + 10, j + 10], [i - 10, j - 10], [i + 10, j - 10], [i - 10, j + 10]]

    for l in locations:
        if l[0] < 0:
            l[0] += 200
        elif l[0] > 199:
            l[0] -= 200
        if l[1] < 0:
            l[1] += 200
        if l[1] > 199:
            l[1] -= 200

    return locations

# return the creature neighbors based on his type - fast or simple
def get_neighbors_locations(location, is_fast):
    i = location[0]
    j = location[1]

    if is_fast:
        locations = get_neighbors_locations_of_simple_creature(i, j)
    else:
        locations = get_neighbors_locations_of_fast_creature(i, j)

    return locations

# return the infection probability based on
# precentage of sicks (above threshold or below)
def get_infection_probability():
    if (NUM_OF_SICKS / N) > (THRESOLD / 100):
        p1 = INFECTION_PROBABILITY_ABOVE_THRESHOLD

        # p1 is the chance to infect
        # 1 - p1 is the chance to not infect
        p = [1 - p1, p1]
    else:
        p2 = INFECTION_PROBABILITY_BELOW_THRESHOLD
        p = [1 - p2, p2]

    return p

# try to infect neighbors
def infect(creature):
    # if the creature is not sick he cant infect
    if creature.sick != 1:
        return

    # for every neighbor:
    for n in creature.neighbors:
        # if the neighbor is already sick or
        # if he recovered from the disease
        # he can't get infect again
        if n.sick == 1 or n.recover == 1:
            continue
        global NUM_OF_SICKS, N
        p = get_infection_probability()


        # get 0 with probability 1 - p and 1 with probability p
        is_infected = np.random.choice(np.arange(0, 2), p=p)

        # if the neighbor got infect
        # increase number of sicks by 1
        # and change the appropriate filed of him
        if is_infected == 1:
            NUM_OF_SICKS += 1
            n.sick = 1
            n.symbol = 5
            n.disease_onset = GENERATION
            n.recovery_from_illness = GENERATION + RECOVERY_TIME

def interactive_plot():
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)
    cbar = fig.colorbar(im, ax=ax)
    text = ax.text(-10, - 10, 'Number Of Sicks {}'.format(NUM_OF_SICKS))
    ani = FuncAnimation(fig, update, fargs=(text, im), interval=1000)
    plt.show()

# simulate n generations
def plot(n):

    # x axis values
    x = []

    # corresponding y axis values
    y = []

    for i in range(n):
        x.append(i)
        y.append(NUM_OF_SICKS)
        global GENERATION
        for c in creatures:
            c.move()
        GENERATION += 1

    # plotting the points
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel('Generation')
    # naming the y axis
    plt.ylabel('Number Of Sicks')

    # giving a title to my graph
    #plt.title('My first graph!')

    # function to show the plot
    plt.show()

def update(i, text, im):
    global GENERATION
    GENERATION += 1
    for c in creatures:
        c.move()
    matrix = creatures_locations_to_matrix(creatures)
    im.set_data(matrix)
    global NUM_OF_SICKS
    text.set_text('Sicks {}'.format(NUM_OF_SICKS))

    im.autoscale()

# choose between two options:
# 1. running with default parameters values
# 2. running with user custom parameters values
def choose_running_option():
    global INTERACTIVE_PLOT
    INTERACTIVE_PLOT = validate_y_n_query("If You Want To Watch The Interactive Plot, Enter 1."
                                                "\nOtherwise Please Enter 0.\n")

    while True:
        opt = input("If You Want To Run Ex1 With The Default Parameters Values "
                    "(values that lead to 3 Morbidity Waves) please enter d.\n"
                    "If You Want To Run Ex1 With The Custom Parameters Values "
                    "please enter c.\n")
        if opt == 'c':
            user_input_params()
            break
        elif opt == 'd':
            break
        else:
            print(f"{opt} Is Invalid Option.\n")

def validate_int_value(text):
    while True:
        try:
            num = int(input(text))
            break
        except ValueError:
            print("Please Input Integer only")
            continue
    return num

def validate_y_n_query(text):
    while True:
        try:
            num = int(input(text))
            if num != 0 and num != 1:
                raise ValueError
            break
        except ValueError:
            print("Please Input 0 or 1 Only")
            continue
    return num

def user_input_params():
    global N,D,R,RECOVERY_TIME,THRESOLD,INFECTION_PROBABILITY_ABOVE_THRESHOLD,INFECTION_PROBABILITY_BELOW_THRESHOLD
    N = validate_int_value("Please Enter Population Size (N):\n")
    D = validate_int_value("Please Enter Primary Percentage Of Sick Creatures (D) - if you choose 10%, just enter 10:\n")
    R = validate_int_value("Please Enter Percentage Of Fast Creatures (R) (if you choose 10%, just enter 10):\n")
    RECOVERY_TIME = validate_int_value("Please Enter Time Until Recovery (X) - in terms of generations:\n")
    print("When The Percentage Of Sicks Is Greater Than T Percent Of The Population,"
          " People Start To Watch Out And"
          " Obey The Instructions")
    THRESOLD = validate_int_value("Please Enter Threshold (T) - Percentage Of The Population:\n")
    INFECTION_PROBABILITY_ABOVE_THRESHOLD = validate_int_value("Please Enter Probability Of Infection In Case The Percentage"
                                                      " Of Sicks Is Greater Than T Percent (if you choose 10%, just enter 10):\n") / 100
    INFECTION_PROBABILITY_BELOW_THRESHOLD = validate_int_value("Please Enter Probability Of Infection In Case The Percentage"
                                                      " Of Sicks Is Less Than T Percent (if you choose 10%, just enter 10):\n") / 100
'''
print("Please Enter Population Size (N): \n")
print("Please Enter Primary Percentage Of Sick Creatures (D) - if you choose 10%, just enter 10: \n")
print("Please Enter Percentage Of Fast Creatures (R) (if you choose 10%, just enter 10): \n")
print("Please Enter Time Until Recovery (X) - in terms of generations: \n")
print("When The Percentage Of Sicks Is Greater Than T Percent Of The Population,"
      "People Start To Watch Out And"
      "Obey The Instructions")
print("Please Enter Threshold (T) - Percentage Of The Population)")
print("Please Enter Probability Of Infection In Case The Percentage"
      " Of Sicks Is Greater Than T Percent (if you choose 10%, just enter 10): \n")
print("Please Enter Probability Of Infection In Case The Percentage"
      " Of Sicks Is Less Than T Percent (if you choose 10%, just enter 10): \n")
'''


choose_running_option()
creatures = initial_population(N, D, R)
if INTERACTIVE_PLOT == 1:
    interactive_plot()
else :
    plot(100)
