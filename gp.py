# tiny genetic programming by © moshe sipper, www.moshesipper.com
# https://github.com/moshesipper/tiny_gp/
import random
from statistics import mean
from copy import deepcopy
import sys
import math
from func_timeout import func_timeout, FunctionTimedOut

#POP_SIZE        = 100   # population size
MIN_DEPTH       = 2    # minimal initial random tree depth
MAX_DEPTH       = 5    # maximal initial random tree depth
#GENERATIONS     = 500  # maximal number of generations to run evolution
XO_RATE         = 0.8  # crossover rate 
PROB_MUTATION   = 0.2  # per-node mutation probability 

def add(x, y): return x + y
def sub(x, y): return x - y
def mul(x, y): return x * y
FUNCTIONS = [add, sub, mul]
TERMINALS = ['x', -2, -1, 0, 1, 2] 

class GPTree:
    def __init__(self, data = None, left = None, right = None):
        self.data  = data
        self.left  = left
        self.right = right
        
    def node_label(self): # string label
        if (self.data in FUNCTIONS):
            return self.data.__name__
        else: 
            return str(self.data)
    
    def print_tree(self, prefix = ""): # textual printout
        print("%s%s" % (prefix, self.node_label()))        
        if self.left:  self.left.print_tree (prefix + "   ")
        if self.right: self.right.print_tree(prefix + "   ")

    def compute_tree(self, x): 
        if (self.data in FUNCTIONS): 
            return self.data(self.left.compute_tree(x), self.right.compute_tree(x))
        elif self.data == 'x': return x
        else: return self.data
            
    def random_tree(self, grow, max_depth, depth = 0): # create random tree using either grow or full method
        if depth < MIN_DEPTH or (depth < max_depth and not grow): 
            self.data = FUNCTIONS[random.randint(0, len(FUNCTIONS)-1)]
        elif depth >= max_depth:   
            self.data = TERMINALS[random.randint(0, len(TERMINALS)-1)]
        else: # intermediate depth, grow
            if random.random () > 0.5: 
                self.data = TERMINALS[random.randint(0, len(TERMINALS)-1)]
            else:
                self.data = FUNCTIONS[random.randint(0, len(FUNCTIONS)-1)]
        if self.data in FUNCTIONS:
            self.left = GPTree()          
            self.left.random_tree(grow, max_depth, depth = depth + 1)            
            self.right = GPTree()
            self.right.random_tree(grow, max_depth, depth = depth + 1)

    def mutation(self):
        if random.random() < PROB_MUTATION: # mutate at this node
            self.random_tree(grow = True, max_depth = 2)
        elif self.left: self.left.mutation()
        elif self.right: self.right.mutation() 

    def size(self): # tree size in nodes
        if self.data in TERMINALS: return 1
        l = self.left.size()  if self.left  else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self): # count is list in order to pass "by reference"
        t = GPTree()
        t.data = self.data
        if self.left:  t.left  = self.left.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t
                        
    def scan_tree(self, count, second): # note: count is list, so it's passed "by reference"
        count[0] -= 1            
        if count[0] <= 1: 
            if not second: # return subtree rooted here
                return self.build_subtree()
            else: # glue subtree here
                self.data  = second.data
                self.left  = second.left
                self.right = second.right
        else:  
            ret = None              
            if self.left  and count[0] > 1: ret = self.left.scan_tree(count, second)  
            if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)  
            return ret

    def crossover(self, other): # xo 2 trees at random nodes
        if random.random() < XO_RATE:
            second = other.scan_tree([random.randint(1, other.size())], None) # 2nd random subtree
            self.scan_tree([random.randint(1, self.size())], second) # 2nd subtree "glued" inside 1st tree

    def __deepcopy__(self, memo):
        # Prevent redundant copying of the same object
        if id(self) in memo:
            return memo[id(self)]
        copied = self.build_subtree()
        memo[id(self)] = copied
        return copied
    
# end class GPTree
                   
def init_population(pop_size): # ramped half-and-half
    pop = []
    for md in range(3, MAX_DEPTH + 1):
        for i in range(math.ceil(pop_size/6)):
            t = GPTree()
            t.random_tree(grow = True, max_depth = md) # grow
            pop.append(t) 
        for i in range(math.ceil(pop_size/6)):
            t = GPTree()
            t.random_tree(grow = False, max_depth = md) # full
            pop.append(t) 
    return pop[:pop_size] # return only pop_size individuals

def fitness(individual, dataset): # absolute error over dataset; unaggregated
    if individual.size() > 100: # bloat control; preventing overly large trees from propagating
        return [sys.maxsize for _ in dataset]
    else:
        return [abs(individual.compute_tree(ds[0]) - ds[1]) for ds in dataset]
                
def gp_loop(pop_size, max_gens, parent_selection, dataset): # main GP loop  
    random.seed() # init internal state of random number generator
    population= init_population(pop_size) 
    best_of_run = None
    best_of_run_f = sys.maxsize
    best_of_run_gen = 0
    fitnesses = [fitness(population[i], dataset) for i in range(pop_size)]

    # go evolution!
    for gen in range(max_gens):        
        nextgen_population=[]
        for i in range(pop_size):
            try: # Checking infinite loops
                parent1 = func_timeout(5*60, parent_selection, args=(population, fitnesses)) #parent1 = parent_selection(population, fitnesses)
                parent2 = func_timeout(5*60, parent_selection, args=(population, fitnesses)) #parent2 = parent_selection(population, fitnesses)
            except FunctionTimedOut:
                print("Function timed out. Skipping this generation.")
                return sys.maxsize, None, None
            parent1.crossover(parent2)
            parent1.mutation()
            nextgen_population.append(parent1)
        population=nextgen_population
        fitnesses = [fitness(population[i], dataset) for i in range(pop_size)]
        total_fitnesses = [sum(fitnesses[i]) for i in range(pop_size)]
        if min(total_fitnesses) < best_of_run_f:
            best_of_run_f = min(total_fitnesses)
            best_of_run_gen = gen
            best_of_run = population[total_fitnesses.index(min(total_fitnesses))]
            print("________________________")
            print("gen:", gen, ", best_of_run_f:", round(min(total_fitnesses),3), ", best_of_run:") 
            best_of_run.print_tree()
        if best_of_run_f == 0: break   
    
    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(best_of_run_gen) +\
          " and has f=" + str(round(best_of_run_f,3)))
    best_of_run.print_tree()
    
    return best_of_run_f, best_of_run, best_of_run_gen
