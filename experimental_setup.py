from copy import deepcopy
import random
import gp
import benchmarks
import time
import os
import pickle
import llm_agent
import sys

TOURNAMENT_SIZE = 5    # size of tournament for tournament selection

def lexicase_selection(population, fitnesses): # select one individual using lexicase selection
    candidates = list(zip(population, fitnesses)) # list of tuples (individual, fitness)
    cases = list(range(len(fitnesses[0]))) # all cases
    for case in cases:
        if len(candidates) <= 1:
                break
        errors_this_case = [pf[1][case] for pf in candidates]
        best_val_for_case = min(errors_this_case)
        candidates = [i for i in candidates if i[1][case] <= best_val_for_case]
    return deepcopy(random.choice(candidates)[0]) # return one individual randomly from the remaining candidates

def tournament_selection(population, fitnesses): # select one individual using tournament selection
    candidates = list(zip(population, fitnesses)) # list of tuples (individual, fitness)
    tournament = random.sample(candidates, TOURNAMENT_SIZE) # select random individuals for tournament
    best_fitness = min([t[1] for t in tournament]) # find best fitness in tournament
    best_individuals = [t[0] for t in tournament if t[1] == best_fitness] # select individuals with best fitness
    return deepcopy(random.choice(best_individuals)) # return one individual randomly from the best individuals

def random_selection(population, fitnesses): # select one individual randomly
    return deepcopy(random.choice(population)) # return one individual randomly from the population
    
def loop_through_tasks(exps_and_reps: dict, base_save_folder, saved_responses_folder:str, pop_size:int, max_gens:int):
    # exps_and_reps: dict, e.g. {"lexicase": 100, "tournament": 100, AGP1: [10, 10], "AGP2": [20,5]}
    
    for exp, num_runs in exps_and_reps.items():
        if exp == "lexicase" or exp == "tournament" or exp == "random":
            for taskid in ["task1", "task2"]:
                save_folder = f"{base_save_folder}/{exp}/{taskid}"
                for r in range(num_runs):
                    save_file = f"{save_folder}/{r}.pkl"
                    time.sleep(random.random()*5)
                    if os.path.exists(save_file):
                        continue
                    else:
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)

                        print("working on ")
                        print(save_folder)

                        parent_selection = lexicase_selection if exp == "lexicase" else tournament_selection if exp == "tournament" else random_selection
                        func = benchmarks.target_func1 if taskid == "task1" else benchmarks.target_func2

                        best_fitness, best_individual, best_gen = gp.gp_loop(pop_size=pop_size, max_gens=max_gens, parent_selection=parent_selection, dataset=benchmarks.generate_dataset(func))
                        result = {
                            "best_fitness": best_fitness,
                            "best_individual": best_individual,
                            "best_gen": best_gen,
                            "task_id": taskid,
                        }

                        with open(save_file, "wb") as f:
                            pickle.dump(result, f)

            return

        elif exp == "AGP1" or exp == "AGP2":
            num_iters, num_runs_per_iter = num_runs
            for iter in range(num_iters):
                print(f"Running AGP1 iteration {iter+1}/{num_iters}")
                code = llm_agent.extract_llm_response(exp, iter, saved_responses_folder)  # results["task1"] contain the results for first task, and so on.
                print(f"Generated code:\n{code}\n")
                for r in range(num_runs_per_iter):
                    for taskid in ["task1", "task2"]:
                        output, error = llm_agent.run_tests(pop_size, max_gens, code, taskid)
                        if error:
                            feedback = f"The code failed to run: {error}"
                            print(feedback)
                            best_fitness, best_individual, best_gen = sys.maxsize, None, None
                        else:
                            print("The best fitness found per benchmark problem is: ", output)
                            feedback = f"The code ran successfully. The best fitness found per benchmark problem is {output}. Improve the code to maximize these fitness values found during the GP runs."
                            best_fitness, best_individual, best_gen = output  # Assuming results is a list of tuples (fitness, individual, gen)

                        results = {
                            "best_fitness": best_fitness,
                            "best_individual": best_individual,
                            "best_gen": best_gen,
                            "task_id": taskid,
                        }
                        save_folder = f"{base_save_folder}/{exp}/{taskid}"
                        if not os.path.exists(save_folder):
                            os.makedirs(save_folder)
                        rep = iter * num_runs_per_iter + r # To keep the output files consistent with other experiments
                        save_file = f"{save_folder}/{rep}.pkl"
                        with open(save_file, "wb") as f:
                            #print(results[taskid][rep])
                            pickle.dump(results, f)
            return

        else:
            raise ValueError("Invalid experiment name")

    print("all finished")
