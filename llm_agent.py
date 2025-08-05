from langchain_ollama import OllamaLLM as Ollama

import gp
import benchmarks
import re
import ast
import sys
import os
import pickle


TASK = "You are an advanced Genetic Programming specialist." \
    "Write a python function for parent selection called selection(population, fitnesses) that takes a population of individuals and their fitnesses as input, both given as lists." \
    "For example, fitnesses[i] is the fitness of population[i]. Fitness[i] is a list of errors on test cases. We want to minimize all the errors to 0. "\
    "The function should return an individual, i.e., one of the elements in population. " \
    "The GP system we are working with has these characteristics: (a) tree-based representation, (b) 'add', 'sub', and 'mul' as non-terminals, (c) 'x', -2, -1, 0, 1, and 2 as terminals." \
    "The GP system is initialized with the following parameters: population size as 50, miximum number of generations as 100, crossover rate as 0.8, mutation rate as 0.2."\
    "Write the selection function such that when used within the complete genetic programming (GP) system, it leads to minimizing the best fitness found during the GP run, denoted by 'best_of_run_f.'" 


def extract_python_code_block(text):
    """Extract code from a markdown-style Python code block."""
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_python_libraries(code):
    """Extract all imported libraries from Python code."""
    match = re.findall(r"^\s*import\s+(\w+)", code, re.MULTILINE)
    return match

def extract_functions(code):
    """Extract all top-level function definitions from Python code."""
    tree = ast.parse(code)
    functions = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_code = ast.get_source_segment(code, node)
            if func_code:
                functions.append(func_code)
    return functions

# 2. LLM setup
llm = Ollama(model="llama3.1") # 8B parameter model

# 3. Main loop
def generate_code(task_description, feedback=""):
    prompt = f"{task_description}\n{feedback}\nPlease provide only the Python function. Do not import any libraries other than numpy (import numpy as np). "
    return llm(prompt)

# 4. Test runner
def run_tests(pop_size, max_gens, llm_reply, taskid: str):
    try:
        namespace = {}
        code = """import math;"""
        code += """import numpy as np;"""
        codes = compile(code, '<string>', 'exec')
        exec(codes, namespace)
        code = extract_python_code_block(llm_reply)
        funcs = extract_functions(code)
        exec(funcs[0], namespace)
        for key in namespace.keys():
            globals().pop(key, None)
            globals()[key] = namespace.get(key)
        func = namespace.get('selection')
        
        print("The type of the function is: ", type(func))
        print("Function found in the code: ", func)

        target_func = benchmarks.target_func1 if taskid == "task1" else benchmarks.target_func2
        output = gp.gp_loop(pop_size=pop_size, max_gens=max_gens, parent_selection=func, dataset=benchmarks.generate_dataset(target_func))
        return output, None
    except Exception as e:
        return None, str(e)

def extract_llm_response(exp:str, iter:int, save_folder:str, generate_only_valid:bool=True):
    task =TASK

    save_file = f"{save_folder}/{exp}_iter{iter}.pkl"

    if os.path.exists(save_file):
        with open(save_file, 'rb') as f:
            response = pickle.load(f)
    else:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if not generate_only_valid:
            response = generate_code(task, "")
            with open(save_file, "wb") as f:
                pickle.dump(response, f)
        else:
            for trial in range(500):
                response = generate_code(task, feedback="")
                if is_valid_code(response):
                    with open(save_file, "wb") as f:
                        pickle.dump(response, f)
                        return response
                else:
                    continue

            with open(save_file, "wb") as f: # if no valid code is generated after 500 trials, save the last response
                pickle.dump(response, f)

    return response

def is_valid_code(code):
    """Check if the code is valid Python code."""
    for rep in range(10):
        for taskid in ["task1", "task2"]:
            output, error = run_tests(pop_size=10, max_gens=50, llm_reply=code, taskid=taskid)
            if error: # Either the fucntion has a syntactic error 
                feedback = f"The code failed to run: {error}"
                print(feedback)
                return False
            if output[1] is None: #or it has infinite loop
                feedback = f"The code failed to run: {error}"
                print(feedback)
                return False
            
    return True  # The code is valid if it runs without errors

def generate_and_test(num_trials:int, save_folder:str):
    """Generate and test code for a number of trials."""
    num_valid = 0
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for trial in range(num_trials):
        code = generate_code(TASK, feedback="")
        save_file = f"{save_folder}/trial_{trial}.pkl"
        with open(save_file, "wb") as f:
            pickle.dump(code, f)
        if is_valid_code(code):
            print(f"Trial {trial + 1}: Code is valid.")
            num_valid += 1
        else:
            print(f"Trial {trial + 1}: Code is invalid.")

    return num_valid

def llm_agent(num_reps:int):
    task ="You are an advanced Genetic Programming specialist." \
    "Write a python function for parent selection called selection(population, fitnesses) that takes a population of individuals and their fitnesses as input, both given as lists." \
    "For example, fitnesses[i] is the fitness of population[i]. Fitness[i] is a list of errors on test cases. We want to minimize all the errors to 0. "\
    "The function should return an individual, i.e., one of the elements in population. " \
    "The GP system we are working with has these characteristics: (a) tree-based representation, (b) 'add', 'sub', and 'mul' as non-terminals, (c) 'x', -2, -1, 0, 1, and 2 as terminals." \
    "The GP system is initialized with the following parameters: population size as 50, miximum number of generations as 100, crossover rate as 0.8, mutation rate as 0.2."\
    "Write the selection function such that when used within the complete genetic programming (GP) system, it leads to minimizing the best fitness found during the GP run, denoted by 'best_of_run_f.'" 

    feedback = "" # not using the feedback for now

    results = {"task1": [], "task2": []} # results["task1"] contain the results for first task, and so on.

    code = generate_code(task, feedback)
    print(f"Generated code:\n{code}\n")
    for rep in range(num_reps):
        for taskid in ["task1", "task2"]:
            output, error = run_tests(code, taskid)
            if error:
                feedback = f"The code failed to run: {error}"
                print(feedback)
                best_fitness, best_individual, best_gen = sys.maxsize, None, None
            else:
                print("The best fitness found per benchmark problem is: ", results)
                feedback = f"The code ran successfully. The best fitness found per benchmark problem is {results}. Improve the code to maximize these fitness values found during the GP runs."
                best_fitness, best_individual, best_gen = output  # Assuming results is a list of tuples (fitness, individual, gen)

            results[taskid].append({
                "best_fitness": best_fitness,
                "best_individual": best_individual,
                "best_gen": best_gen,
                "task_id": taskid,
            })


    return results