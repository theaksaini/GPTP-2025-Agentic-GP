from langchain_ollama import OllamaLLM as Ollama

import gp
import benchmarks
import re
import ast

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
def run_tests(llm_reply):
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
        result1 = gp.gp_loop(parent_selection=func, dataset=benchmarks.generate_dataset(benchmarks.target_func1))
        result2 = gp.gp_loop(parent_selection=func, dataset=benchmarks.generate_dataset(benchmarks.target_func2))
        result = [result1, result2]
        return result, None
    except Exception as e:
        return [], str(e)
    

if __name__ == "__main__":
    task ="You are an advanced Genetic Programming specialist." \
    "Write a python function for parent selection called selection(population, fitnesses) that takes a population of individuals and their fitnesses as input, both given as lists." \
    "For example, fitnesses[i] is the fitness of population[i]. Fitness[i] is a list of errors on test cases. We want to minimize all the errors to 0. "\
    "The function should return an individual, i.e., one of the elements in population. " \
    "The GP system we are working with has these characteristics: (a) tree-based representation, (b) 'add', 'sub', and 'mul' as non-terminals, (c) 'x', -2, -1, 0, 1, and 2 as terminals." \
    "The GP system is initialized with the following parameters: population size as 50, miximum number of generations as 100, crossover rate as 0.8, mutation rate as 0.2."\
    "Write the selection function such that when used within the complete genetic programming (GP) system, it leads to minimizing the best fitness found during the GP run, denoted by 'best_of_run_f.'" 

    feedback = ""

    for i in range(5):
        code = generate_code(task, feedback)
        print(f"Generated code:\n{code}\n")
        results, error = run_tests(code)
        if error:
            feedback = f"The code failed to run: {error}"
            print(feedback)
        else:
            print("The best fitness found per benchmark problem is: ", results)
            feedback = f"The code ran successfully. The best fitness found per benchmark problem is {results}. Improve the code to maximize these fitness values found during the GP runs."
       
