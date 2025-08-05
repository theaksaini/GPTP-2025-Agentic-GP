import llm_agent
import sys
from experimental_setup import loop_through_tasks

def proportion_of_valids():
    # Prompt the LLM 100 times and see how many of them are valid
    # To test the validity: run 10 pop size * 50 gens for 10 replicates
    save_folder = "experiment1_responses"
    print("The number of valid codes generated in Experiment 1 out of 100 is:")
    return llm_agent.generate_and_test(num_trials=100, save_folder=save_folder)

def save_valids(saved_responses_folder):
    # We can save the valid responses first, and then run the GP on them
    for exp in ["AGP1", "AGP2"]:
        num_iters = 10 if exp == "AGP1" else 20
        for iter in range(num_iters):
            print(f"Running {exp} iteration {iter+1}/{num_iters}")
            code = llm_agent.extract_llm_response(exp, iter, saved_responses_folder)  # results["task1"] contain the results for first task, and so on.
            print(f"Generated code:\n{code}\n")

if __name__ == "__main__":
    experiments = {"lexicase": 100, "tournament": 100, "random": 100, "AGP1": [10, 10], "AGP2": [20,5]}
    results_folder = "experimental_results"
    saved_responses_folder = "experiment2_valid_responses"

    # Experiment 1
    # Prompt the LLM 100 times and see how many of them are valid
    # print(proportion_of_valids())

    # Experiment 2
    # Step 1: Save the responses from the LLM
    # save_valids(saved_responses_folder)

    # Step 2: Run the experiments
    pop_size = 100
    max_gens = 500
    exp = sys.argv[1] # e.g. "lexicase", "tournament", "random", "AGP1", "AGP2"
    loop_through_tasks({exp: experiments[exp]}, results_folder, saved_responses_folder, pop_size, max_gens)
