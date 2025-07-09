
def target_func1(x): 
    # return x*x*x + x*x + x + 1
    return x*x*x + x + 2

def target_func2(x): 
    # return x*x*x + x*x + x + 1
    return x*x*x*x + x*x*x + x*x + x + 1

def generate_dataset(target_func): # generate 101 data points from target_func
    dataset = []
    for x in range(-100,101,2): 
        dataset.append([x, target_func(x)])
    return dataset
