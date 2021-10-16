import sys
from validation import *

def log_results(dataset_list):
    for dataset_name in dataset_list:
        #cross_validate_m1_without_prune(dataset)
        #cross_validate_m1_with_prune(dataset)
        #cross_validate_m2_without_prune(dataset)
        # cross_validate_m2_with_prune(dataset)
        paths = ["_m1_without_prune", "_m1_with_prune", "_m2_without_prune", "_m2_with_prune"]
        funcs = [cross_validate_m1_without_prune, cross_validate_m1_with_prune, cross_validate_m2_without_prune, cross_validate_m2_with_prune]
        for i in range(len(paths)):
            with open('./results/{}{}.txt'.format(dataset_name, paths[i]), 'w+') as fileobj:
                sys.stdout = fileobj
                funcs[i](dataset_name)
            sys.stdout = sys.__stdout__  # resetting stdout to the terminal
            print("Completed for", dataset_name+paths[i])

if __name__ == "__main__":
    log_results(["iris", "facebook", "zoo", "tic-tac-toe", "banknote_authentication"])