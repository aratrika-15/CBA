from new_preprocessing import preprocess_to_array,read_dataset
from cmar_rg_tester import rule_generator
import time
import random
from split_dataframe import define_splitting
from cmar_classifier import chi_sq_classifier
from cmar_pruner import *

def calculate_accuracy(ruleList, dataset):
    count = 0
    for case in dataset:
        pred = chi_sq_classifier(case, ruleList)
        if pred == case[-1]:
            count += 1
    return count/len(dataset)

def cmar_with_prune(dataset_name, minsup=0.01, minconf=0.5):
    df, schema = read_dataset(dataset_name)
    dataset = preprocess_to_array(df,schema)
    random.shuffle(dataset)

    # block_size = int(len(dataset) / 10)
    # split_point = [k * block_size for k in range(0, 10)]
    # split_point.append(len(dataset))
    split_point=define_splitting(dataset)

    cba_rg_total_runtime = 0
    cba_cb_total_runtime = 0
    total_car_number = 0
    # total_classifier_rule_num = 0
    error_total_rate = 0
    total_accuracy = 0

    for k in range(len(split_point)-1):
        print("\nRound %d:" % k)

        training_dataset = dataset[:split_point[k]] + dataset[split_point[k+1]:]
        test_dataset = dataset[split_point[k]:split_point[k+1]]

        start_time = time.time()
        cars = rule_generator(training_dataset, int(minsup*len(dataset)), minconf)
        # cars.prune_rules(training_dataset)
        # cars.rules = cars.pruned_rules


        # CMAR Stuff
        ruleList = list(cars.rules)
        ruleList = sort_rules(ruleList)
        ruleList = list(prune_with_cover(training_dataset, ruleList))



        end_time = time.time()
        cba_rg_runtime = end_time - start_time
        cba_rg_total_runtime += cba_rg_runtime

        # start_time = time.time()
        # classifier_m2 = classifier_builder_m2(cars, training_dataset)
        # end_time = time.time()
        # cba_cb_runtime = end_time - start_time
        # cba_cb_total_runtime += cba_cb_runtime

        # error_rate = get_error_rate(classifier_m2, test_dataset)
        # error_total_rate += error_rate

        # predictor_accuracy = get_accuracy(classifier_m2,test_dataset)
        # total_accuracy += predictor_accuracy

        # total_car_number += len(cars.rules)
        # total_classifier_rule_num += len(classifier_m2.rule_list)
        
        start_time = time.time()
        accuracy = calculate_accuracy(ruleList, test_dataset)
        end_time = time.time()
        cba_cb_runtime = end_time - start_time
        cba_cb_total_runtime += cba_cb_runtime
        
        total_accuracy += accuracy
        error_rate = 1 - accuracy
        error_total_rate += error_rate
        numRules = len(ruleList)
        total_car_number += numRules

        print("CMAR's error rate with pruning: %.1lf%%" % (error_rate * 100))
        print("No. of rules with pruning: %d" % numRules)
        print("CBA-RG's run time with pruning: %.2lf s" % cba_rg_runtime)
        print("CMAR's classification run time with pruning: %.2lf s" % cba_cb_runtime)
        # print("No. of rules in classifier of CBA-CB M2 with pruning: %d" % len(classifier_m2.rule_list))

    print("\nAverage CBA's error rate with pruning: %.1lf%%" % (error_total_rate / 10 * 100))
    print("Average CBA's accuracy with pruning: %.1lf%%" % (total_accuracy/ 10 * 100))
    print("Average No. of CARs with pruning: %d" % int(total_car_number / 10))
    print("Average CMAR's RG+pruning run time: %.2lf s" % (cba_rg_total_runtime / 10))
    print("Average CMAR's classification run time with pruning: %.2lf s" % (cba_cb_total_runtime / 10))
    # print("Average No. of rules in classifier of CBA-CB M2 with pruning: %d" % int(total_classifier_rule_num / 10))


# test entry goes here
if __name__ == "__main__":
    # using the relative path, all data sets are stored in datasets directory
    # dataset='iris'
    dataset = 'seeds'

    # just choose one mode to experiment by removing one line comment and running
    cmar_with_prune(dataset)
