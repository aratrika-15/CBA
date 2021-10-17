"""
Description: This is our experimental code. We provide 4 test modes for experiments:
    10-fold cross-validations on CBA (M1) without pruning
    10-fold cross-validations on CBA (M1) with pruning
    10-fold cross-validations on CBA (M2) without pruning
    10-fold cross-validations on CBA (M2) with pruning
Input: the relative directory path of data file and scheme file
Output: the experimental results (similar to Table 1: Experiment Results in this paper)
Author: CBA Studio
"""

from new_preprocessing import preprocess_to_array,read_dataset
from cba_rg import rule_generator
from cba_rg import topKRules
from cba_cb_m1 import classifier_builder_m1
from cba_cb_m1 import is_satisfy
from cba_cb_m2 import classifier_builder_m2
import time
import random
from split_dataframe import define_splitting

def get_accuracy(classifier,dataset):
    size = len(dataset)
    # for case in dataset:
    #     is_satisfy_value = False
    #     for rule in classifier.rule_list:
    #         is_satisfy_value = is_satisfy(case, rule)
    #         if is_satisfy_value == True:
    #             break
    #     if is_satisfy_value == False:
    #         if classifier.default_class != case[-1]:
    #             error_number += 1
    # return error_number / size
    count = 0
    for case in dataset:
        is_satisfy_value=False
        for rule in classifier.rule_list:
            is_satisfy_value=is_satisfy(case,rule)
            if is_satisfy_value==True:
                count = count+1
                break
            if is_satisfy_value==False:
                break
        if is_satisfy_value==None:
            if(classifier.default_class==case[-1]):
                count = count+1
    print("value of count:",count)
    return count/size


# calculate the error rate of the classifier on the dataset
def get_error_rate(classifier, dataset):
    size = len(dataset)
    error_number = 0
    # for case in dataset:
    #     is_satisfy_value = False
    #     for rule in classifier.rule_list:
    #         is_satisfy_value = is_satisfy(case, rule)
    #         if is_satisfy_value == True:
    #             break
    #     if is_satisfy_value == False:
    #         if classifier.default_class != case[-1]:
    #             error_number += 1
    # return error_number / size
    for case in dataset:
        is_satisfy_value=False
        for rule in classifier.rule_list:
            is_satisfy_value=is_satisfy(case,rule)
            flag=0
            if is_satisfy_value==True:
                flag=1
                break
            if is_satisfy_value==False:
                flag=1
                error_number+=1
                break
        if flag==1:
            continue
        else:
            if classifier.default_class != case[-1]:
                error_number+=1
    return error_number/size


# 10-fold cross-validations on CBA (M1) without pruning
def cross_validate_m1_without_prune(dataset_name, minsup=0.01, minconf=0.5):
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
    total_classifier_rule_num = 0
    error_total_rate = 0
    total_accuracy = 0

    for k in range(len(split_point)-1):
        print("\nRound %d:" % k)

        training_dataset = dataset[:split_point[k]] + dataset[split_point[k+1]:]
        test_dataset = dataset[split_point[k]:split_point[k+1]]

        start_time = time.time()
        cars = rule_generator(training_dataset, minsup, minconf)

        print("CARs:(set)")
        #cars.print_rule()
        print("len:")
        print(len(cars.rules))
        print("Rule_list")
        #cars.print_rule_list()
        print("len:")
        print(len(cars.rule_list))
        for i in cars.rules:
            if i in cars.rule_list:
                # print("Hello")
                continue
            else:
                print("World")

        end_time = time.time()
        cba_rg_runtime = end_time - start_time
        cba_rg_total_runtime += cba_rg_runtime

        start_time = time.time()
        classifier_m1 = classifier_builder_m1(cars, training_dataset)
        end_time = time.time()
        cba_cb_runtime = end_time - start_time
        cba_cb_total_runtime += cba_cb_runtime

        error_rate = get_error_rate(classifier_m1, test_dataset)
        error_total_rate += error_rate

        predictor_accuracy = get_accuracy(classifier_m1,test_dataset)
        total_accuracy += predictor_accuracy

        total_car_number += len(cars.rules)
        total_classifier_rule_num += len(classifier_m1.rule_list)

        print("CBA's error rate without pruning: %.1lf%%" % (error_rate * 100))
        print("No. of CARs without pruning: %d" % len(cars.rules))
        print("CBA-RG's run time without pruning: %.2lf s" % cba_rg_runtime)
        print("CBA-CB M1's run time without pruning: %.2lf s" % cba_cb_runtime)
        print("No. of rules in classifier of CBA-CB M1 without pruning: %d" % len(classifier_m1.rule_list))
            
    print("\nAverage CBA's error rate without pruning: %.1lf%%" % (error_total_rate / 10 * 100))
    print("Average CBA's accuracy without pruning: %.1lf%%" % (total_accuracy/ 10 * 100))
    print("Average No. of CARs without pruning: %d" % int(total_car_number / 10))
    print("Average CBA-RG's run time without pruning: %.2lf s" % (cba_rg_total_runtime / 10))
    print("Average CBA-CB M1's run time without pruning: %.2lf s" % (cba_cb_total_runtime / 10))
    print("Average No. of rules in classifier of CBA-CB M1 without pruning: %d" % int(total_classifier_rule_num / 10))


# 10-fold cross-validations on CBA (M1) with pruning
def cross_validate_m1_with_prune(dataset_name, minsup=0.01, minconf=0.5):
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
    total_classifier_rule_num = 0
    error_total_rate = 0
    total_accuracy = 0

    for k in range(len(split_point)-1):
        print("\nRound %d:" % k)

        training_dataset = dataset[:split_point[k]] + dataset[split_point[k+1]:]
        print("Length of training dataset: ",len(training_dataset))
        test_dataset = dataset[split_point[k]:split_point[k+1]]
        print("Length of testing dataset: ",len(test_dataset))
        start_time = time.time()
        cars = rule_generator(training_dataset, minsup, minconf)
        # print("CARs:(set)")
        # #cars.print_rule()
        # print("len:")
        # print(len(cars.rules))
        # print("Rule_list")
        # #cars.print_rule_list()
        # print("len:")
        # print(len(cars.rule_list))
        # for i in cars.rules:
        #     if i in cars.rule_list:
        #         # print("Hello")
        #         continue
        #     else:
        #         print("World")
        print("Number of rules without pruning: ",len(cars.rules))
        cars.prune_rules(training_dataset)
        cars.rules = cars.pruned_rules
        end_time = time.time()
        cba_rg_runtime = end_time - start_time
        cba_rg_total_runtime += cba_rg_runtime

        start_time = time.time()
        classifier_m1 = classifier_builder_m1(cars, training_dataset)
        end_time = time.time()
        cba_cb_runtime = end_time - start_time
        cba_cb_total_runtime += cba_cb_runtime

        error_rate = get_error_rate(classifier_m1, test_dataset)
        error_total_rate += error_rate

        predictor_accuracy = get_accuracy(classifier_m1,test_dataset)
        total_accuracy += predictor_accuracy

        total_car_number += len(cars.rules)
        total_classifier_rule_num += len(classifier_m1.rule_list)

        print("CBA's error rate with pruning: %.1lf%%" % (error_rate * 100))
        print("No. of CARs with pruning: %d" % len(cars.rules))
        print("CBA-RG's run time with pruning: %.2lf s" % cba_rg_runtime)
        print("CBA-CB M1's run time with pruning: %.2lf s" % cba_cb_runtime)
        print("No. of rules in classifier of CBA-CB M1 with pruning: %d" % len(classifier_m1.rule_list))

    print("\nAverage CBA's error rate with pruning: %.1lf%%" % (error_total_rate / 10 * 100))
    print("Average CBA's accuracy with pruning: %.1lf%%" % (total_accuracy/ 10 * 100))
    print("Average No. of CARs with pruning: %d" % int(total_car_number / 10))
    print("Average CBA-RG's run time with pruning: %.2lf s" % (cba_rg_total_runtime / 10))
    print("Average CBA-CB M1's run time with pruning: %.2lf s" % (cba_cb_total_runtime / 10))
    print("Average No. of rules in classifier of CBA-CB M1 with pruning: %d" % int(total_classifier_rule_num / 10))


# 10-fold cross-validations on CBA (M2) without pruning
def cross_validate_m2_without_prune(dataset_name, minsup=0.01, minconf=0.5):
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
    total_classifier_rule_num = 0
    error_total_rate = 0
    total_accuracy = 0

    for k in range(len(split_point)-1):
        print("\nRound %d:" % k)

        training_dataset = dataset[:split_point[k]] + dataset[split_point[k+1]:]
        test_dataset = dataset[split_point[k]:split_point[k+1]]

        start_time = time.time()
        cars = rule_generator(training_dataset, minsup, minconf)
        end_time = time.time()
        cba_rg_runtime = end_time - start_time
        cba_rg_total_runtime += cba_rg_runtime

        start_time = time.time()
        classifier_m2 = classifier_builder_m2(cars, training_dataset)
        end_time = time.time()
        cba_cb_runtime = end_time - start_time
        cba_cb_total_runtime += cba_cb_runtime

        error_rate = get_error_rate(classifier_m2, test_dataset)
        error_total_rate += error_rate

        predictor_accuracy = get_accuracy(classifier_m2,test_dataset)
        total_accuracy += predictor_accuracy

        total_car_number += len(cars.rules)
        total_classifier_rule_num += len(classifier_m2.rule_list)

        print("CBA's error rate without pruning: %.1lf%%" % (error_rate * 100))
        print("No. of CARs without pruning: %d" % len(cars.rules))
        print("CBA-RG's run time without pruning: %.2lf s" % cba_rg_runtime)
        print("CBA-CB M2's run time without pruning: %.2lf s" % cba_cb_runtime)
        print("No. of rules in classifier of CBA-CB M2 without pruning: %d" % len(classifier_m2.rule_list))

    print("\nAverage CBA's error rate without pruning: %.1lf%%" % (error_total_rate / 10 * 100))
    print("Average CBA's accuracy without pruning: %.1lf%%" % (total_accuracy/ 10 * 100))
    print("Average No. of CARs without pruning: %d" % int(total_car_number / 10))
    print("Average CBA-RG's run time without pruning: %.2lf s" % (cba_rg_total_runtime / 10))
    print("Average CBA-CB M2's run time without pruning: %.2lf s" % (cba_cb_total_runtime / 10))
    print("Average No. of rules in classifier of CBA-CB M2 without pruning: %d" % int(total_classifier_rule_num / 10))


# 10-fold cross-validations on CBA (M2) with pruning
def cross_validate_m2_with_prune(dataset_name, minsup=0.01, minconf=0.5):
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
    total_classifier_rule_num = 0
    error_total_rate = 0
    total_accuracy = 0

    for k in range(len(split_point)-1):
        print("\nRound %d:" % k)

        training_dataset = dataset[:split_point[k]] + dataset[split_point[k+1]:]
        test_dataset = dataset[split_point[k]:split_point[k+1]]

        start_time = time.time()
        cars = rule_generator(training_dataset, minsup, minconf)


        # cars.prune_rules(training_dataset)
        

        # cars.rules = cars.pruned_rules
        cars.rules=topKRules(int(len(cars.rule_list)*0.85),0.5,cars)
        end_time = time.time()
        cba_rg_runtime = end_time - start_time
        cba_rg_total_runtime += cba_rg_runtime

        start_time = time.time()
        classifier_m2 = classifier_builder_m2(cars, training_dataset)
        end_time = time.time()
        cba_cb_runtime = end_time - start_time
        cba_cb_total_runtime += cba_cb_runtime

        error_rate = get_error_rate(classifier_m2, test_dataset)
        error_total_rate += error_rate

        predictor_accuracy = get_accuracy(classifier_m2,test_dataset)
        total_accuracy += predictor_accuracy

        total_car_number += len(cars.rules)
        total_classifier_rule_num += len(classifier_m2.rule_list)

        print("CBA's error rate with pruning: %.1lf%%" % (error_rate * 100))
        print("No. of CARs without pruning: %d" % len(cars.rules))
        print("CBA-RG's run time with pruning: %.2lf s" % cba_rg_runtime)
        print("CBA-CB M2's run time with pruning: %.2lf s" % cba_cb_runtime)
        print("No. of rules in classifier of CBA-CB M2 with pruning: %d" % len(classifier_m2.rule_list))

    print("\nAverage CBA's error rate with pruning: %.1lf%%" % (error_total_rate / 10 * 100))
    print("Average CBA's accuracy with pruning: %.1lf%%" % (total_accuracy/ 10 * 100))
    print("Average No. of CARs with pruning: %d" % int(total_car_number / 10))
    print("Average CBA-RG's run time with pruning: %.2lf s" % (cba_rg_total_runtime / 10))
    print("Average CBA-CB M2's run time with pruning: %.2lf s" % (cba_cb_total_runtime / 10))
    print("Average No. of rules in classifier of CBA-CB M2 with pruning: %d" % int(total_classifier_rule_num / 10))


# test entry goes here
if __name__ == "__main__":
    # using the relative path, all data sets are stored in datasets directory
    # dataset='iris'
    dataset = 'seeds'

    # just choose one mode to experiment by removing one line comment and running
    #cross_validate_m1_without_prune(dataset)
    #cross_validate_m1_with_prune(dataset)
    #cross_validate_m2_without_prune(dataset)
    cross_validate_m2_with_prune(dataset)
