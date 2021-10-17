import rule_data_satisfaction
import cba_rg
from functools import cmp_to_key


class M1_Classifier:
    def __init__(self):
        self.rules_list=list()
        self.errors_list=list()
        self.rule_error_list=list()
        self.default_class_list=list()
        self.default_class=None

    def print_details(self):
        for rule in self.rules_list:
            rule.print_rule()

        print("The default class is: ",self.default_class)

    def insert_rule(self, rule):
        self.rules_list.append(rule)

    def select_default_class(self,dataset):
        if len(dataset)<=0:
            self.default_class_list.append(self.default_class_list[-1])
            return 
        class_labels=[data[-1] for data in dataset]
        self.default_class_list.append(max(class_labels, key=class_labels.count))

    def compute_total_errors(self,dataset,number_of_rule_errors):
        self.rule_error_list.append(number_of_rule_errors)
        rule_errors=sum(self.rule_error_list)

        default_class_errors=0
        class_labels=[data[-1] for data in dataset]
        for label in class_labels:
            if label!=self.default_class_list[-1]:
                default_class_errors+=1

        total_errors=rule_errors+default_class_errors
        self.errors_list.append(total_errors)

    def finalize_classifier(self):
        index=self.errors_list.index(min(self.errors_list))
        self.default_class=self.default_class_list[index]
        self.rules_list=self.rules_list[:(index+1)]
        self.errors_list=[]
        self.default_class_list=[]
        self.rule_error_list=[]


def update_dataset(dataset,temp):
    new_dataset=[]
    for data in dataset:
        if data not in temp:
            new_dataset.append(data)
    return new_dataset

def sort_rules(car_list):
    def check_precedence(a, b):
        if a.confidence < b.confidence:
            return 1
        elif a.confidence == b.confidence:
            if a.support < b.support:  
                return 1
            elif a.support == b.support:
                if len(a.cond_set) < len(b.cond_set):  
                    return -1
                elif len(a.cond_set) == len(b.cond_set):
                    
                    for rule in car_list.rule_list:
                        if rule==a:
                            
                            return -1
                        if rule==b:
                            
                            return 1

                    return 0

                else:
                    return 1
            else:
                return -1
        else:
            return -1
    
    rule_list = list(car_list.rules)
    rule_list.sort(key=cmp_to_key(check_precedence))
    return rule_list



def create_classification_model(dataset,car_list):
    car_list=sort_rules(car_list)
    m1_classifier=M1_Classifier()
    for rule in car_list:
        temp=[]
        marked=False
        temp_satisfies_consequent=0
        for data in dataset:
            satisfied=rule_data_satisfaction.check_rule_data_satisfaction(rule,data)
            if satisfied==None:
                continue
            temp.append(data)
            if satisfied==True:
                marked=True
                temp_satisfies_consequent+=1

        if marked==True:
            number_of_rule_errors=len(temp)-temp_satisfies_consequent
            m1_classifier.insert_rule(rule)
            updated_dataset=update_dataset(dataset,temp)
            dataset=updated_dataset
            m1_classifier.select_default_class(updated_dataset)
            m1_classifier.compute_total_errors(updated_dataset,number_of_rule_errors)

    m1_classifier.finalize_classifier()
    return m1_classifier


# just for test
if __name__ == '__main__':
    #dataset given should be in the form of a 2D list, where every list element is one sample input
    dataset = [[1, 1, 1], [1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 2, 1],
               [2, 2, 0], [2, 3, 0], [2, 3, 0], [1, 1, 0], [3, 2, 0]]
    minsup = 0.15
    minconf = 0.6
    #cars is a Car Object, which has a set of rules, and a set of pruned_rules. Items in this set are RuleItem objects
    car_list = cba_rg.rule_generator(dataset, minsup, minconf)
    m1_classifier = create_classification_model(dataset,car_list)
    m1_classifier.print_details()

    dataset = [[1, 1, 1], [1, 1, 1], [1, 2, 1], [2, 2, 1], [2, 2, 1],
               [2, 2, 0], [2, 3, 0], [2, 3, 0], [1, 1, 0], [3, 2, 0]]
    car_list.prune_rules(dataset)
    car_list.rules = car_list.pruned_rules
    m1_classifier = create_classification_model(dataset,car_list)
    m1_classifier.print_details()



    




