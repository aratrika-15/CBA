from cmar_rule import *

def calculate_wcs(ruleList):
    """
    Calculates weighted chi square of list of rules that have the same consequent.
    WCS can be used as a fair measure of which label to assign to a test case.
    """
    wcs = 0
    for rule in ruleList:
        wcs += rule.chi ** 2 / rule.mcs
    return wcs

def chi_sq_classifier(case, ruleList):
    """
    Classifies a test case "case" using the full rule list "ruleList".
    Uses the CMAR method of checking weighted chi square to use multiple rules for classification.
    """
    groups = {}  # measures groups of class_label:satisfied_rules
    for rule in ruleList:
        if rule.antecedants_matched(case):
            if rule.class_label in groups:
                groups[rule.class_label].append(rule)
            else:
                groups[rule.class_label] = [rule]
    if len(groups) == 0:  # no rule matched, return default class label which is class_label of rule with highest priority
        return ruleList[0].class_label
    if len(groups) == 1:  # only one consequent possible from matching rules, return directly
        return [i for i in groups.keys()][0]
    
    # calculate wcs for each group and return class label with maximum value
    wcsList = {k:calculate_wcs(groups[k]) for k in groups}
    return max(wcsList, key=wcsList.get)