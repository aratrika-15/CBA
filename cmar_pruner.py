def antecedants_matched(datacase, rule):
    """
    Checks if all of a rule's antecedants are present in a datacase
    """
    for item in rule.cond_set:
        if datacase[item] != rule.cond_set[item]:
            return False
    return True


def sort_rules(ruleList):
    """
    Returns a sorted list of rules based on the following criteria:
    1. Larger confidence is better
    2. If confidence is same, larger support is better
    3. If confidence and support are the same, smaller antecedant size is better
    """
    sorted_rules = sorted(ruleList, key = lambda x: (x.confidence, x.support, -len(x.cond_set)), reverse=True)
    return sorted_rules


def prune_with_cover(dataset, ruleList, minCover=3):
    """
    Prunes the ruleList using the smallest set of rules that cover the dataset cases at least minCover times.
    Used as a part of the CMAR implementation.

    ruleList must have already been sorted by precedence of confidence->support->size of antecedant.

    Returns a set of pruned rules.
    """
    coverCount = {i:0 for i in range(len(dataset))}
    # using dictionary instead of the traditional list for efficiency of access and removal, and to prevent modifying dataset
    prunedRules = set()
    for rule in ruleList:
        removeSet = set()  # set of datacases that got covered in this iteration and can be removed
        if len(coverCount) == 0:  # all cases have been covered and removed
            break
        coverFlag = False
        for index in coverCount:
            datacase = dataset[index]
            if antecedants_matched(datacase, rule):
                coverCount[index] += 1
                coverFlag = True
                if coverCount[index] >= minCover and index not in removeSet:
                    removeSet.add(index)
        if coverFlag:
            prunedRules.add(rule)
        for index in removeSet:
            coverCount.pop(index, None)
    
    return prunedRules
