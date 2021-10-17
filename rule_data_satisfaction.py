def check_rule_data_satisfaction(rule,data):
    for attribute in rule.cond_set:
        if rule.cond_set[attribute]!=data[attribute]:
            return None
    if rule.class_label!=data[-1]:
        return False
    else:
        return True