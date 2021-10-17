class CMAR_Rule:
    def __init__(self, cond_set, label, dataset):
        self.cond_set = cond_set
        self.class_label = label
        self.sup_a, self.sup_c, self.support = self.calculate_supports(dataset)
        if self.sup_a != 0:
            self.confidence = self.support/self.sup_a
        else:
            self.confidence = 0

        # required for chi square statistic
        self.n = len(dataset)  # number of values trained on
        self.mcs = self.calculate_mcs()
        self.chi = self.calculate_chi_squared()
    
    def antecedants_matched(self, datacase):
        """
        Checks if all of a rule's antecedants are present in a datacase
        """
        for item in self.cond_set:
            if datacase[item] != self.cond_set[item]:
                return False
        return True
    
    def calculate_supports(self, dataset):
        antecedant_count = 0
        consequent_count = 0
        support_count = 0
        for case in dataset:
            flag = 0
            if self.antecedants_matched(case):
                antecedant_count += 1
                flag += 1
            if case[-1] == self.class_label:
                consequent_count += 1
                flag += 1
            if flag == 2:
                support_count += 1
        return antecedant_count, consequent_count, support_count
    
    def calculate_mcs(self):
        """
        Calculates maximum chi squared (MCS) value for the rule
        """
        if self.support == 0:
            return 0

        e = 1/(self.sup_a*self.sup_c) + 1/(self.sup_a*(self.n-self.sup_c)) + 1/((self.n-self.sup_a)*self.sup_c) + 1/((self.n-self.sup_a)*(self.n-self.sup_c))
        mcs = (min(self.sup_a, self.sup_c) - self.sup_a * self.sup_c / self.n)**2 * self.n * e
        return mcs
    
    def calculate_obs_and_exp(self):
        """
        Calculates observed and expected values required for Chi Square values
        """

        # 2x2 matrix of (antecedants satisfied, not satisfied)x(consequent satisfied, not satisfied) flattened to 1D array of length 4
        obsVals = [self.support, self.sup_a - self.support, self.sup_c - self.support, self.n - self.sup_a - self.sup_c + self.support]

        # calculating expected counts assuming that each satisfied/not satified chance is 50/50 i.e. no correlation
        sup_not_a = self.n - self.sup_a
        sup_not_c = self.n - self.sup_c
        expVals = [self.sup_a * self.sup_c / self.n, self.sup_a * sup_not_c / self.n, sup_not_a * self.sup_c / self.n, sup_not_a * sup_not_c / self.n]
        
        return obsVals, expVals
    
    def calculate_chi_squared(self):
        """
        Calculates the Chi Squared statistic of the rule based on observed and expected values.
        """
        chi = 0
        obsVals, expVals = self.calculate_obs_and_exp()
        for i in range(4):
            if expVals[i] != 0:
                chi += (obsVals[i] - expVals[i])**2 / expVals[i]
        return chi


    # DELETE/REWRITE BELOW FUNCTIONS LATER
    # print out the ruleitem
    def print(self):
        cond_set_output = ''
        for item in self.cond_set:
            cond_set_output += '(' + str(item) + ', ' + str(self.cond_set[item]) + '), '
        cond_set_output = cond_set_output[:-2]
        print('<({' + cond_set_output + '}, ' + str(self.cond_sup_count) + '), (' +
              '(class, ' + str(self.class_label) + '), ' + str(self.rule_sup_count) + ')>')

    # print out rule
    def print_rule(self):
        cond_set_output = ''
        for item in self.cond_set:
            cond_set_output += '(' + str(item) + ', ' + str(self.cond_set[item]) + '), '
        cond_set_output = '{' + cond_set_output[:-2] + '}'
        print(cond_set_output + ' -> (class, ' + str(self.class_label) + ')')