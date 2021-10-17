import fpgrowth
from collections import defaultdict

def backtrack_fp_tree(cur_node,path):
    parent=cur_node.parent
    if parent==None:
        return
    else:
        path.append(cur_node.item)
        backtrack_fp_tree(parent,path)


def find_all_paths(header_table, condition_set):
    head=header_table[condition_set][1]
    condition_frequency=list()
    condition_patterns=list()
    while head!=None:
        path=[]
        backtrack_fp_tree(head,path)
        if len(path)>1:
            condition_frequency.append(head.frequency)
            condition_patterns.append(path[1:])
        head=head.next

    return condition_frequency, condition_patterns

def mine_fp_tree(header_table, minsup, prefix, frequent_items,dataset):
    sorted_condition_list = [x[0] for x in sorted(list(header_table.items()), key=lambda p:p[1][0])] 
    for i in sorted_condition_list:  
        frequent_patterns = prefix.copy()
        frequent_patterns.add(i)
        frequent_items.append(frequent_patterns)
        condition_frequency, condition_patterns = find_all_paths(header_table,i) 
        cur_header = defaultdict(int)
        for index, pattern in enumerate(condition_patterns):
            for i in pattern:
                cur_header[i] += condition_frequency[index]

        header=dict()
        for key, value in cur_header.items():
            if value>=minsup:
                header[key]=value
        new_header, conditional_fp_tree = fpgrowth.create_fp_tree(dataset,header) 
        if conditional_fp_tree!=None:
            print("Tree:")
            print()
            conditional_fp_tree.print_fp_tree()
        if new_header ==None:
            continue
        else:
            mine_fp_tree(new_header, minsup, frequent_patterns, frequent_items,dataset)

