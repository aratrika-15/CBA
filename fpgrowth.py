from collections import Counter
import pandas as pd
import mine_fp_tree

class Tree_Node:
    def __init__(self, frequency, item, parent):
        self.item=item
        self.parent=parent
        self.frequency=frequency
        self.next=None
        self.children={}

    def increase_frequency(self,frequency):
        self.frequency+=frequency

    def print_fp_tree(self, ind=1):
        print('  ' * ind, self.item, ' ', self.frequency)
        for child in list(self.children.values()):
            child.print_fp_tree(ind+1)


def find_one_frequent_items(dataset,support):
    #find list of 1-frequent items stored in support descending order
    one_frequent_items={}
    for (columnName, columnData) in dataset.iloc[:, :-1].iteritems():
        column_counts=Counter(columnData)
        for col in column_counts.most_common():
            if col[1]>=support:
                one_frequent_items[(columnName,col[0])]=col[1]
    sorted_one_frequent_items = sorted(one_frequent_items.items(), key=lambda x: x[1], reverse=True)
    return sorted_one_frequent_items


def add_to_tree(condition, current_node, header_table,freq):
    if condition in current_node.children:
        current_node.children[condition].increase_frequency(freq)
    else:
        new_node=Tree_Node(1, condition, current_node)
        current_node.children[condition]=new_node

        #update header table
        latest_node=header_table[condition][1]
        if latest_node!=None:
            while latest_node.next!=None:
                latest_node=latest_node.next
            latest_node.next=new_node
        else:
            header_table[condition][1]=new_node


    return current_node.children[condition]
            
def create_fp_tree(dataset,one_frequent_items):
    header_table=one_frequent_items
    if len(header_table)==0:
        return None, None
    for key in header_table.keys():
        header_table[key]=[header_table[key],None]

    fp_tree = Tree_Node(1,'Null', None)
    dataset_list=dataset.values.tolist()
    for idx,data in enumerate(dataset_list):
        condition_set=[]
        for idx,col in enumerate(data[:-1]):
            if (idx,col) in header_table.keys():
                condition_set.append((idx,col))

        condition_set.sort(key=lambda item: header_table[item][0], reverse=True)
        current_node=fp_tree
        for condition in condition_set:
            current_node=add_to_tree(condition, current_node, header_table,1)

    return header_table, fp_tree
        

def fp_growth(dataset,minsup, minconf):
    #first find list of 1-frequent items stored in support descending order
    one_frequent_items=find_one_frequent_items(dataset,minsup)
    one_frequent_items=dict(one_frequent_items)
    #creating the fp_tree
    header_table,fp_tree=create_fp_tree(dataset,one_frequent_items)
    if fp_tree!=None:
        fp_tree.print_fp_tree()
    else:
        print("Tree is empty")
    empty_set=set()
    frequent_items=[]
    mine_fp_tree.mine_fp_tree(header_table, minsup, empty_set,frequent_items,dataset)
    

    



if __name__ == '__main__':
    dataset = [['a1','b1','c1','d1','A'],
                ['a1','b2','c1','d2','B'],
                ['a2','b3','c2','d3','A'],
                ['a1','b2','c3','d3','C'],
                ['a1','b2','c1','d3','C'],
                ]
    dataset= pd.DataFrame(dataset)
    #print(dataset)
    minsup = 2
    minconf = 0.5
    fp_growth(dataset,minsup, minconf)
