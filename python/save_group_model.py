import json
from collections import deque

class TreeAgg:
    def __init__(self, roots, reg, idx):
        self.roots = roots
        self.reg = [[int(xi[1:]) for x in reg for xi in x]]
        self.idx = idx

def binary_tree_to_dict(root, reg):
    
    tree = {'weight': [], 'index': []}
    queue = deque([root])

    while queue:
        node = queue.popleft()
        if 'leaf' not in node.keys():
            tree['weight'].append(node['split_condition'])
            
            if node['depth'] < 3:
                tree['index'].append(reg[0].index(int(node['split'][1:])))
            else:
                tree['index'].append(int(node['split'][1:]))

            for child in node['children']:
                queue.append(child)

        else:
            tree['weight'].append(node['leaf'])
            tree['index'].append(0)
        
    return tree


def tree_agg_to_dict(tree_agg):
    
    tree_agg_dict = {
        'reg': list(tree_agg.reg),
        'idx': tree_agg.idx,
        'trees': []}

    for root in tree_agg.roots:
        tree_agg_dict['trees'].append(binary_tree_to_dict(root, list(tree_agg.reg)))
    
    return tree_agg_dict

def save_tree_agg(tree_dict, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(tree_dict))

def convert_tree_agg(groups_root, feat_set, gid_set):

    tree_agg = TreeAgg(groups_root, feat_set, gid_set)
    tree_agg_dict = tree_agg_to_dict(tree_agg)
    save_tree_agg(tree_agg_dict, 'test.json')

