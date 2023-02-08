import json

class Node:
    def __init__(self, feat_val, feat_idx, left=None, right=None):
        self.feat_val = feat_val
        self.feat_idx = feat_idx
        self.left = left
        self.right = right

class TreeAgg:
    def __init__(self, roots, reg, idx):
        self.roots = roots
        self.reg = reg
        self.idx = idx

def binary_tree_to_dict(root):
    if root is None:
        return None
    return {'feat_val': root.feat_val,
            'feat_idx': root.feat_idx,
            'left': binary_tree_to_dict(root.left),
            'right': binary_tree_to_dict(root.right)}

def tree_agg_to_dict(tree_agg):
    
    tree_agg_dict = {
        'reg': tree_agg.reg,
        'idx': tree_agg.idx,
        'trees': []}

    for root in tree_agg.roots:
        tree_agg_dict['trees'].append(binary_tree_to_dict(root))
    
    return tree_agg_dict


def save_tree_agg(tree_dict, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(tree_dict))



