
import random
import numpy as np

import matplotlib.pyplot as plt
from collections import Counter, deque

feature_list = ['f0',  'f1',  'f2',  'f3',  'f4',  'f5',  'f6',  'f7',  'f8',  'f9',  
                'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 
                'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27']


def get_subtrees(root, layer_skip, layer_share):

    subtrees = []
    queue = deque([root])

    while queue:
        node = queue.popleft()

        # if node['depth'] % 3 == 0:
        # if node['depth'] % 8 == 0:
        if node['depth'] % layer_skip == 0:
            subtrees.append(node)

        for child in node['children']:
            if 'leaf' not in child.keys():
                queue.append(child)

    result = []
    for subtree in subtrees:
        # result.append(get_subfeat(subtree, 4))
        result.append(get_subfeat(subtree, layer_share))

    return result


def get_subfeat(root, n):
    result = []
    queue = deque([(root, 0)])
    
    while queue:
        node, depth = queue.popleft()

        if depth < n:
            result.append(node['split'])

            for child in node['children']:
                if 'leaf' not in child.keys():
                    queue.append((child, depth + 1))
    return result


def calc_feat_groups(groups, group_set = []):
    #
    result = [] 
    # for tree in trees:
    #     for group in tree:
    #         result += [x for x in list(set(group))]
    
    for group in groups:
        # result += [len(set(group))]
        # result += [x for x in group]
        result += [x for x in list(set(group))]

    feat_groups_dict = Counter(result)
    # for k in feature_list:
    #     if not feat_groups_dict[k]:
    #         feat_groups_dict[k] = 0

    # plot_groups(feat_groups_dict)

    feat_groups_items = sorted(feat_groups_dict.items(), key=lambda x:x[1], reverse=True)
    labels, values = zip(*feat_groups_items)

    cur_set = labels[:16]
    group_set.append(cur_set)

    new_trees = []
    group_share_num = 0
    # update trees
    for i, group in enumerate(groups):
        if set(group).issubset(set(cur_set)):
            group_share_num += 1
        else:
            new_trees.append(group)

    print(f'group sharing number is {group_share_num}')

    return new_trees



def calc_feat_set(groups, groups_tid, set_grp = [], set_gid = []):

    raw_num = len(groups)
    # 
    # cur_set = set(groups[0])
    cur_set = set(groups[random.randint(0, raw_num-1)])
    cur_gid = []

    while len(cur_set) <= 16:
        groups_sorted = sorted(groups, \
            key=lambda x:(cur_set.intersection(x), -len(x)), reverse=True)
        if len(cur_set.union(set(groups_sorted[0]))) > 16:
            break
        else:
            cur_set = cur_set.union(set(groups_sorted[0]))

        grp_remaining = []
        grp_tid_remaining = []
        for idx, group in enumerate(groups):
            if not set(group).issubset(cur_set):
                grp_remaining.append(group)
                grp_tid_remaining.append(groups_tid[idx])
            else:
                cur_gid.append(groups_tid[idx])
        
        groups = grp_remaining
        groups_tid = grp_tid_remaining

        if not any(groups):
            break

    print(f'group sharing number is {raw_num - len(groups)}')
    set_grp.append(cur_set)
    set_gid.append(cur_gid)

    return groups, groups_tid

def plot_groups(feat_groups_dict):

    plt.figure(figsize=(12,4))
    feat_groups_items = sorted(feat_groups_dict.items(), key=lambda x:x[1], reverse=True)
    labels, values = zip(*feat_groups_items)
    indexes = np.arange(len(labels))
    width = 1
    plt.bar(indexes, values, width, edgecolor="k", lw=0.5)
    plt.xticks(indexes, labels)
    # plt.xlabel('group length', fontsize=10)
    # plt.ylabel('group number', fontsize=10)

    plt.xlabel('feature index', fontsize=10)
    plt.ylabel('feature number', fontsize=10)
    # plt.ylabel('group number', fontsize=10)

    plt.savefig('groups_length')
    # plt.savefig('feat_groups')
    plt.close()


def flat_group(trees, trees_off):

    groups = []
    groups_tid = []
    for idx, tree in enumerate(trees):
        for group in tree:
            # remove duplicated element in group
            groups.append(list(set(group)))
            # groups.append(list(group))
            groups_tid.append(idx + trees_off)
        
    return groups, groups_tid

def make_group_set(trees, trees_off):
    groups, groups_tid = flat_group(trees, trees_off)
    feat_set = []
    gid_set  = []
    while 1:
        groups = calc_feat_set(groups, groups_tid, feat_set, gid_set)
        if not any(groups):
            break
    
    print(len(flat_group(trees))/len(feat_set))

    pass
