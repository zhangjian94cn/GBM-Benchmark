import json	
import numpy as np
import xgboost as xgb

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from collections import Counter

# plot
def plot_tree(model, num_trees=0):
    fig, ax = plt.subplots(figsize=(300, 300))
    xgb.plot_tree(model, num_trees=num_trees, ax=ax)
    plt.savefig(f"tree-{num_trees}.pdf")

def plot_hist(tree_depth, save_path):

    legend = ['distribution']
 
    # Creating histogram
    fig, axs = plt.subplots(1, 1,
                            figsize =(10, 7),
                            tight_layout = True)
    
 
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        axs.spines[s].set_visible(False)
    
    # Remove x, y ticks
    axs.xaxis.set_ticks_position('none')
    axs.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    axs.xaxis.set_tick_params(pad = 5)
    axs.yaxis.set_tick_params(pad = 10)
    
    # Add x, y gridlines
    axs.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.6)
 
    # Creating histogram
    tree_depth_array = np.array(tree_depth, dtype=int)
    N, bins, patches = axs.hist(tree_depth_array, bins = tree_depth_array.max())
    
    # Setting color
    fracs = ((N**(1 / 5)) / N.max())
    norm = colors.Normalize(fracs.min(), fracs.max())
    
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    
    # Adding extra features   
    plt.xlabel("tree depth")
    plt.ylabel("tree number")
    # plt.legend(legend)
    plt.title('GBT Distribution')
    # plt.xlim((0, max_depth+1))

    # Show plot
    plt.savefig(save_path)

# get tree depth
def item_generator(json_input, lookup_key):
    if isinstance(json_input, dict):
        for k, v in json_input.items():
            if k == lookup_key:
                yield v
            else:
                yield from item_generator(v, lookup_key)
    elif isinstance(json_input, list):
        for item in json_input:
            yield from item_generator(item, lookup_key)

def get_tree_depth(json_text):
    json_input = json.loads(json_text)
    return max(list(item_generator(json_input, 'depth'))) + 1

def get_tree_split_indices(json_text):
    json_input = json.loads(json_text)
    return list(item_generator(json_input, 'split'))


def get_tree_depth_list(model):
    booster = model.get_booster()
    return [get_tree_depth(x) for x in booster.get_dump(dump_format="json")]

def get_tree_feature(model):
    return [get_tree_split_indices(x) for x in model.get_dump(dump_format="json")]


# interface
def visualize(model, save_path):
    tree_depth_list = get_tree_depth_list(model)
    plot_hist(tree_depth_list, save_path)


# interface
def distribution(model):

    feature_list = ['f0',  'f1',  'f2',  'f3',  'f4',  'f5',  'f6',  'f7',  'f8',  'f9',  
                    'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 
                    'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27']

    tree_feature_list = get_tree_feature(model)
    tree_feature_full_list = sum(tree_feature_list, [])
    # plot_hist(tree_feature_list, save_path)

    # n, bins, patches=plt.hist(tree_feature_full_list)
    # plt.xlabel('feature',fontsize=8)
    # plt.savefig('full')
    # plt.close()

    single_tree = tree_feature_list[0]
    single_tree_dict = Counter(single_tree)
    for k in feature_list:
        if not single_tree_dict[k]:
            single_tree_dict[k] = 0
    
    plt.figure(figsize=(12,4))   
    single_tree_items = sorted(single_tree_dict.items(), key=lambda x:x[1], reverse=True)
    labels, values = zip(*single_tree_items)
    indexes = np.arange(len(labels))
    width = 1
    plt.bar(indexes, values, width, edgecolor="k", lw=0.5)
    plt.xticks(indexes, labels)
    plt.xlabel('attribute index', fontsize=10)
    plt.ylabel('attribute occurrence', fontsize=10)
    plt.savefig('single')
    plt.close()
    
    n, bins, patches=plt.hist(tree_feature_list[0])
    
    plt.xlabel('feature',fontsize=10)
    plt.savefig('single')
    plt.close()

    # plt.xlabel("Values")
    # plt.ylabel("Frequency")
    # plt.title("Histogram")


if __name__ == '__main__':

    model_path = ''
    save_path = ''
    
    visualize(model, save_path)



