import json	
import numpy as np
import xgboost as xgb

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


# plot
def plot_tree(model, num_trees=0):
    fig, ax = plt.subplots(figsize=(300, 300))
    xgb.plot_tree(model, num_trees=num_trees, ax=ax)
    plt.savefig(f"tree-{num_trees}.pdf")

def plot_hist(tree_depth, max_depth, hist_path):

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
    N, bins, patches = axs.hist(np.array(tree_depth, dtype=int), bins = max_depth)
    
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
    plt.savefig(hist_path)

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

def get_tree_depth_list(model):
    booster = model.get_booster()
    return [get_tree_depth(x) for x in booster.get_dump(dump_format="json")]

# interface
def visualize(args, model):

    tree_depth_list = get_tree_depth_list(model)
    plot_hist(tree_depth_list, args.max_depth, args.hist_path)
