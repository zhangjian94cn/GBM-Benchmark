from collections import deque

def get_subtree(root, n):

    subtrees = []
    queue = deque([root])

    while queue:
        node = queue.popleft()

        if node['depth'] % n == 0:
            subtrees.append(node)

        for child in node['children']:
            if 'leaf' not in child.keys():
                queue.append(child)

    result = []
    for subtree in subtrees:
        result.append(get_subfeat(subtree, 4))

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
