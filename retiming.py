import networkx as nx
from sys import maxint


def clock_period(graph):
    """
    Determine the minimum clock period of a given synchronous network
    graph
    """
    G0 = graph.copy()
    for i, j in graph.edges():
        if graph.edge[i][j]['weight'] != 0:
            G0.remove_edge(i,j)
    sorted_nodes = nx.topological_sort(G0)
    delta = [0 for node in sorted_nodes]
    for node in sorted_nodes:
        in_edges = G0.in_edges(node)
        d = graph.node[node]
        if not in_edges:
            delta[node] = d
        else:
            delta_prevs = [delta[u] for u, v in in_edges
                           if graph.edge[u][v]['weight'] == 0]
            if not delta_prevs:
                delta_prevs = [0]
            delta[node] = d + max(delta_prevs)
    return delta


def floyd_warshall(graph, weight='weight', min=0, max=maxint,
                   less=(lambda x, y: x < y),
                   add=(lambda x, y: x + y)):
    """
    Floyd-Warshall Algorithm
    Ref: https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
    """
    dist = [[max for node in graph] for node in graph]
    for node in graph:
        dist[node][node] = min
    for u, v in graph.edges():
        dist[u][v] = graph.edge[u][v][weight]
    for k in graph:
        for i in graph:
            for j in graph:
                new_dist = add(dist[i][k], dist[k][j])
                if new_dist < dist[i][j]:
                    dist[i][j] = new_dist
    return dist


def add_tuple(tup1, tup2):
    return tuple(a + b for a, b in zip(tup1, tup2))


def weight_matrices(graph):
    """
    Compute W & D matrices
    """
    for u, v in graph.edges():
        w = graph[u][v]['weight']
        d = graph.node[u]
        graph[u][v]['wd'] = (w, -d)

    wd = floyd_warshall(graph, min=(0, 0), max=(maxint, 0),
                        weight='wd', add=add_tuple)

    W = [[0 for n in graph] for n in graph]
    D = [[0 for n in graph] for n in graph]
    for u, row in enumerate(wd):
        for v, (w, d) in enumerate(row):
            W[u][v] = w
            D[u][v] = graph.node[v] - d

    return W, D


# def topological_sort(graph, root=0):
#     sorted_nodes = list()
#     visited = set()
#     queue = [root]
#     while queue:
#         node = queue.pop(0)
#         if node not in visited:
#             sorted_nodes.append(node)
#             visited.add(node)
#             children = graph.edge[node]
#             if children:
#                 queue += sorted(children)
#     return sorted_nodes


def retime(graph, r):
    graph = graph.copy()
    for node in graph:
        prevs = graph.in_edges(node)
        nexts = graph.edge[node].keys()
        for prev, _ in prevs:
            graph.edge[prev][node]['weight'] += r[node]
        for next in nexts:
            graph.edge[node][next]['weight'] -= r[node]
    return graph


def feas(graph, c):
    r = [0 for node in graph]
    for i in range(len(r) - 1):
        G_r  = retime(graph, r)
        delta = clock_period(G_r)
        for node in graph:
            if delta[node] > c:
                r[node] += 1
    G_r = retime(graph, r)
    phi = max(clock_period(G_r))
    if phi > c:
        return None
    else:
        return r


def opt2(graph):
    W, D = weight_matrices(graph)
    unrolled = [val for row in D for val in row]
    D_sorted = sorted(unrolled) # TODO: remove duplicates
    retimings = []
    for d in D_sorted:
        r = feas(graph, d)
        if r:
            retimings.append(r)
    return retimings

# TODO: init phi <- D matrix
# TODO: verbose mode
# TODO: calculate initial and final area
