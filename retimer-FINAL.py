# Benjamin Zhu bjz2107
# Sean Liu     sl3497
# CSEE 6861
# Final Project: CAD Retiming Tool

import copy
import networkx as nx
import numpy as np
import sys



def read_input_file(filename):
    """
    read input file and generate directed G
    """
    G=nx.DiGraph()
    f = open(filename)
    v_size = f.readline()
    while '.n ' not in v_size:
        v_size = f.readline()
    v_size = int(v_size.split()[1])
    v_data = f.readline().split()
    G.add_node(0)
    G.node[0] = 0
    for i in range(1, v_size+1):
        G.add_node(i)
        G.node[i] = int(v_data[i])
    f.readline()
    line = f.readline()
    area = 0
    while line and (not '.e' in line):
        line = line.split()
        w = int(line[2])
        area += w
        G.add_edge(int(line[0]), int(line[1]), weight=w)
        line = f.readline()
    f.close()
    return G, area


def generate_weight_matrices(G, mode):
    """
    simultaneously generate W and D from G, using floyd-warshall
    """
    for u, v in G.edges():
        G[u][v]['wd'] = (G[u][v]['weight'], -G.node[u])
    fw = floyd_warshall(G)
    W = []
    D = []
    for node in G:
        temp1 = []
        temp2 = []
        for node in G:
            temp1.append(0)
            temp2.append(0)
        W.append(temp1)
        D.append(temp2)
    for u in range(len(fw)):
        for v in range(len(fw[u])):
            (w, d) = fw[u][v]
            W[u][v] = w
            D[u][v] = G.node[v] - d
            if mode == 'v':
                print 'W'
                for row in W:
                    print row
                print
                print 'D'
                for row in D:
                    print row
                print
    return W, D


def floyd_warshall(G):
    """
    Floyd-Warshall Algorithm
    https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
    """
    d = []
    for node in G:
        temp = []
        for node in G:
            temp.append((float('inf'), 0))
        d.append(temp)
    for node in G:
        d[node][node] = (0, 0)
    for u, v in G.edges():
        d[u][v] = G.edge[u][v]['wd']
    for k in G:
        for i in G:
            for j in G:
                t = (d[i][k][0]+d[k][j][0], d[i][k][1]+d[k][j][1])
                if t < d[i][j]:
                    d[i][j] = t
    return d


def clock_period_init(W, D):
    """
    finds clock period using W and D
    """
    delay = 0
    for u in range(len(W)):
        for v in range(len(W)):
            if u == v:
                continue
            if u == 0 and v != len(W)-1:
                continue
            if W[u][v] != 0:
                continue
            if D[u][v] > delay:
                delay = D[u][v]
    return delay


def clock_period(G):
    """
    calculate the clock periods of G
    """
    G0 = G.copy()
    for i, j in G0.edges():
        if G0.edge[i][j]['weight'] != 0:
            G0.remove_edge(i,j)
    nodes_sorted = nx.topological_sort(G0)
    cps = []
    for node in nodes_sorted:
        cps.append(0)
    for node in nodes_sorted:
        cps[node] = G.node[node]
        in_edges = G0.in_edges(node)
        if in_edges:
            cps_prevs = []
            for u, v in in_edges:
                if G.edge[u][v]['weight'] == 0:
                    cps_prevs.append(cps[u])
            if cps_prevs:
                cps[node] += max(cps_prevs)
    return cps


def retime_graph(G, r):
    """
    used in FEAS, for retiming a graph given r vector
    """
    G = G.copy() # don't alter original G
    for node in G:
        prevs = G.in_edges(node)
        nexts = G.edge[node].keys()
        for prev, p in prevs:
            G.edge[prev][node]['weight'] += r[node]
        for next in nexts:
            G.edge[node][next]['weight'] -= r[node]
    return G


def feas(G, c, mode, fp=None):
    """
    FEAS from the text
    """
    r = []
    for node in G:
        r.append(0)
    for i in range(1, len(r)):
        if mode == 'v':
            fp.write('iteration: ' + str(i) + '\n')
        G_r  = retime_graph(G, r)
        cps = clock_period(G_r)
        if mode == 'v':
            fp.write('graph vertices: ' + str(G_r.node) + '\n')
            fp.write('graph edges: ' + str(G_r.edge) + '\n')
            fp.write('delta(v): ' + str(cps) + '\n\n')
        for node in G:
            if cps[node] > c:
                r[node] += 1
    G_r = retime_graph(G, r)
    cps = clock_period(G_r)
    if mode == 'v':
        fp.write('final graph:\n')
        fp.write('graph vertices: ' + str(G_r.node) + '\n')
        fp.write('graph edges: ' + str(G_r.edge) + '\n')
        fp.write('delta(v): ' + str(cps) + '\n')
    cp = max(cps)
    if cp > c:
        return None
    return r


def opt2(G, mode, prefix):
    """
    opt2 using FEAS, with binary search
    """
    W, D = generate_weight_matrices(G, 0)           # 4
    cpi = clock_period_init(W, D)                   # 5
    D_flattened = flatten(D)
    D_unique = remove_duplicates(D_flattened)
    D_sorted = sorted(D_unique)                     # 7
    print 'W'
    for row in W:
        print row
    print
    print 'D'
    for row in D:
        print row
    print
    print 'phi_init:', cpi
    print 'D, sorted and flattened:', D_sorted     
    d_best = D_sorted[-1]
    if not d_best:
        return []
    r_best = feas(G, d_best, 0)
    begin = 0
    end = len(D_sorted) - 1
    f5 = None
    if mode == 'v':
        file5 = prefix + '-part1-FEAS.txt'
        f5 = open(file5, 'w')
    while begin <= end:
        mid = (begin+end)/2
        d = D_sorted[mid]
        r = feas(G, d, mode, f5)
        if r and d < d_best:
            r_best = r;
            end = mid - 1
        else:
            begin = mid + 1
    if mode == 'v':
        f5.close()
    return r_best


def flatten(x):
    """
    flatten a 2-d list
    """
    y = []
    for row in x:
        for value in row:
            y.append(value)
    return y


def remove_duplicates(x):
    """
    remove duplicates from a list
    """
    y = {}
    for value in x:
        y[value] = 0
    return y.keys()


#
# PART 2
#

def generate_c(G):
    c = []
    for node in G:
        in_edges = G.in_edges(node)
        out_edges = G.out_edges(node)
        c.append(len(in_edges) - len(out_edges))
    return c


def generate_tableau(G, W, D, phi, prefix):
    c = generate_c(G)
    c_neg = copy.deepcopy(c)
    for i in range(len(c_neg)):
        c_neg[i] = -c_neg[i]
    tableau_data = []
    num_edges = 0
    file4 = prefix + '-part2-complete-constraint.txt'
    file5 = prefix + '-part2-reduced-constraint.txt'
    fp4 = open(file4, 'w')
    fp5 = open(file5, 'w')
    count1 = 0;
    count2 = 0;
    for u, v in G.edges():
        row = [u, v, G.edge[u][v]['weight']]
        tableau_data.append(row)
        num_edges += 1
        row_string = 'v'+str(u)+' - v'+str(v)+' <= '+str(row[2])
        fp4.write(row_string + '\n')
        fp5.write(row_string + '\n')
        count1 += 1
    count2 = count1
    for u in range(len(W)):
        for v in range(len(W)):
            d = D[u][v]
            row = [u, v, W[u][v]-1]
            if d <= phi:
                continue
            row_string = 'v'+str(u)+' - v'+str(v)+' <= '+str(row[2])
            count1 += 1
            fp4.write(row_string + '\n')
            if d-G.node[u]>phi or d-G.node[v]>phi:
                continue
            row_string = 'v'+str(u)+' - v'+str(v)+' <= '+str(row[2])
            fp5.write(row_string + '\n')
            count2 += 1
            tableau_data.append(row)
    row_string = 'p'
    count = 0
    for term in c_neg:
        if term != 0:
            if term == 1:
                row_string += ' + v'+str(count)
            elif term == -1:
                row_string += ' - v'+str(count)
            else: 
                row_string += ' + ' +str(term)+'v'+str(count)
        count += 1
    row_string += ' = 0'
    fp4.write(row_string + '\n')
    fp5.write(row_string + '\n')
    fp4.close()
    fp5.close()
    row_size = len(tableau_data) + len(G.node)
    tableau = []
    for r in tableau_data:
        row = []
        for i in range(row_size):
            row.append(0)
        row.append(0)
        row.append(r[2])
        tableau.append(row)
    i = len(G.node)
    for j in range(len(tableau)):
        row = tableau[j]
        row[i] = 1
        i += 1
        row[tableau_data[j][0]] = 1
        row[tableau_data[j][1]] = -1
    row = []
    for i in range(row_size):
        row.append(0)
    row.append(1)
    row.append(0)
    for i in range(len(c_neg)):
        row[i] = c_neg[i]
    tableau.append(row)
    basic_vars = []
    for i in range(len(G.node), len(tableau[0])-1):
        basic_vars.append(i)
    return np.array(tableau), basic_vars


def pivot(M, ip, jp):
    """
    performs pivot on a matrix
    from http://projects.scipy.org/scipy/attachment/ticket/1252/lp3.py
    """
    # M is MODIFIED
    n, m = M.shape
    piv = M[ip,jp]
    if piv == 0:
        return False
    else:
        M[ip,:] /= piv
        for i in xrange(n):
            if i != ip:
                M[i,:] -= M[i,jp]*M[ip,:]
    return True


def basic_solve(M, basic_vars):
    """
    calculates standard maximization solution from M and basic varibales
    """
    solution = []
    i = 0;
    for i in range(len(basic_vars)):
        bv = basic_vars[i]
        s = M[i,:][-1] / M[i,:][bv]
        solution.append(s)
    if min(solution) < 0:
        return False
    if solution[-1] == 0:
        return False
    return solution


def simplex_solve(G, M, basic_vars, mode, prefix):
    fp = None
    if mode == 'v':
        file6 = prefix + '-part2-simplex.txt'
        fp = open(file6, 'w')
        fp.write('iteration: 0\n')
        fp.write(str(M))
        fp.write('\n\n')
    test = basic_solve(M, basic_vars)
    if test:
        return test
    iteration = 1
    while True:
        last_row = M[-1,:(-1)]
        min_val = sys.maxint
        col_pivot = None
        for i in range(len(last_row)):
            val = last_row[i]
            if val < 0 and val < min_val:
                min_val = val
                col_pivot = i
        if col_pivot == None:
            break
        if mode == 'v':
            fp.write('iteration: ' + str(iteration) + '\n')
        column = M[:, col_pivot].T
        ans = M[:, -1].T
        ratios = []
        for i in range(len(column)):
            if column[i] <= 0:
                ratios.append(sys.maxint)
            else:
                ratios.append(float(ans[i])/float(column[i]))
        min_val = sys.maxint
        row_pivot = None
        for i in range(len(ratios)):
            if ratios[i] < min_val:
                min_val = ratios[i]
                row_pivot = i
        pivot(M, row_pivot, col_pivot)
        if mode == 'v':
            fp.write('pivot: (' + str(row_pivot) + ', ' + str(col_pivot) + ')')
            fp.write(' ' + str(M[row_pivot, col_pivot]) + '\n')
            fp.write(str(M) + '\n\n')
        basic_vars[row_pivot] = col_pivot
        iteration += 1
    if mode == 'v':
        fp.close()
    solution = basic_solve(M, basic_vars)
    if solution:
        r = []
        for i in range(len(G.node)):
            if i in basic_vars:
                r.append(solution[basic_vars.index(i)])
            else:
                r.append(0)
        return r
    return False


def count_registers(G, r, initial):
    """
    counts number of registers
    """
    c = generate_c(G)
    registers = initial
    for i in range(len(c)):
        registers += r[i]*c[i]
    return registers


def main():
    """
    main function
    """
    filename = sys.argv[1]
    end = filename.find('-in')
    prefix = filename[0:end]
    cycle_time = int(sys.argv[2])
    mode = sys.argv[3]
    
    # part 1
    G, init_area = read_input_file(filename)        # 0
    if mode == 'v':
        file4 = prefix + '-part1-Floyd-Warshall.txt'
        sys.stdout = open(file4, 'w')
    W, D = generate_weight_matrices(G, mode)        # 6
    if mode == 'v':
        sys.stdout.close()
    file3 = prefix + '-part1-WD.txt'
    sys.stdout = open(file3, 'w')
    r = opt2(G, mode, prefix)                       # 1
    sys.stdout.close()
    G_r = retime_graph(G, r)                        # 2
    final_area = count_registers(G, r, init_area)   # 3
    cp = max(clock_period(G_r))                     # 3 
    file1 = prefix + '-part1-summary.txt'
    sys.stdout = open(file1, 'w')
    print 'initial area:', init_area
    print 'retiming vector:', r
    print 'phi_opt:', cp
    print 'final area:', final_area
    sys.stdout.close()
    file2 = prefix + '-part1-CDFG-output.txt'
    sys.stdout = open(file2, 'w')
    print 'retimed graph vertices: ', G_r.node
    print 'retimed graph edges: ', G_r.edge
    sys.stdout.close()

    # part 2
    c = generate_c(G)                                       # 4
    t, bv = generate_tableau(G, W, D, cycle_time, prefix)   # 5, 6
    r = simplex_solve(G, t, bv, mode, prefix)               # 1, 7
    G_r = retime_graph(G, r)                                # 2
    final_area = count_registers(G, r, init_area)           # 3
    cp = max(clock_period(G_r))                     

    file1 = prefix + '-part2-summary.txt'
    sys.stdout = open(file1, 'w')
    print 'initial area:', init_area
    print 'retiming vector:', r
    print 'final area:', final_area
    sys.stdout.close()
    file2 = prefix + '-part2-CDFG-output.txt'
    sys.stdout = open(file2, 'w')
    print 'retimed graph vertices: ', G_r.node
    print 'retimed graph edges: ', G_r.edge
    sys.stdout.close()
    file3 = prefix + '-part2-c-vector.txt'
    sys.stdout = open(file3, 'w')
    print 'c vector: ', c
    sys.stdout.close()


main()
