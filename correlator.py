# Benjamin Zhu bjz2107
# Sean Liu     sl3497
# CSEE 6861
# Final Project: CAD Retiming Tool

import copy

def read_input_file(filename):
	f = open(filename)
	f.readline()
	vertices = {}
	edges = {}
	v_size = int(f.readline().split()[1])
	v_weights = f.readline().split()
	vertices[0] = 0
	for i in range(1, v_size+1):
		vertices[i] = int(v_weights[i])
	f.readline()
	line = f.readline()
	while not '.e' in line:
		line = line.split()
		pair = (int(line[0]), int(line[1]))
		edges[pair] = int(line[2])
		line = f.readline()
	f.close()
	return vertices, edges


def floyd_warshall(vertices, edges):
	"""
	taken from https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
	"""
	v = len(vertices)
	distances = []
	next = []
	for i in range(v):
		distance_row = []
		next_row = []
		for j in range(v):
			distance_row.append(float('inf'))
			next_row.append(None)
		distances.append(distance_row)
		next.append(next_row)
	for i in range(v):
		distances[i][i] = 0
	for edge_pair in edges:
		distances[edge_pair[0]][edge_pair[1]] = edges[edge_pair]
		next[edge_pair[0]][edge_pair[1]] = edge_pair[1]
	for k in range(v):
		for i in range(v):
			for j in range(v):
				d = distances[i][k] + distances[k][j]
				if distances[i][j] > d:
					distances[i][j] = d
					next[i][j] = next[i][k]
	W = distances
	D = copy.deepcopy(next)
	for u in range(len(next)):
		for v in range(len(next[u])):
			path = calculate_path(next, u, v)
			delay = 0
			for p in path:
				delay += vertices[p]
			if u == v:
				delay = vertices[v]
			D[u][v] = delay
	return W, D


def calculate_path(next, u, v):
	"""
	taken from https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm
	"""
	if next[u][v] == None:
		return []
	path = [u]
	while u != v:
		u = next[u][v]
		path.append(u)
	return path


def get_subgraph(vertices, edges, weight):
	v_new = {}
	e_new = {}
	for edge in edges:
		if edges[edge] == weight:
			e_new[edge] = weight
			v_new[edge[0]] = vertices[edge[0]]
			v_new[edge[1]] = vertices[edge[1]]
	return v_new, e_new


def topological_sort(vertices, edges):
	l = []
	s = []
	edges = copy.deepcopy(edges)
	for v in vertices:
		no_incoming_edge = True
		for edge in edges:
			if v == edge[1]:
				no_incoming_edge = False
				break
		if no_incoming_edge:
			s.append(v)
	while s:
		v = s.pop(0)
		l.append(v)
		for edge in edges.keys():
			n = edge[0]
			m = edge[1]
			if n == v:
				del edges[edge]
				no_incoming_edge = True
				for e in edges:
					if e[1] == m:
						no_incoming_edge = False
						break
				if no_incoming_edge:
					s.append(m)
	return l


def clock_period(vertices, edges):
	v, e = get_subgraph(vertices, edges, 0)
	v_ordered = topological_sort(v, e)
	delta_v = {}
	for vertex in v_ordered:
		v_delay = v[vertex]
		u_delay = 0
		for edge in e:
			u = edge[0]
			if edge[1] == vertex:
				if u in delta_v:
					u_delay = delta_v[u]
				else:
					u_delay = v[u]
		v_delay += u_delay
		delta_v[vertex] = v_delay
	return delta_v


def feas(vertices, edges, c):
	r = []
	for v in vertices:
		r.append(0)
	for i in range(len(vertices)-1):
		edges_new = copy.deepcopy(edges)
		for e in edges_new:
			edges_new[e] = edges_new[e] + r[e[1]] - r[e[0]]
		delta_v = clock_period(vertices, edges_new)
		for dv in delta_v:
			if delta_v[dv] > c:
				r[dv] += 1
	edges_new = copy.deepcopy(edges)
	for e in edges_new:
		edges_new[e] = edges_new[e] + r[e[1]] - r[e[0]]
	delta_v = clock_period(vertices, edges_new)
	time = max(delta_v)
	if time > c:
		return None
	return r


def opt2(vertices, edges, D):
	flattened = []
	for sublist in D:
	    for val in sublist:
	        flattened.append(val)
	flattened = sorted(flattened)
	r_list = []
	for f in flattened:
		r = feas(vertices, edges, f)
		if r:
			r_list.append(r)


vert, ed = read_input_file('corr_input.txt')
W, D = floyd_warshall(vert, ed)
for row in W:
	print row
print
for row in D:
	print row
print
print max(clock_period(vert, ed).values())