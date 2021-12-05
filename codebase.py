# -*- coding: utf-8 -*-
"""
@author: Nathan
"""

import time
from random import seed, randint, gauss
import random
import os

"""
Code for Adjacency List/Graph Data Structure
created with reference:
    https://www.programiz.com/dsa/graph-adjacency-list
"""
class Graph:
    def __init__(self, num):
        self.V = num
        self.graph = {}
        for j in range(num):
            self.graph[j] = []
            
            
    def add_edge(self, src, dest):
        if not dest in self.graph[src] and src != dest:
            self.graph[src].append(dest)
            self.graph[dest].append(src)
            return True
        else:
            return False
        
    def add_edge_single(self, src, dest):
        if src != dest:
            self.graph[src].append(dest)
            return True
        else:
            return False
    
    def print_graph(self):
        for i in range(self.V):
            print("Vertex " + str(i) + ":", end="")
            for edge in self.graph[i]:
                print(" -> {}".format(edge), end="")
            print(" \n")    
    
    def remove_edge(self, src, dest):
        if dest in self.graph[src] and src != dest:
            #del
            self.graph[src].remove(dest)
            self.graph[dest].remove(src)
            return True
        else:
            return False
            
class SLVOGraphRep:
    def __init__(self, graph):
        self.graph = graph
        self.V = graph.V
        self.deleted = [False]*self.V
        self.maxDegree = [None]*self.V
        self.totalMaxDegree = 0
        self.currDegree = [None]*self.V
        self.point = [None]*self.V
        
        self.secondGraph = {}

        for j in range(self.V):
            deg = len(graph.graph[j])
            if deg > self.totalMaxDegree:
                self.totalMaxDegree = deg
            self.maxDegree[j] = self.currDegree[j] = deg
            if deg in self.secondGraph:
                self.secondGraph[deg].append(j)
                self.point[j] = len(self.secondGraph[deg])-1
            else:
                self.secondGraph[deg] = [j]
                self.point[j] = 0

        
    
    def SLVO(self):
        degree_when_deleted = []
        order = []

        while len(order) < self.V:
            keys = [*self.secondGraph]
            keys = [x for x in keys if len(self.secondGraph[x]) > 0]
            keys.sort()

            min_degree = keys[0]
            vertex_to_delete = self.secondGraph[min_degree][len(self.secondGraph[min_degree])-1]
            #print("second_graph", self.secondGraph)
            order.append(vertex_to_delete)
            degree_when_deleted.append(min_degree)

            self.secondGraph[min_degree].pop() #this will run in constant time
            self.deleted[vertex_to_delete] = True

            for edge_v in self.graph.graph[vertex_to_delete]:
                
                if self.deleted[edge_v] is False:
                    position = (self.currDegree[edge_v], self.point[edge_v])

                    if position[1] != len(self.secondGraph[position[0]])-1:
                        self.secondGraph[position[0]][position[1]] = self.secondGraph[position[0]][len(self.secondGraph[position[0]])-1]
                        self.point[self.secondGraph[position[0]][len(self.secondGraph[position[0]])-1]] = position[1]

                    self.secondGraph[position[0]].pop() #this will run in constant time since it exists at end of structure

                    self.currDegree[edge_v] -= 1

                    if self.currDegree[edge_v] in self.secondGraph:
                        self.secondGraph[self.currDegree[edge_v]].append(edge_v)
                        self.point[edge_v] = len(self.secondGraph[self.currDegree[edge_v]]) - 1
                    else:
                        self.secondGraph[self.currDegree[edge_v]] = [edge_v]
                        self.point[edge_v] = 0
        
                    
        #self.reset_graph()
        return {"order": order, "deleted_degrees":degree_when_deleted}
    
    def LLVO(self):
        order = []

        while len(order) < self.V:
            keys = [*self.secondGraph]
            keys = [x for x in keys if len(self.secondGraph[x]) > 0]
            keys.sort()

            max_degree = keys[len(keys)-1]
            vertex_to_delete = self.secondGraph[max_degree][len(self.secondGraph[max_degree])-1]
            order.append(vertex_to_delete)

            self.secondGraph[max_degree].pop() #this will run in constant time
            self.deleted[vertex_to_delete] = True

            for edge_v in self.graph.graph[vertex_to_delete]:
                
                if self.deleted[edge_v] is False:
                    position = (self.currDegree[edge_v], self.point[edge_v])

                    if position[1] != len(self.secondGraph[position[0]])-1:
                        self.secondGraph[position[0]][position[1]] = self.secondGraph[position[0]][len(self.secondGraph[position[0]])-1]
                        self.point[self.secondGraph[position[0]][len(self.secondGraph[position[0]])-1]] = position[1]

                    self.secondGraph[position[0]].pop() #this will run in constant time since it exists at end of structure

                    self.currDegree[edge_v] -= 1

                    if self.currDegree[edge_v] in self.secondGraph:
                        self.secondGraph[self.currDegree[edge_v]].append(edge_v)
                        self.point[edge_v] = len(self.secondGraph[self.currDegree[edge_v]]) - 1
                    else:
                        self.secondGraph[self.currDegree[edge_v]] = [edge_v]
                        self.point[edge_v] = 0
                    
        #self.reset_graph()
        return order

    def smallest_original_degree(self):
        order = []

        while len(order) < self.V:
            keys = [*self.secondGraph]
            keys = [x for x in keys if len(self.secondGraph[x]) > 0]
            keys.sort()

            min_degree = keys[0]
            vertex_to_delete = self.secondGraph[min_degree][len(self.secondGraph[min_degree])-1]
            order.append(vertex_to_delete)

            self.secondGraph[min_degree].pop() #this will run in constant time
            self.deleted[vertex_to_delete] = True
                    
        #self.reset_graph()
        return order

    def uniform_random_ordering(self):
        order = []
        vertices = list(range(0, self.V))

        while len(order) < self.V-1:
            vertex = randint(0,len(vertices)-1)
            order.append(vertices[vertex])
            vertices[vertex] = vertices[len(vertices)-1]
            vertices.pop()
        order.append(vertices[0])

        #self.reset_graph()
        return order

    def SLVO_with_swaps(self):
        ordering = self.SLVO()
        order = ordering["order"]

        num_swaps = randint(1, self.V*2)
        for j in range(num_swaps):
            swap_pos_1 = randint(0, len(order)-1)
            swap_pos_2 = randint(0, len(order)-1)

            temp = order[swap_pos_1]
            order[swap_pos_1] = order[swap_pos_2]
            order[swap_pos_2] = temp

        #self.reset_graph()
        return order


    def depth_first_search(self):
        order = []
        stack = [0]
        
        while stack:
            node = stack.pop()
            if node not in order:
                order.append(node)
                stack.extend([x for x in self.graph.graph[node] if x not in order])
        return order

    def color_graph_SLVO(self):
        ordering = self.SLVO()
        deletion_degrees = ordering["deleted_degrees"]
        order = ordering["order"]

        max_deleted_degree = max(deletion_degrees)

        #Get size of terminal clique
        terminal_size = 0
        for j in range(len(order)-1, -1, -1):
            if deletion_degrees[j] == terminal_size:
                terminal_size += 1
            else:
                break

        original_degrees = []
        for node in order:
            original_degrees.append(self.maxDegree[node])

        average_degree = sum(self.maxDegree)/self.V
        
        start_time = time.time()
        colors = [None]*self.V
        colored = [-1]*self.V
        #give 1st node default color
        colors[0] = 0
        colored[order[0]] = 0
        totalNumberColors = 1


        for index, node in enumerate(order[1:], 1):
            #print("index, node", (index, node))
            color_options = [True]*self.V
            for adjacent in self.graph.graph[node]:
                if colored[adjacent] != -1:
                    color_options[colored[adjacent]] = False
            color = 0
            while color < self.V:
                if color_options[color]:
                    break
                color += 1
            
            colors[index] = color
            colored[node] = color
            if color == totalNumberColors:
                totalNumberColors += 1
        
        end_time = time.time()-start_time
        
        return {
                "colors":colors, 
                "originalDegrees":original_degrees,
                "averageOriginalDegree":average_degree,
                "deletedDegrees":deletion_degrees,
                "maxDeletedDegree":max_deleted_degree,
                "terminalSize": terminal_size,
                "totalNumColors":totalNumberColors,
                "time":end_time
                #terminal clique
                }

    def color_graph_LLVO(self):
        order = self.LLVO()

        original_degrees = []
        for node in order:
            original_degrees.append(self.maxDegree[node])

        average_degree = sum(self.maxDegree)/self.V
        

        start_time = time.time()
        colors = [None]*self.V
        colored = [-1]*self.V
        #give 1st node default color
        colors[0] = 0
        colored[order[0]] = 0
        totalNumberColors = 1

        for index, node in enumerate(order[1:], 1):
            #print("index, node", (index, node))
            color_options = [True]*self.V
            for adjacent in self.graph.graph[node]:
                if colored[adjacent] != -1:
                    color_options[colored[adjacent]] = False
            color = 0
            while color < self.V:
                if color_options[color]:
                    break
                color += 1
            
            colors[index] = color
            colored[node] = color
            if color == totalNumberColors:
                totalNumberColors += 1
        
        end_time = time.time()-start_time
        
        return {
                "colors":colors, 
                "originalDegrees":original_degrees,
                "averageOriginalDegree":average_degree,
                "totalNumColors":totalNumberColors,
                "time":end_time
                }

    def color_graph_smallest_original(self):
        order = self.smallest_original_degree()

        original_degrees = []
        for node in order:
            original_degrees.append(self.maxDegree[node])

        average_degree = sum(self.maxDegree)/self.V
        
        start_time = time.time()
        colors = [None]*self.V
        colored = [-1]*self.V
        #give 1st node default color
        colors[0] = 0
        colored[order[0]] = 0
        totalNumberColors = 1

        for index, node in enumerate(order[1:], 1):
            #print("index, node", (index, node))
            color_options = [True]*self.V
            for adjacent in self.graph.graph[node]:
                if colored[adjacent] != -1:
                    color_options[colored[adjacent]] = False
            color = 0
            while color < self.V:
                if color_options[color]:
                    break
                color += 1
            
            colors[index] = color
            colored[node] = color
            if color == totalNumberColors:
                totalNumberColors += 1
        
        end_time = time.time() - start_time
        
        return {
                "colors":colors, 
                "originalDegrees":original_degrees,
                "averageOriginalDegree":average_degree,
                "totalNumColors":totalNumberColors,
                "time":end_time
                }

    def color_graph_uniform_random(self):
        order = self.uniform_random_ordering()

        original_degrees = []
        for node in order:
            original_degrees.append(self.maxDegree[node])

        average_degree = sum(self.maxDegree)/self.V
        

        start_time = time.time()
        colors = [None]*self.V
        colored = [-1]*self.V
        #give 1st node default color
        colors[0] = 0
        colored[order[0]] = 0
        totalNumberColors = 1

        for index, node in enumerate(order[1:], 1):
            #print("index, node", (index, node))
            color_options = [True]*self.V
            for adjacent in self.graph.graph[node]:
                if colored[adjacent] != -1:
                    color_options[colored[adjacent]] = False
            color = 0
            while color < self.V:
                if color_options[color]:
                    break
                color += 1
            
            colors[index] = color
            colored[node] = color
            if color == totalNumberColors:
                totalNumberColors += 1
        
        end_time = time.time() - start_time
        
        return {
                "colors":colors, 
                "originalDegrees":original_degrees,
                "averageOriginalDegree":average_degree,
                "totalNumColors":totalNumberColors,
                "time":end_time
                }

    def color_graph_SLVO_swaps(self):
        order = self.SLVO_with_swaps()

        original_degrees = []
        for node in order:
            original_degrees.append(self.maxDegree[node])

        average_degree = sum(self.maxDegree)/self.V
        
        
        start_time = time.time()
        colors = [None]*self.V
        colored = [-1]*self.V
        #give 1st node default color
        colors[0] = 0
        colored[order[0]] = 0
        totalNumberColors = 1

        for index, node in enumerate(order[1:], 1):
            #print("index, node", (index, node))
            color_options = [True]*self.V
            for adjacent in self.graph.graph[node]:
                if colored[adjacent] != -1:
                    color_options[colored[adjacent]] = False
            color = 0
            while color < self.V:
                if color_options[color]:
                    break
                color += 1
            
            colors[index] = color
            colored[node] = color
            if color == totalNumberColors:
                totalNumberColors += 1
        
        end_time = time.time() - start_time
        
        return {
                "colors":colors, 
                "originalDegrees":original_degrees,
                "averageOriginalDegree":average_degree,
                "totalNumColors":totalNumberColors,
                "time":end_time
                }

    def color_graph_DFS(self):
        order = self.depth_first_search()

        original_degrees = []
        for node in order:
            original_degrees.append(self.maxDegree[node])

        average_degree = sum(self.maxDegree)/self.V
        
        start_time = time.time()
        colors = [None]*self.V
        colored = [-1]*self.V
        #give 1st node default color
        colors[0] = 0
        colored[order[0]] = 0
        totalNumberColors = 1

        for index, node in enumerate(order[1:], 1):
            #print("index, node", (index, node))
            color_options = [True]*self.V
            for adjacent in self.graph.graph[node]:
                if colored[adjacent] != -1:
                    color_options[colored[adjacent]] = False
            color = 0
            while color < self.V:
                if color_options[color]:
                    break
                color += 1
            
            colors[index] = color
            colored[node] = color
            if color == totalNumberColors:
                totalNumberColors += 1
        
        end_time = time.time() - start_time
        
        return {
                "colors":colors, 
                "originalDegrees":original_degrees,
                "averageOriginalDegree":average_degree,
                "totalNumColors":totalNumberColors,
                "time":end_time
                }

    def reset_graph(self):
        self.deleted = [False]*self.V
        self.currDegree = [None]*self.V
        self.point = [None]*self.V

        for j in range(self.V):
            deg = len(self.graph.graph[j])
            self.currDegree[j] = deg
            if deg in self.secondGraph:
                self.secondGraph[deg].append(j)
                self.point[j] = len(self.secondGraph[deg])-1



    def print_graph(self):
        print("graph: ")
        self.graph.print_graph()
        
        print("deleted nodes: ", self.deleted)
        print("max node degrees: ", self.maxDegree)
        print("current node degrees: ", self.currDegree)
        print("graph -> degree pointers: ", self.point)
        print("node degrees: ", self.secondGraph)
        

def createCompleteGraph(V):
    newGraph = Graph(V)
    for j in range(V):
        for k in range(j+1, V):
            newGraph.add_edge(j, k)
    
    return newGraph
    
def createCycleGraph(V):
    newGraph = Graph(V)
    for j in range(V-1):
        newGraph.add_edge(j, j+1)
    newGraph.add_edge(V-1, 0)
    
    return newGraph

def createRandomGraph(V, E, DIST="uniform"):
    newGraph = Graph(V)

    mean = (V-1)/2 #(0+V-1)/2
    std = V/3 #(0+V)/3, shifts to account for 3 standard deviations (should cover pretty much any vertex)

    if V < 2000 or E < V*(V-1)/2*3/4:    
        for j in range(E):
            flag = False
            while(not flag):
                if DIST == "uniform":
                    x = randint(0,V-1)
                    y = randint(0,V-1)
                elif DIST == "skewed":
                    x = int(random.triangular(0, V-1,1))
                    y = int(random.triangular(0, V-1,1))
                elif DIST == "normal":
                    x = int(abs(random.gauss(mean, std)))
                    while x >= V:
                        x = int(abs(random.gauss(mean, std)))
                    y = int(abs(random.gauss(mean, std)))
                    while y >= V:
                        y = int(abs(random.gauss(mean, std)))

                flag = newGraph.add_edge(x, y)
    elif E == V*(V-1)/2:
        newGraph = createCompleteGraph(V)
    else:
        newGraph = createCompleteGraph(V)
        numToDel = int(V*(V-1)/2 - E)
        for j in range(numToDel):
            flag = False
            while(not flag):
                if DIST == "uniform":
                    x = randint(0,V-1)
                    y = randint(0,V-1)
                elif DIST == "skewed":
                    x = int(random.triangular(0, V-1,1))
                    y = int(random.triangular(0, V-1,1))
                elif DIST == "normal":
                    x = int(abs(random.gauss(mean, std)))
                    while x >= V:
                        x = int(abs(random.gauss(mean, std)))
                    y = int(abs(random.gauss(mean, std)))
                    while y >= V:
                        y = int(abs(random.gauss(mean, std)))

                flag = newGraph.remove_edge(x, y)
    
    return newGraph

#Utility function for creating random numbers with a normal/gaussian distribution
#Sourced from: https://stackoverflow.com/questions/36894191/how-to-get-a-normal-distribution-within-a-range-in-numpy
def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def writeGraphToFile(fname, graph):
    outF = open(fname, "w")
    
    # Number of nodes in Graph
    num = graph.V
    outF.write(str(num) + "\n")
    
    offset = 1
    for j in range(num):
        outF.write(str(num+offset) + "\n")
        offset += len(graph.graph[j])
    
    for key in graph.graph.keys():
        for edge in graph.graph[key]:
            outF.write(str(edge) + "\n")
    
    outF.close()
    

def readGraphFromFile(fname):
    lines = []
    with open(fname) as f:
        lines = f.readlines()
    #strip any whitespace
    lines = list(map(str.strip,lines))
    #convert from strings to ints
    lines = list(map(int, lines))
    
    numV = lines[0]
    VList = [None]*numV
    
    graph = Graph(numV)
    
    #get number of edges each vertex has
    for line in range(1, numV+1):
        VList[line-1] = lines[line]

    currLine = 1+numV
    currV = -1
    
    for j in range(currLine, len(lines)):
        if currV < len(VList)-1:
            if j == VList[currV + 1]:
                currV += 1
        
        graph.add_edge_single(currV, lines[j])
    
    return graph


if __name__ == "__main__":
    '''
    Max Vertices: 10000
    Max Edges: 2Million
    '''
    seed(time.time())
    
    
#     pp = readGraphFromFile("complete_4.txt")
#     pp.print_graph()
#     pp2 = SLVOGraphRep(pp)
#     print(pp2.SLVO())
#
#
#     ff = createRandomGraph(7, 10, "skewed")
#     ff.print_graph()
#     writeGraphToFile("random_skewed_7_10.txt", ff)


#    ff = readGraphFromFile("random_7_10.txt")
#    ff.print_graph()
#    ff2 = SLVOGraphRep(ff)
#    print("SLVO", ff2.SLVO())
#    ff2.reset_graph()
#    print("Coloring", ff2.color_graph_SLVO())
    
    
#    print("LLVO", ff2.LLVO())
#     print("small original", ff2.smallest_original_degree())
#     print("uniform random", ff2.uniform_random_ordering())
#     print("SLVO with swaps", ff2.SLVO_with_swaps())
#     print("depth first search", ff2.depth_first_search())
#    pp2.print_graph()
    
    
    
    
#    test = [100, 200, 400, 500, 1000, 2000, 4000, 5000]
#    times = []
#    for n in test:
#        start_time = time.time()
#        g = createCompleteGraph(n)
#        times.append(time.time() - start_time)
#        print(time.time() - start_time)
#        writeGraphToFile("complete_" + str(n) + ".txt", g)
#        
#    
#    test2 = [5000, 10000, 15000, 20000, 40000, 50000, 100000]
#    timesCycle = []
#    for n in test2:
#        start_time = time.time()
#        g = createCycleGraph(n)
#        timesCycle.append(time.time() - start_time)
#        print(time.time() - start_time)
#        writeGraphToFile("cycle_" + str(n) + ".txt", g)
#    
#    
#    testV = [100, 200, 400, 500, 1000, 2000, 4000]
#    testE = [0, 500, 1000, 2000, 4000, 10000, 20000, 40000]
#    types = ["uniform", "skewed", "normal"]
#    randomValues = []
#    timesRandom = []
#    for n in testV:
#        edges = [x for x in testE if x <= (n*(n-1)/2)]
#        for e in edges:
#            for type in types:                
#                randomValues.append((n, e, type))
#                start_time = time.time()
#                g = createRandomGraph(n, e, type)
#                timesRandom.append(time.time() - start_time)
#                print(time.time() - start_time)
#                writeGraphToFile("random_" + type + "_" + str(n) + "_" + str(e) + ".txt", g)
#    

#    testV2000 = [5000, 10000, 20000]
#    times2000 = []
#    for n in testV2000:
#        start_time = time.time()
#        g = createRandomGraph(n, 2000)
#        times2000.append(time.time() - start_time)
    
    
    
    test = [100, 200, 400, 500, 1000, 2000, 4000, 5000]
    test2 = [5000, 10000, 15000, 20000, 40000, 50000, 100000]
    SLVOtimes = []
    SLVOoutput = []
    
    smallestOtimes = []
    smallestOoutput = []
    
    uRandtimes = []
    uRandoutput = []
    
    LLVOtimes = []
    LLVOoutput = []
    
    SLVOswapTimes = []
    SLVOswapoutput = []
    
    dfsTimes = []
    dfsoutput = []
    
#    for n in test2:
#        g = readGraphFromFile("cycle_" + str(n) + ".txt")
#        g2 = SLVOGraphRep(g)
#
#        start_time = time.time()        
#        o = g2.SLVO()
#        SLVOtimes.append(time.time() - start_time)
#        SLVOoutput.append(o)
#        print(time.time()-start_time)

#        g2.reset_graph()
#        start_time = time.time()
#        o = g2.smallest_original_degree()
#        smallestOtimes.append(time.time() - start_time)
#        smallestOoutput.append(0)
#        print(time.time() - start_time)
#        
#        g2.reset_graph()
#        start_time = time.time()
#        o = g2.uniform_random_ordering()
#        uRandtimes.append(time.time() - start_time)
#        uRandoutput.append(o)
#        print(time.time() - start_time)
#        
#        g2.reset_graph()
#        start_time = time.time()
#        o = g2.LLVO()
#        LLVOtimes.append(time.time() - start_time)
#        LLVOoutput.append(o)
#        print(time.time() - start_time)
#        
#        g2.reset_graph()
#        start_time = time.time()
#        o = g2.SLVO_with_swaps()
#        SLVOswapTimes.append(time.time() - start_time)
#        SLVOswapoutput.append(o)
#        print(time.time() - start_time)
#        
#        g2.reset_graph()
#        start_time = time.time()
#        o = g2.depth_first_search()
#        dfsTimes.append(time.time() - start_time)
#        dfsoutput.append(o)
#        print(time.time() - start_time)
    
#    testV = [1000,2000,4000,5000,10000]
#    testE = [2000,4000,10000,20000,40000]
#    randomValues = []
#    timesRandom = [[],[],[],[],[]]
#    for n in testV:
##        edges = [x for x in testE if x <= (n*(n-1)/2)]
#        for row, e in enumerate(testE):
#            if e <= (n*(n-1)/2):
#                g = readGraphFromFile("random_uniform_" + str(n) +  "_" + str(e) + ".txt")
#                g2 = SLVOGraphRep(g)            
#                o = g2.color_graph_SLVO()
#                timesRandom[row].append(o["time"])
#                print(n, e, o["time"])
#            else:
#                print("skipped")
#                timesRandom[row].append("-")
   
                
    testV = [1500, 10000]
    testE = [200000, 600000]
    maxDegrees = []
        
    
#    for v in testV:
#        for e in testE:
#            g = createRandomGraph(v, e, "normal")  #readGraphFromFile("random_uniform" + str(4000) + ".txt")
#            g2 = SLVOGraphRep(g)
#            
#            start_time = time.time()        
#            o = g2.color_graph_SLVO() #g2.SLVO()
#            SLVOoutput.append(o["totalNumColors"])
#            print(time.time()-start_time)
#            
#            g2.reset_graph()                    
    
    g = createRandomGraph(5000, 1000000, "normal")  #readGraphFromFile("random_uniform" + str(4000) + ".txt")
    g2 = SLVOGraphRep(g)
    
    start_time = time.time()        
    o = g2.color_graph_SLVO() #g2.SLVO()
    #SLVOoutput.append(o["totalNumColors"])
    f = open("SLVO_analysis.txt", "w")
    for degree in o["deletedDegrees"]:
        f.write(str(degree) + "\n")
    f.close()
    print(time.time()-start_time)
                