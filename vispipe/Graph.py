from hash import Hash

class Graph():

    def __init__(self, num_vertices: int):
        self.adj_list = []  # adj[i] is populated with a tuple (node, out_idx, in_idx, bool) where bool is true if the connection is to the node, false if is from the node
        self.vertices = set()
        self.available_ids = set()
        self.node_ids = Hash(num_vertices)
        self.num_vertices = num_vertices
        self.last_id_used = 0

        for i in range(num_vertices):
            self.available_ids.add(i)
            self.adj_list.append(set())

    def insertNode(self, node):
        if not bool(self.available_ids) or node in self.vertices:
            raise Exception("Trying to add a node already present in the graph")
        node_id = self.available_ids.pop()
        self.node_ids.insert(str(hash(node)), node_id)
        self.vertices.add(node)
        if node_id > self.last_id_used:
            self.last_id_used = node_id
        return hash(node)

    def insertEdge(self, node_a, node_b, out_idx, in_idx):
        id_a = self.node_ids.lookup(str(hash(node_a)))
        id_b = self.node_ids.lookup(str(hash(node_b)))
        if id_a is None or id_b is None or node_b not in self.vertices or node_a not in self.vertices or id_a == id_b:
            raise Exception("error: one of the two nodes is not in the graph or trying to connect the same node")
        adj_b = self.adj(node_b).copy()
        for el in adj_b:
            direction = el[3] # get the direction : true if b->node, False if node->b
            if direction == False:
                node_in_idx = el[2]
                if node_in_idx == in_idx:
                    self.deleteEdge(el[0], node_b, el[1], el[2])
        self.adj_list[id_a].add((node_b, out_idx, in_idx, True))
        self.adj_list[id_b].add((node_a, out_idx, in_idx, False))
        #print(self.adj_list[id_a])

    def deleteNode(self, node):
        node_id = self.node_ids.lookup(str(hash(node)))
        if node not in self.vertices:
            raise Exception("error: node not in the graph")
        self.adj_list[node_id] = set()
        self.node_ids.remove(str(hash(node)))
        self.vertices.remove(node)
        self.available_ids.add(node_id)
        sorted(self.available_ids)
        for i in range(self.last_id_used + 1):
            adj = self.adj_list[i].copy()
            for el in adj:
                if el[0] == node:
                    self.adj_list[i].remove(el)

    def deleteEdge(self, node_a, node_b, out_idx, in_idx):
        if node_a not in self.vertices or node_b not in self.vertices:
            raise Exception("error: one of the two nodes not in the graph")
        node_a_id = self.node_ids.lookup(str(hash(node_a)))
        node_b_id = self.node_ids.lookup(str(hash(node_b)))
        node_a_tuple = (node_b, out_idx, in_idx, True)
        node_b_tuple = (node_a, out_idx, in_idx, False)
        self.adj_list[node_a_id].remove(node_a_tuple)
        self.adj_list[node_b_id].remove(node_b_tuple)

    def adj(self, node):
        if node not in self.vertices:
            raise Exception("error: node not in the graph")
        node_id = self.node_ids.lookup(str(hash(node)))
        return self.adj_list[node_id]

    def v(self):
        return list(self.vertices)

    def get_node(self, node_hash):
        for el in self.vertices:
            if node_hash == hash(el):
                return el
        raise Exception("get_node : node not in graph")

    def resetGraph(self):
        vertices_copy = self.vertices.copy()  # cant change set size during iteration
        for node in vertices_copy:
            self.deleteNode(node)


def main():
    def printset(a):
        for el in a:
            print(el)

    class node:
        def __init__(self, value):
            self.x = value
        def __repr__(self):
            return str(hash(self))

    a = node(1)
    print("nodo a: ", a)
    b = node(2)
    print("nodo b: ", b)
    c = node(3)
    print("nodo c: ", c)
    d = node(4)
    print("nodo d: ", d)

    graph = Graph(30)

    graph.insertNode(a)
    graph.insertNode(b)
    graph.insertNode(c)
    graph.insertNode(d)
    graph.insertEdge(a, b, 0, 1)
    # graph.insertEdge(a, c)
    # graph.insertEdge(a, d)
    # graph.insertEdge(b, a)
    # graph.insertEdge(b, c)
    # graph.insertEdge(a, b)
    # graph.resetGraph()
    # print(graph.v())
    print(graph.adj(a))
    print(graph.adj(b))
    graph.insertEdge(c, b, 2, 1)
    graph.insertEdge(d, c, 0, 0)
    print("adj a : ", graph.adj(a))
    print("adj b : ", graph.adj(b))
    print("adj c : ", graph.adj(c))
    print("adj d : ", graph.adj(d))
    graph.deleteNode(d)
    print("after d is deleted")
    print("adj a : ", graph.adj(a))
    print("adj b : ", graph.adj(b))
    print("adj c : ", graph.adj(c))
    #print("adj d : ", graph.adj(d))

if __name__ == "__main__":
    main()


