from hash import Hash

class Graph():

    def __init__(self, num_vertices: int):
        self.adj_list = []
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

    def insertEdge(self, node_a, node_b):
        id_a = self.node_ids.lookup(str(hash(node_a)))
        id_b = self.node_ids.lookup(str(hash(node_b)))
        if id_a is None or id_b is None or node_b not in self.vertices or node_a not in self.vertices or id_a == id_b:
            raise Exception("error: one of the two nodes is not in the graph")
        self.adj_list[id_a].add(node_b)
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
            try:
                self.adj_list[i].remove(node)
            except KeyError:
                continue

    def deleteEdge(self, node_a, node_b):
        if node_a not in self.vertices or node_b not in self.vertices:
            raise Exception("error: one of the two nodes not in the graph")
        node_a_id = self.node_ids.lookup(str(hash(node_a)))
        self.adj_list[node_a_id].remove(node_b)

    def adj(self, node):
        if node not in self.vertices:
            raise Exception("error: node not in the graph")
        node_id = self.node_ids.lookup(str(hash(node)))
        return self.adj_list[node_id]

    def v(self):
        return list(self.vertices)


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
    graph.insertEdge(a, b)
    graph.insertEdge(a, c)
    graph.insertEdge(a, d)
    graph.insertEdge(b, a)
    graph.insertEdge(b, c)
    graph.insertEdge(a, b)
    print(graph.adj(a))
    print(graph.adj(b))
    graph.deleteNode(c)
    print(graph.adj(a))
    print(graph.adj(b))

if __name__ == "__main__":
    main()


