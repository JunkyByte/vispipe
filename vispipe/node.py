class Node:
    def __init__(self, node_block, **kwargs):
        self.block = node_block
        self.custom_args = kwargs
        self._hash = None
        self.out_queues = []
        self.clear_out_queues()

    def clear_out_queues(self):
        self.out_queues = []
        for _ in range(self.block.num_outputs()):
            self.out_queues.append([])

    def __hash__(self):
        if self._hash is None:
            return id(self)
        else:
            return self._hash

    def __getstate__(self):
        state = self.__dict__.copy()
        state['out_queues'] = [[] for _ in range(self.block.num_outputs())]
        return state

