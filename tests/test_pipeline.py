import unittest
from vispipe import Pipeline
import logging
logging.disable(logging.CRITICAL)


class TestPipelineOutputs(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline()
        self.node = self.pipeline.add_node('np_iter_file', path='tests/data/10elementarray.npy')

    def tearDown(self):
        self.pipeline.clear_pipeline()

    def test_missing_output(self):
        self.pipeline.run()
        with self.assertRaises(KeyError):
            _ = self.pipeline.outputs[self.node]

    def test_working_output(self):
        self.pipeline.add_output(self.node)
        self.pipeline.run()
        out = list(self.pipeline.outputs[self.node])
        self.assertEqual(list(range(10)), out)

    def test_output_naming(self):
        self.pipeline.get_node(self.node).name = 'named_file'
        self.pipeline.add_output('named_file')
        self.pipeline.run()
        _ = self.pipeline.outputs['named_file']

    def test_output_accessing(self):
        self.pipeline.get_node(self.node).name = 'named_file'
        self.pipeline.add_output('named_file')
        self.pipeline.run()
        access_name = self.pipeline.outputs['named_file']
        with self.assertRaises(KeyError):
            self.pipeline.outputs[self.node]
        access_hash = self.pipeline.get_output(self.node)  # Correctly access the output by hash
        self.assertEqual(access_name, access_hash)


class TestPipelineNodesAndConnections(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline()

    def tearDown(self):
        self.pipeline.clear_pipeline()

    def test_add_node(self):
        node_hash = self.pipeline.add_node(list(self.pipeline._blocks.keys())[0])
        self.assertEqual(len(self.pipeline.nodes), 1)
        self.assertEqual(hash(self.pipeline.nodes[0]), node_hash)

    def test_remove_node(self):
        node_hash = self.pipeline.add_node(list(self.pipeline._blocks.keys())[0])
        self.pipeline.remove_node(node_hash)
        with self.assertRaises(KeyError):
            self.pipeline.get_node(node_hash)


if __name__ == '__main__':
    unittest.main()
