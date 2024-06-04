from collections import OrderedDict
import onnxruntime as rt
import numpy as np
import onnx
import onnx_graphsurgeon as gs

"""
Tiny tutorial of onnx graphsurgeon

#########Basic data structure
gs.Variable
 --similar to "tensor" in pytorch

gs.Constant
--const tensors

gs.Node
-- define operation: like add , relut
-- declare input and ouptut tensor(variable)

gs.Graph
-- define input,output tensors, and nodes

######## Baisc operation #############
add node (what we do here)
cut node
combine graph
"""


def show_node_info(node):
    for input in node.inputs:
        print(input)

def onnx_add_output(onnx_path, ops_type = "MatMul", node_name= "MatMul_7"):
    #find the node
    graph = gs.import_onnx(onnx.load(onnx_path))
    for node in graph.nodes:
        if node.op == ops_type and node.name == node_name :
            print(f"onnx output number: {len(node.outputs)} ")
            node.outputs[0].dtype = float
            node.outputs[0].name = "TensorN"
            graph.outputs.append(node.outputs[0])
    graph.cleanup().toposort()
    return gs.export_onnx(graph)

def main():
    onnx_path = "../onnx_export/self_attention.onnx"
    input_dicts = {}


    onnx_modify_fn = "addoutput.onnx"

    batch_size = 1
    seq_length = 12
    input_dim = 16
    input_arr = np.ones((1, seq_length, input_dim))
    onnx_modify =onnx_add_output(onnx_path)

    onnx.save( onnx_modify, onnx_modify_fn)
    print(f"save modified onnx file to {onnx_modify_fn}")

if __name__ =="__main__":
    main()