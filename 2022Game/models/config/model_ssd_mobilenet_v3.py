import graphsurgeon as gs
import tensorflow as tf

name = 'ssd_mobilenet_v3'
path = '/home/ubuntu/tensorflow_workspace/2020Game/models/trained_ssd_mobilenet_v3/best/' + name + '.pb'
TRTbin = 'TRT_' + name + '.bin'
output_name = ['NMS']
dims = [3,300,300]
layout = 7


def create_const_for_anchor_generator():
    """Creates a 'Const' node as an input to 'MultipleGridAnchorGenerator'
    Note the 'MultipleGridAnchorGenerator' TRT plugin node requires a
    [1.0, 1.0] array as input.
    Reference: https://stackoverflow.com/a/56296195/7596504
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow.core.framework.tensor_pb2 import TensorProto
    from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto

    value = np.array([1.0, 1.0], dtype=np.float32)
    dt = tf.as_dtype(value.dtype).as_datatype_enum
    tensor_shape = TensorShapeProto(
        dim=[TensorShapeProto.Dim(size=s) for s in value.shape])
    tensor_proto = TensorProto(
        tensor_content=value.tobytes(),
        tensor_shape=tensor_shape,
        dtype=dt)
    return tf.NodeDef(name='const_for_anchors',
                      op='Const',
                      attr={'value': tf.AttrValue(tensor=tensor_proto),
                            'dtype': tf.AttrValue(type=dt)})


def replace_addv2(graph):
    """Replace all 'AddV2' in the graph with 'Add'.
    NOTE: 'AddV2' is not supported by UFF parser.
    """
    for node in graph.find_nodes_by_op('AddV2'):
        gs.update_node(node, op='Add')
    return graph


def replace_fusedbnv3(graph):
    """Replace all 'FusedBatchNormV3' in the graph with 'FusedBatchNorm'.
    NOTE: 'FusedBatchNormV3' is not supported by UFF parser.
    https://devtalk.nvidia.com/default/topic/1066445/tensorrt/tensorrt-6-0-1-tensorflow-1-14-no-conversion-function-registered-for-layer-fusedbatchnormv3-yet/post/5403567/#5403567
    """
    for node in graph.find_nodes_by_op('FusedBatchNormV3'):
        gs.update_node(node, op='FusedBatchNorm')
    return graph


def add_plugin(graph):
    all_assert_nodes = graph.find_nodes_by_op("Assert")
    graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)

    all_identity_nodes = graph.find_nodes_by_op("Identity")
    graph.forward_inputs(all_identity_nodes)

    Input = gs.create_plugin_node(
        name="Input",
        op="Placeholder",
        dtype=tf.float32,
        shape=[None, 3, 300, 300]
    )

    PriorBox = gs.create_plugin_node(
        name="MultipleGridAnchorGenerator",
        op="GridAnchor_TRT",
        minSize=0.2,
        maxSize=0.95,
        aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
        variance=[0.1,0.1,0.2,0.2],
        #featureMapShapes=[32, 16, 8, 4, 2, 1],
        nmsFeatureMapShapes = [19, 10, 5, 3, 2, 1],
        numLayers=6
    )

    NMS = gs.create_plugin_node(
        name="NMS",
        op="NMS_TRT",
        inputs=['concat_box_conf', 'Squeeze', 'concat_priorbox'],
        shareLocation=1,
        varianceEncodedInTarget=0,
        backgroundLabelId=0,
        confidenceThreshold=.015,
        nmsThreshold=0.4,
        topK=100,
        keepTopK=100,
        numClasses=39, # 38 object + 1 for unknown class
        inputOrder=[1, 0, 2],
        confSigmoid=1,
        isNormalized=1
    )

    concat_priorbox = gs.create_node(
        "concat_priorbox",
        op="ConcatV2",
        dtype=tf.float32,
        inputs=['MultipleGridAnchorGenerator'],
        axis=2
    )

    concat_box_loc = gs.create_plugin_node(
        "concat_box_loc",
        op="FlattenConcat_TRT",
        dtype=tf.float32,
        axis=1,
        ignoreBatch=0  )

    concat_box_conf = gs.create_plugin_node(
        "concat_box_conf",
        op="FlattenConcat_TRT",
        dtype=tf.float32,
        axis=1,
        ignoreBatch=0 )

    
    # Create a dummy node in the 'MultipleGridAnchorGenerator' namespace.
    # This is a hack for 'ssd_mobilenet_v3_large/small'...
    if not any([n.startswith('MultipleGridAnchorGenerator/')
                for n in graph.node_map.keys()]):
        const = create_const_for_anchor_generator()
        dummy = gs.create_node(
            'MultipleGridAnchorGenerator/dummy_for_anchors',
            op='Dummy',  # not important here, node will be collapsed later
            inputs=['const_for_anchors']
        )
        graph.add(const)
        graph.add(dummy)

    namespace_plugin_map = {
        "MultipleGridAnchorGenerator": PriorBox,
        "Postprocessor": NMS,
        "Preprocessor": Input,
        "Cast": Input,
        "ToFloat": Input,  # Maybe replaced by Cast?
        "image_tensor": Input,
        'normalized_input_image_tensor': Input,
        'MultipleGridAnchorGenerator/Concatenate': concat_priorbox,
        "Concatenate": concat_priorbox,
        "concat": concat_box_loc,
        "concat_1": concat_box_conf
    }

    graph.collapse_namespaces(namespace_plugin_map)
    if 'anchors' in [node.name for node in graph.graph_outputs]:
        graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)
    graph.find_nodes_by_op("NMS_TRT")[0].input.remove("Input")

    if 'NMS' not in [node.name for node in graph.graph_outputs]:
        graph.remove(graph.graph_outputs, remove_exclusive_dependencies=False)
        if 'NMS' not in [node.name for node in graph.graph_outputs]:
            # We expect 'NMS' to be one of the outputs
            raise RuntimeError('bad graph_outputs')
    if 'Input' in list(graph.find_nodes_by_name('NMS')[0].input):
        graph.find_nodes_by_name('NMS')[0].input.remove('Input')
    if 'image_tensor:0' in list(graph.find_nodes_by_name('Input')[0].input):
        graph.find_nodes_by_name('Input')[0].input.remove('image_tensor:0')

    graph = replace_addv2(graph)
    graph = replace_fusedbnv3(graph)
    return graph
