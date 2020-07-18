from classes.constant import MODEL_PATH, MODEL_NAME
from classes.model_definition import compile_model


def get_concrete_function():
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    compile_model(model)

    full_model = tf.function(lambda x: model(x))

    x_tensor_spec = tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    y_tensor_spec = tf.TensorSpec(model.inputs[1].shape, model.inputs[1].dtype)

    return full_model.get_concrete_function(x=[x_tensor_spec, y_tensor_spec])


def convert_to_tf_lite(path):
    import tensorflow as tf
    cf = get_concrete_function()

    converter = tf.lite.TFLiteConverter.from_concrete_functions([cf])
    lite_model = converter.convert()

    with tf.io.gfile.GFile(path, 'wb') as f:
        f.write(lite_model)


def write_graph():
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    full_model = get_concrete_function()
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./models",
                      name=MODEL_NAME + '.pb',
                      as_text=False)


def read_graph():
    import tensorflow as tf
    with tf.io.gfile.GFile('./models/' + MODEL_NAME + '.pb', "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())
    return loaded
