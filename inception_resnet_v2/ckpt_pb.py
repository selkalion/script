
import tensorflow as tf
from tensorflow.contrib import slim

from nets import inception
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from preprocessing import inception_preprocessing

checkpoints_dir = './data/training'
OUTPUT_PB_FILENAME = 'minimal_graph.proto'
NUM_CLASSES = 8

# We need default size of image for a particular network.
# The network was trained on images of that size -- so we
# resize input image later in the code.
image_size = inception.inception_resnet_v2.default_image_size

with tf.Graph().as_default():
    # Inject placeholder into the graph
    input_image_t = tf.placeholder(tf.string, name='input_image')
    image = tf.image.decode_jpeg(input_image_t, channels=3)

    # Resize the input image, preserving the aspect ratio
    # and make a central crop of the resulted image.
    # The crop will be of the size of the default image size of
    # the network.
    # I use the "preprocess_for_eval()" method instead of "inception_preprocessing()"
    # because the latter crops all images to the center by 85% at
    # prediction time (training=False).
    processed_image = inception_preprocessing.preprocess_for_eval(image,
                                                                  image_size,
                                                                  image_size, central_fraction=None)

    # Networks accept images in batches.
    # The first dimension usually represents the batch size.
    # In our case the batch size is one.
    processed_images = tf.expand_dims(processed_image, 0)

    # Load the inception network structure
    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
        logits, _ = inception.inception_resnet_v2(processed_images,
                                                  num_classes=NUM_CLASSES,
                                                  is_training=False)
    # Apply softmax function to the logits (output of the last layer of the network)
    probabilities = tf.nn.softmax(logits)

    model_path = tf.train.latest_checkpoint(checkpoints_dir)

    # Get the function that initializes the network structure (its variables) with
    # the trained values contained in the checkpoint
    init_fn = slim.assign_from_checkpoint_fn(
        model_path,
        slim.get_model_variables())

    with tf.Session() as sess:
        # Now call the initialization function within the session
        init_fn(sess)

        # Convert variables to constants and make sure the placeholder input_image is included
        # in the graph as well as the other neccesary tensors.
        constant_graph = convert_variables_to_constants(sess, sess.graph_def, ["input_image", "DecodeJpeg",
                                                                               "InceptionResnetV2/Logits/Predictions"])

        # Define the input and output layer properly
        optimized_constant_graph = optimize_for_inference(constant_graph, ["input_image"],
                                                          ["InceptionResnetV2/Logits/Predictions"],
                                                          tf.string.as_datatype_enum)
        # Write the production ready graph to file.
        tf.train.write_graph(optimized_constant_graph, '.', OUTPUT_PB_FILENAME, as_text=False)