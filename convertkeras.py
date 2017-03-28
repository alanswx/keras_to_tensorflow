#
#     Copywrite 2017 Alan Steremberg and Arthur Conner
#

import argparse
from keras import backend as K
from keras.models import load_model
#from tensorflow_serving.session_bundle import exporter
from keras.models import model_from_config
from keras.models import Sequential
import tensorflow as tf
import os


def convert(prevmodel,export_path,freeze_graph_binary):

   # open up a Tensorflow session
   sess = tf.Session()
   # tell Keras to use the session
   K.set_session(sess)

   # From this document: https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
   
   # let's convert the model for inference
   K.set_learning_phase(0)  # all new operations will be in test mode from now on
   # serialize the model and get its weights, for quick re-building
   previous_model = load_model(prevmodel)
   previous_model.summary()

   config = previous_model.get_config()
   weights = previous_model.get_weights()

   # re-build a model where the learning phase is now hard-coded to 0
   model= Sequential.from_config(config) 
   #model= model_from_config(config)
   model.set_weights(weights)

   print("Input name:")
   print(model.input.name)
   print("Output name:")
   print(model.output.name)
   output_name=model.output.name.split(':')[0]

   #  not sure what this is for
   export_version = 1 # version number (integer)

   graph_file=export_path+"_graph.pb"
   ckpt_file=export_path+".ckpt"
   # create a saver 
   saver = tf.train.Saver(sharded=True)
   tf.train.write_graph(sess.graph_def, '', graph_file)
   save_path = saver.save(sess, ckpt_file)
#~/tensorflow/bazel-bin/tensorflow/python/tools/freeze_graph --input_graph=./graph.pb  --input_checkpoint=./model.ckpt --output_node_names=add_72 --output_graph=frozen.pb
   command = freeze_graph_binary +" --input_graph=./"+graph_file+" --input_checkpoint=./"+ckpt_file+" --output_node_names="+output_name+" --output_graph=./"+export_path+".pb"
   print(command)
   os.system(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Keras Tensorflow Converter')
    parser.add_argument(
        'model',
        type=str,
        help='Path to the keras model'
    )
    parser.add_argument(
        'frozen',
        type=str,
        help='Path to the frozen output'
    )
    parser.add_argument(
        'freezegraph',
        type=str,
        help='Path to the freeze_graph binary'
    )
    args = parser.parse_args()

    convert(args.model,args.frozen,args.freezegraph)
