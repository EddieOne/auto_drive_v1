import tensorflow as tf, sys, os
import time
import v1

print('test loaded')

# Unpersists graph from file
with tf.gfile.FastGFile("graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("labels.txt")]

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    while (True):
        v1.capture()
        time.sleep(0.05)
        image_data = tf.gfile.FastGFile('image1.jpg', 'rb').read()
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print(human_string, score)
            v1.drive(human_string, score)
            break