# Improved MNIST Architecture based on implementation provided in Genetic CNN Notebook.
import random
import numpy as np

from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from dag import DAG, DAGValidationError

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_imgs = mnist.train.images
train_labels = mnist.train.labels
test_imgs = mnist.test.images
test_labels = mnist.test.labels

train_imgs = np.reshape(train_imgs, [-1, 28, 28, 1])
test_imgs = np.reshape(test_imgs, [-1, 28, 28, 1])

STAGES = np.array(["s1", "s2"])  # S
NUM_NODES = np.array([3, 5])  # K

L = 0  # genome length
BITS_INDICES, l_bpi = np.empty((0, 2), dtype=np.int32), 0  # to keep track of bits for each stage S
for nn in NUM_NODES:
    t = nn * (nn - 1)
    BITS_INDICES = np.vstack([BITS_INDICES, [l_bpi, l_bpi + int(0.5 * t)]])
    l_bpi = int(0.5 * t)
    L += t
L = int(0.5 * L)

TRAINING_EPOCHS = 20
BATCH_SIZE = 20
TOTAL_BATCHES = train_imgs.shape[0] #BATCH_SIZE

def weight_variable(weight_name, weight_shape):
    return tf.Variable(tf.truncated_normal(weight_shape, stddev=0.1), name=''.join(["weight_", weight_name]))


def bias_variable(bias_name, bias_shape):
    return tf.Variable(tf.constant(0.01, shape=bias_shape), name=''.join(["bias_", bias_name]))


def linear_layer(x, n_hidden_units, layer_name):
    n_input = int(x.get_shape()[1])
    weights = weight_variable(layer_name, [n_input, n_hidden_units])
    biases = bias_variable(layer_name, [n_hidden_units])
    return tf.add(tf.matmul(x, weights), biases)


def apply_convolution(x, kernel_height, kernel_width, in_channels, out_chanels, layer_name):
    weights = weight_variable(layer_name, [kernel_height, kernel_width, in_channels, out_chanels])
    biases = bias_variable(layer_name, [out_chanels])
    return tf.nn.relu(tf.add(tf.nn.conv2d(x, weights, [1, 2, 2, 1], padding="SAME"), biases))


def apply_pool(x, kernel_height, kernel_width, stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1],
                          strides=[1, stride_size, stride_size, 1], padding="SAME")


def add_node(node_name, connector_node_name, h=5, w=5, ic=1, oc=1):
    with tf.name_scope(node_name) as scope:
        conv = apply_convolution(tf.get_default_graph().get_tensor_by_name(connector_node_name),
                                 kernel_height=h, kernel_width=w, in_channels=ic, out_chanels=oc,
                                 layer_name=''.join(["conv_", node_name]))


def sum_tensors(tensor_a, tensor_b, activation_function_pattern):
    if not tensor_a.startswith("Add"):
        tensor_a = ''.join([tensor_a, activation_function_pattern])

    return tf.add(tf.get_default_graph().get_tensor_by_name(tensor_a),
                  tf.get_default_graph().get_tensor_by_name(''.join([tensor_b, activation_function_pattern])))


def has_same_elements(x):
    return len(set(x)) <= 1


'''This method will come handy to first generate DAG independent of Tensorflow, 
    afterwards generated graph can be used to generate Tensorflow graph'''


def generate_dag(optimal_indvidual, stage_name, num_nodes):
    # create nodes for the graph
    nodes = np.empty((0), dtype=np.str)
    for n in range(1, (num_nodes + 1)):
        nodes = np.append(nodes, ''.join([stage_name, "_", str(n)]))

    # initialize directed asyclic graph (DAG) and add nodes to it
    dag = DAG()
    for n in nodes:
        dag.add_node(n)

    # split best indvidual found via GA to identify vertices connections and connect them in DAG
    edges = np.split(optimal_indvidual, np.cumsum(range(num_nodes - 1)))[1:]
    v2 = 2
    for e in edges:
        v1 = 1
        for i in e:
            if i:
                dag.add_edge(''.join([stage_name, "_", str(v1)]), ''.join([stage_name, "_", str(v2)]))
            v1 += 1
        v2 += 1

    # delete nodes not connected to anyother node from DAG
    for n in nodes:
        if len(dag.predecessors(n)) == 0 and len(dag.downstream(n)) == 0:
            dag.delete_node(n)
            nodes = np.delete(nodes, np.where(nodes == n)[0][0])

    return dag, nodes


def generate_tensorflow_graph(individual, stages, num_nodes, bits_indices):
    activation_function_pattern = "/Relu:0"

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="X")
    Y = tf.placeholder(tf.float32, [None, 10], name="Y")

    d_node = X
    for stage_index, stage_name, num_node, bpi in zip(range(0, len(stages)),stages, num_nodes, bits_indices):
        indv = individual[bpi[0]:bpi[1]]

        ic = 1
        oc = 1
        if stage_index == 0:
            add_node(''.join([stage_name, "_input"]), d_node.name, ic=1, oc=20)
            ic = 20
            oc = 20
        elif stage_index == 1:
            add_node(''.join([stage_name, "_input"]), d_node.name, ic=20, oc=50)
            ic = 50
            oc = 50

        pooling_layer_name = ''.join([stage_name, "_input", activation_function_pattern])

        if not has_same_elements(indv):
            # ------------------- Temporary DAG to hold all connections implied by GA solution ------------- #

            # get DAG and nodes in the graph
            dag, nodes = generate_dag(indv, stage_name, num_node)
            # get nodes without any predecessor, these will be connected to input node
            without_predecessors = dag.ind_nodes()
            # get nodes without any successor, these will be connected to output node
            without_successors = dag.all_leaves()

            # ----------------------------------------------------------------------------------------------- #

            # --------------------------- Initialize tensforflow graph based on DAG ------------------------- #

            for wop in without_predecessors:
                add_node(wop, ''.join([stage_name, "_input", activation_function_pattern]), ic=ic, oc=oc)

            for n in nodes:
                predecessors = dag.predecessors(n)
                if len(predecessors) == 0:
                    continue
                elif len(predecessors) > 1:
                    first_predecessor = predecessors[0]
                    for prd in range(1, len(predecessors)):
                        t = sum_tensors(first_predecessor, predecessors[prd], activation_function_pattern)
                        first_predecessor = t.name
                    add_node(n, first_predecessor, ic=ic, oc=oc)
                elif predecessors:
                    add_node(n, ''.join([predecessors[0], activation_function_pattern]), ic=ic, oc=oc)

            if len(without_successors) > 1:
                first_successor = without_successors[0]
                for suc in range(1, len(without_successors)):
                    t = sum_tensors(first_successor, without_successors[suc], activation_function_pattern)
                    first_successor = t.name
                add_node(''.join([stage_name, "_output"]), first_successor, ic=ic, oc=oc)
            else:
                add_node(''.join([stage_name, "_output"]),
                         ''.join([without_successors[0], activation_function_pattern]), ic=ic, oc=oc)

            pooling_layer_name = ''.join([stage_name, "_output", activation_function_pattern])
            # ------------------------------------------------------------------------------------------ #

        d_node = apply_pool(tf.get_default_graph().get_tensor_by_name(pooling_layer_name),
                            kernel_height=2, kernel_width=2, stride_size=2)

    shape = d_node.get_shape().as_list()
    flat = tf.reshape(d_node, [-1, shape[1] * shape[2] * shape[3]])
    logits500 = tf.nn.dropout(linear_layer(flat, 500, "logits500"), 0.5, name="dropout")
    logits = linear_layer(logits500, 10, "logits")

    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
    loss_function = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer().minimize(loss_function)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(Y, 1)), tf.float32))

    return X, Y, optimizer, loss_function, accuracy

def evaluateModel(individual):
    score = 0.0
    X, Y, optimizer, loss_function, accuracy = generate_tensorflow_graph(individual, STAGES, NUM_NODES, BITS_INDICES)

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        for epoch in range(TRAINING_EPOCHS):
            for b in range(TOTAL_BATCHES):
                offset = (epoch * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
                batch_x = train_imgs[offset:(offset + BATCH_SIZE), :, :, :]
                batch_y = train_labels[offset:(offset + BATCH_SIZE), :]
                _, c = session.run([optimizer, loss_function], feed_dict={X: batch_x, Y: batch_y})

        score = session.run(accuracy, feed_dict={X: test_imgs, Y: test_labels})
        print('Accuracy: ',score)
    return score,

population_size = 20
num_generations = 3

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("binary", bernoulli.rvs, 0.5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.binary, n=L)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.8)
toolbox.register("select", tools.selRoulette)
toolbox.register("evaluate", evaluateModel)

popl = toolbox.population(n=population_size)
result = algorithms.eaSimple(popl, toolbox, cxpb=0.4, mutpb=0.05, ngen=num_generations, verbose=True)
print(result)

# print top-3 optimal solutions
best_individuals = tools.selBest(popl, k=3)
for bi in best_individuals:
    print(bi)
