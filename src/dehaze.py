import tensorflow as tf

def BreLU(input):
    # you can also define the func by using masks
    input=tf.nn.relu(input)
    input=tf.math.minimum(input,1)
    return input

def feature_extraction(input, **params):
    '''

    :param input: should be in shape [batch,H,W,C]
    :param params: a dictionary contains the params
    :return:maxouted feature map
    '''
    num_filters = params.get('num_filters', 16)
    # assume only one layer
    featuremap = tf.layers.conv2d(input, num_filters, padding='same')
    feature_map_maxout = tf.contrib.layers.maxout(featuremap, 4)
    # output should be in 4 channels
    return feature_map_maxout


def multi_scale_mapping(input, **params):
    '''

    :param input: maxouted feature map
    :param params: params: a dictionary contains the params
    :return:
    '''
    temp_conv = []
    num_filters = params.get('num_filters', 16)
    filter_size_list = params.get('filter_size_list', [3, 5, 7])
    for filter_size in filter_size_list:
        temp_conv.append(
            tf.layers.conv2d(
                input, num_filters,
                filter_size,
                padding='same'
            ))
    rtn = tf.concat(temp_conv, axis=-1)
    return rtn


def MaxPool(input, **params):
    '''

    :param input:
    :param params:
    :return:
    '''
    rtn = tf.layers.max_pooling2d(input, 7, 1, padding='same')
    return rtn


def nl_regression(input, **params):
    rtn = tf.layers.conv2d(input, 1, 1, padding='same')
    return rtn
