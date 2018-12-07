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
    featuremap = tf.layers.conv2d(input, num_filters,5, padding='same')
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
    residual = params.get('residual', False)
    raw = 0
    if residual:
        raw = params.get('raw_img', KeyError('NO raw img provided'))
        raw=tf.layers.conv2d(raw, 1, 1, padding='same')
        raw = tf.layers.max_pooling2d(raw, 7, 1, padding='same')
    rtn = tf.layers.conv2d(input, 1, 1, padding='same')
    return rtn+raw

if __name__=='__main__':
    minstdata = tf.keras.datasets.mnist.load_data('mnist.npz')

    inputplhd = tf.placeholder(tf.float32, shape=(None, 28, 28))
    input = tf.reshape(inputplhd, [-1, 28,28, 1])
    layer1=feature_extraction(input)
    layer2=multi_scale_mapping(layer1)
    layer3= MaxPool(layer2,residual=False)
    layer4=nl_regression(layer3)
    sess=tf.Session()
    print('initalize the variables')
    init_op = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    sess.run(init_op)
    sess.run(init_l)
    lendata = len(minstdata[0][0])
    bs=32
    for j in range(1, 100000):
        i = (j % int(lendata / bs))
        if not i%10:
            print(i)
        data = minstdata[0][0][bs * i:bs * i + bs]
        tlabel = minstdata[0][1][bs * i:bs * i + bs]
        temp = sess.run([layer4], feed_dict={inputplhd: data})

    pass