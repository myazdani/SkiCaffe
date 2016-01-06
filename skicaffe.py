from sklearn.base import BaseEstimator, TransformerMixin
import sys
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

class SkiCaffe(BaseEstimator, TransformerMixin):
    '''Compute the features from a layer of a pre-trained neural network from
    the Caffe Model Zoo

    Parameters
    ----------
    caffe_root: path destination of where caffe is located

    Attributes
    ----------
    layer_names: list

    Examples
    --------
    from skicaffe import SkiCaffe
    caffe_features = SkiCaffe(caffe_root = '/usr/local/src/caffe/caffe-master/')

    caffe_features.fit(model_prototxt = 'deploy.prototxt',
                       model_trained = 'bvlc_googlenet.caffemodel',
                       label = 'synset_words.txt',
                       mean_path = 'ilsvrc_2012_mean.npy')

    image_feature = caffe_features.transform(layer_name = 'pool5/7x7_s1', image_paths = 'image.jpg')
    '''
    def __init__(self, caffe_root):
        self.caffe_root = caffe_root
        sys.path.insert(0, caffe_root + 'python')
        global caffe
        import caffe
        caffe.set_mode_gpu()
        print 'caffe imported successfully'

    def fit(self, model_prototxt_path, model_trained_path, labels_path = 'default-imagenet-labels', mean_path = 'default-imagenet-mean-image'):
        if labels_path == 'default-imagenet-labels':
            labels_path = self.caffe_root + 'data/ilsvrc12/synset_words.txt'
        if mean_path == 'default-imagenet-mean-image':
            mean_path = self.caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
        #global net
        self.net = caffe.Classifier(model_prototxt_path, model_trained_path, mean = np.load(mean_path).mean(1).mean(1), channel_swap = (2,1,0), raw_scale = 255, image_dims = (256,256))

        #print 'net loaded successfully'

        with open(labels_path) as f:
            self.labels = f.readlines()

        #print 'labels loaded successfully'
        #self.net.blobs['data'].reshape(50,3,227,227)

        self.layer_sizes = [(k, v.data.shape) for k, v in self.net.blobs.items()]
        self.layer_dict = {}
        for k, v in self.net.blobs.items():
            self.layer_dict[k] = v.data.shape



    def transform(self, layer_name, image_paths, return_type = 'numpy_array'):
        features = []
        for image_path in image_paths:
            input_image = caffe.io.load_image(image_path)
            prediction = self.net.predict([input_image], oversample=False)
            #print os.path.basename(image_path), ' : ' , self.labels[prediction[0].argmax()].strip() , ' (', prediction[0][prediction[0].argmax()] , ')'
            #np.savetxt(writer, net.blobs[layer_name].data[0].reshape(1,-1), fmt='%.8g')
            f = np.copy(self.net.blobs[layer_name].data[0].reshape(1,-1))
            #print f[:2]
            features.append(f)

        if len(image_paths) == 1:
            df = pd.DataFrame(np.asarray(features[0]))
        else:
            df = pd.DataFrame(np.asarray(features).squeeze())

        if return_type == 'pandasDF':
            df.columns = [layer_name + '.' + str(column) for column in df.columns]
            df.index = image_paths
            return df
        return features

    # take an array of shape (n, height, width) or (n, height, width, channels)
    # and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    def vis_square(self, data, padsize=1, padval=0):
        data -= data.min()
        data /= data.max()

        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

        plt.imshow(data)

    def vis_img_feature(self, layer_name, image_path, num_filters = 'all'):
        filter_shape = self.layer_dict[layer_name][1:]
        if num_filters == 'all':
            max_filter = filter_shape[0]
        else:
            max_filter = num_filters
        input_image = caffe.io.load_image(image_path)
        prediction = self.net.predict([input_image], oversample=False)
        feature = np.copy(self.net.blobs[layer_name].data[0].reshape(1,-1))
        feat = feature.reshape(filter_shape)
        print feat.shape
        self.vis_square(feat[:max_filter,:,:], padval=1)



if __name__ == "__main__":
    caffe_features = SkiCaffe('/usr/local/src/caffe/caffe-master/')
    caffe_root = '/usr/local/src/caffe/caffe-master/'
    # Model prototxt file
    model_prototxt = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
    # Model caffemodel file
    model_trained = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
    # File containing the class labels
    imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'
    # Path to the mean image (used for input processing
    mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'

    caffe_features.fit(model_prototxt, model_trained, imagenet_labels, mean_path)
    image_path = '/usr/local/src/caffe/caffe-master/examples/images/cat.jpg'
    res = caffe_features.transform(layer_name = 'pool5/7x7_s1', image_paths = [image_path])
    print 'number of features', len(res)
