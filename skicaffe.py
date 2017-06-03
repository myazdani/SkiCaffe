from sklearn.base import BaseEstimator, TransformerMixin
import sys
import numpy as np
import pandas as pd
import os



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
    def __init__(self,model_prototxt_path, caffe_root, model_trained_path, labels_path = 'default-imagenet-labels',
    mean_path = 'default-imagenet-mean-image', include_labels = True, return_type = 'numpy_array', 
                 include_image_paths = False, gpu_device = 0, verbose = 0):
        self.caffe_root = caffe_root
        self.include_labels = include_labels
        self.labels_path = labels_path
        self.mean_path = mean_path


        self.model_prototxt_path = model_prototxt_path
        self.model_trained_path = model_trained_path
        self.return_type = return_type
        self.include_image_paths = include_image_paths
        
        self.gpu_device = gpu_device
        self.verbosity = verbose

    def fit(self, X=None, y=None):
        sys.path.insert(0, self.caffe_root + 'python')
        global caffe
        import caffe
        caffe.set_mode_gpu()
        caffe.set_device(self.gpu_device)
        print 'caffe imported successfully'
        if self.labels_path == 'default-imagenet-labels':
            self.labels_path = self.caffe_root + 'data/ilsvrc12/synset_words.txt'
        if self.mean_path == 'default-imagenet-mean-image':
            self.mean_path = self.caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
        #global net
        self.net = caffe.Classifier(self.model_prototxt_path, self.model_trained_path, mean = np.load(self.mean_path).mean(1).mean(1), channel_swap = (2,1,0), raw_scale = 255, image_dims = (256,256))

        #print 'net loaded successfully'

        with open(self.labels_path) as f:
            self.labels = f.readlines()

        #print 'labels loaded successfully'
        #self.net.blobs['data'].reshape(50,3,227,227)

        self.layer_sizes = [(k, v.data.shape) for k, v in self.net.blobs.items()]
        self.layer_dict = {}
        for k, v in self.net.blobs.items():
            self.layer_dict[k] = v.data.shape
            
        self.last_layer = self.layer_sizes[-2][0]
        #self.layer_name = last_layer
        return self



    def transform(self, X, layer_name = None):
        if layer_name is None:
            last_layer = self.layer_sizes[-2][0]
            self.layer_name = last_layer
        else:
            self.layer_name = layer_name
        image_paths = X
        features = []
        if self.include_labels:
            predicted_labels = []
            predicted_conf = []
        total_image_paths = len(image_paths)
        image_idx = -1 #For keeping track of transformation progress
        for image_path in image_paths:
            image_idx +=1
            if self.verbosity == 1:
                try: 
                    print '\rTransforming image '+str(image_idx)+' of'+str(total_image_paths)
                except SyntaxError:
                    print('\rTransforming image '+str(image_idx)+' of'+str(total_image_paths))
            input_image = caffe.io.load_image(image_path)
            prediction = self.net.predict([input_image], oversample=False)
            #print os.path.basename(image_path), ' : ' , self.labels[prediction[0].argmax()].strip() , ' (', prediction[0][prediction[0].argmax()] , ')'
            #np.savetxt(writer, net.blobs[layer_name].data[0].reshape(1,-1), fmt='%.8g')
            if self.include_labels:
                pred = prediction[0].argmax()
                predicted_labels.append(self.labels[pred].strip())
                predicted_conf.append(prediction[0][pred])

            f = np.copy(self.net.blobs[self.layer_name].data[0].reshape(1,-1))
            #print f[:2]
            features.append(f)

        if self.return_type == 'pandasDF':
            if len(image_paths) == 1:
                df = pd.DataFrame(np.asarray(features[0]))
            else:
                df = pd.DataFrame(np.asarray(features).squeeze())
            df.columns = [self.layer_name + '.' + str(column) for column in df.columns]
            if self.include_labels:
                df.insert(0,'pred.class', predicted_labels)
                df.insert(1,'pred.conf', predicted_conf)
            if self.include_image_paths:
                image_paths_df = pd.DataFrame({'image_paths': image_paths})
                df = pd.concat([image_paths_df, df], axis=1)
            return df

        if len(image_path) == 1:
            features_np = np.asarray(features[0])
        else:
            features_np = np.asarray(features).squeeze()

        return features_np



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
    try:
        print 'number of features', len(res)
    except SyntaxError:
        print('number of features', len(res))
