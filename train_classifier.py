from genclass import GenerativeClassifier
from vae import VariationalAutoencoder
import numpy as np
import data.mnist as mnist #https://github.com/dpkingma/nips14-ssl

def encode_dataset( model_path, min_std = 0.0 ):

    VAE = VariationalAutoencoder( dim_x = 28*28, dim_z = 50 ) #Should be consistent with model being loaded
    with VAE.session:
        VAE.saver.restore( VAE.session, VAE_model_path )

        enc_x_lab_mean, enc_x_lab_var = VAE.encode( x_lab )
        enc_x_ulab_mean, enc_x_ulab_var = VAE.encode( x_ulab )
        enc_x_valid_mean, enc_x_valid_var = VAE.encode( x_valid )
        enc_x_test_mean, enc_x_test_var = VAE.encode( x_test )

        id_x_keep = np.std( enc_x_ulab_mean, axis = 0 ) > min_std

        enc_x_lab_mean, enc_x_lab_var = enc_x_lab_mean[ :, id_x_keep ], enc_x_lab_var[ :, id_x_keep ]
        enc_x_ulab_mean, enc_x_ulab_var = enc_x_ulab_mean[ :, id_x_keep ], enc_x_ulab_var[ :, id_x_keep ]
        enc_x_valid_mean, enc_x_valid_var = enc_x_valid_mean[ :, id_x_keep ], enc_x_valid_var[ :, id_x_keep ]
        enc_x_test_mean, enc_x_test_var = enc_x_test_mean[ :, id_x_keep ], enc_x_test_var[ :, id_x_keep ]

        data_lab = np.hstack( [ enc_x_lab_mean, enc_x_lab_var ] )
        data_ulab = np.hstack( [ enc_x_ulab_mean, enc_x_ulab_var ] )
        data_valid = np.hstack( [enc_x_valid_mean, enc_x_valid_var] )
        data_test = np.hstack( [enc_x_test_mean, enc_x_test_var] )

    return data_lab, data_ulab, data_valid, data_test

if __name__ == '__main__':
    
    #############################
    ''' Experiment Parameters '''
    #############################

    num_lab = 100           #Number of labelled examples (total)
    num_batches = 100       #Number of minibatches in a single epoch
    dim_z = 50              #Dimensionality of latent variable (z)
    epochs = 1001           #Number of epochs through the full dataset
    learning_rate = 3e-4    #Learning rate of ADAM
    alpha = 0.1             #Discriminatory factor (see equation (9) of http://arxiv.org/pdf/1406.5298v2.pdf)
    seed = 31415            #Seed for RNG

    #Neural Networks parameterising p(x|z,y), q(z|x,y) and q(y|x)
    hidden_layers_px = [ 500 ]
    hidden_layers_qz = [ 500 ]
    hidden_layers_qy = [ 500 ]

    ####################
    ''' Load Dataset '''
    ####################

    mnist_path = 'mnist/mnist_28.pkl.gz'
    #Uses anglpy module from original paper (linked at top) to split the dataset for semi-supervised training
    train_x, train_y, valid_x, valid_y, test_x, test_y = mnist.load_numpy_split(mnist_path, binarize_y=True) 
    x_l, y_l, x_u, y_u = mnist.create_semisupervised(train_x, train_y, num_lab)

    x_lab, y_lab = x_l.T, y_l.T
    x_ulab, y_ulab = x_u.T, y_u.T
    x_valid, y_valid = valid_x.T, valid_y.T
    x_test, y_test = test_x.T, test_y.T

    ################
    ''' Load VAE '''
    ################

    VAE_model_path = 'models/VAE_600-600-0.0003-50.cpkt'
    min_std = 0.1 #Dimensions with std < min_std are removed before training with GC

    data_lab, data_ulab, data_valid, data_test = encode_dataset( VAE_model_path, min_std )

    dim_x = data_lab.shape[1] / 2
    dim_y = y_lab.shape[1]
    num_examples = data_lab.shape[0] + data_ulab.shape[0]

    ###################################
    ''' Train Generative Classifier '''
    ###################################

    GC = GenerativeClassifier(  dim_x, dim_z, dim_y,
                                num_examples, num_lab, num_batches,
                                hidden_layers_px    = hidden_layers_px, 
                                hidden_layers_qz    = hidden_layers_qz, 
                                hidden_layers_qy    = hidden_layers_qy,
                                alpha               = alpha )

    GC.train(   x_labelled      = data_lab, y = y_lab, x_unlabelled = data_ulab,
                x_valid         = data_valid, y_valid = y_valid,
                epochs          = epochs, 
                learning_rate   = learning_rate,
                seed            = seed,
                print_every     = 10,
                load_path       = None )


    ############################
    ''' Evaluate on Test Set '''
    ############################

    GC.predict_labels( data_test, y_test )