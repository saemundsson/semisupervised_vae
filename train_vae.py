from vae import VariationalAutoencoder
import numpy as np
import data.mnist as mnist #https://github.com/dpkingma/nips14-ssl

if __name__ == '__main__':

    #############################
    ''' Experiment Parameters '''
    #############################

    num_batches = 1000      #Number of minibatches in a single epoch
    dim_z = 50              #Dimensionality of latent variable (z)
    epochs = 3001           #Number of epochs through the full dataset
    learning_rate = 3e-4    #Learning rate of ADAM
    l2_loss = 1e-6          #L2 Regularisation weight
    seed = 31415            #Seed for RNG

    #Neural Networks parameterising p(x|z), q(z|x)
    hidden_layers_px = [ 600, 600 ]
    hidden_layers_qz = [ 600, 600 ]

    ####################
    ''' Load Dataset '''
    ####################

    mnist_path = 'mnist/mnist_28.pkl.gz'
    #Uses anglpy module from original paper (linked at top) to load the dataset
    train_x, train_y, valid_x, valid_y, test_x, test_y = mnist.load_numpy(mnist_path, binarize_y=True)

    x_train, y_train = train_x.T, train_y.T
    x_valid, y_valid = valid_x.T, valid_y.T
    x_test, y_test = test_x.T, test_y.T

    dim_x = x_train.shape[1]
    dim_y = y_train.shape[1]

    ######################################
    ''' Train Variational Auto-Encoder '''
    ######################################

    VAE = VariationalAutoencoder(   dim_x = dim_x, 
                                    dim_z = dim_z,
                                    hidden_layers_px = hidden_layers_px,
                                    hidden_layers_qz = hidden_layers_qz,
                                    l2_loss = l2_loss )

    #draw_img uses pylab and seaborn to draw images of original vs. reconstruction 
    #every n iterations (set to 0 to disable)

    VAE.train(  x = x_train, x_valid = x_valid, epochs = epochs, num_batches = num_batches,
                learning_rate = learning_rate, seed = seed, stop_iter = 30, print_every = 10, draw_img = 0 )

    