###
'''
Replication of M1 from http://arxiv.org/abs/1406.5298
Title: Semi-Supervised Learning with Deep Generative Models
Authors: Diederik P. Kingma, Danilo J. Rezende, Shakir Mohamed, Max Welling
Original Implementation (Theano): https://github.com/dpkingma/nips14-ssl
---
Code By: S. Saemundsson
'''
###

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import utils
import numpy as np
import time

from neuralnetworks import FullyConnected
from prettytensor import bookkeeper

class VariationalAutoencoder(object):

	def __init__(   self,
					dim_x, dim_z,
					p_x = 'bernoulli',
					q_z = 'gaussian_marg',
					p_z = 'gaussian_marg',
					hidden_layers_px = [600, 600],
					hidden_layers_qz = [600, 600],
					nonlin_px = tf.nn.softplus,
					nonlin_qz = tf.nn.softplus,
					l2_loss = 0.0   ):

		self.dim_x, self.dim_z = dim_x, dim_z
		self.l2_loss = l2_loss

		self.distributions = {      'p_x':  p_x,            
									'q_z':  q_z,            
									'p_z':  p_z    }

		''' Create Graph '''

		self.G = tf.Graph()

		with self.G.as_default():

			self.x = tf.placeholder( tf.float32, [None, self.dim_x] )

			self.encoder = FullyConnected(      dim_output      = 2 * self.dim_z,
												hidden_layers   = hidden_layers_qz,
												nonlinearity    = nonlin_qz   )

			self.decoder = FullyConnected(      dim_output      = self.dim_x,
												hidden_layers   = hidden_layers_px,
												nonlinearity    = nonlin_px  )

			self._objective()
			self.saver = tf.train.Saver()
			self.session = tf.Session()

	def _draw_sample( self, mu, log_sigma_sq ):

		epsilon = tf.random_normal( ( tf.shape( mu ) ), 0, 1 )
		sample = tf.add( mu, 
				 tf.mul(  
				 tf.exp( 0.5 * log_sigma_sq ), epsilon ) )

		return sample

	def _generate_zx( self, x, phase = pt.Phase.train, reuse = False ):

		with tf.variable_scope('encoder', reuse = reuse):
			encoder_out     = self.encoder.output( x, phase = phase )
		z_mu, z_lsgms   = encoder_out.split( split_dim = 1, num_splits = 2 )
		z_sample        = self._draw_sample( z_mu, z_lsgms )

		return z_sample, z_mu, z_lsgms 

	def _generate_xz( self, z, phase = pt.Phase.train, reuse = False ):

		with tf.variable_scope('decoder', reuse = reuse):
			x_recon_logits = self.decoder.output( z, phase = phase )
		x_recon = tf.nn.sigmoid( x_recon_logits )

		return x_recon, x_recon_logits

	def _objective( self ):

		############
		''' Cost '''
		############

		self.z_sample, self.z_mu, self.z_lsgms = self._generate_zx( self.x )
		self.x_recon, self.x_recon_logits = self._generate_xz( self.z_sample )

		if self.distributions['p_z'] == 'gaussian_marg':

			prior_z = tf.reduce_sum( utils.tf_gaussian_marg( self.z_mu, self.z_lsgms ), 1 )

		if self.distributions['q_z'] == 'gaussian_marg':
			
			post_z = tf.reduce_sum( utils.tf_gaussian_ent( self.z_lsgms ), 1 )

		if self.distributions['p_x'] == 'bernoulli':

			self.log_lik = - tf.reduce_sum( utils.tf_binary_xentropy( self.x, self.x_recon ), 1 )

		l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

		self.cost = tf.reduce_mean( post_z - prior_z - self.log_lik ) + self.l2_loss * l2

		##################
		''' Evaluation '''
		##################

		self.z_sample_eval, _, _ = self._generate_zx( self.x, phase = pt.Phase.test, reuse = True )
		self.x_recon_eval, _ = self._generate_xz( self.z_sample_eval, phase = pt.Phase.test, reuse = True )

		self.eval_log_lik = - tf.reduce_mean( tf.reduce_sum( utils.tf_binary_xentropy( self.x, self.x_recon_eval ), 1 ) )


	def train(      self, x, x_valid,
					epochs, num_batches,
					print_every = 1,
					learning_rate = 3e-4,
					beta1 = 0.9,
					beta2 = 0.999,
					seed = 31415,
					stop_iter = 100,
					save_path = None,
					load_path = None,
					draw_img = 1    ):

		self.num_examples = x.shape[0]
		self.num_batches = num_batches

		assert self.num_examples % self.num_batches == 0, '#Examples % #Batches != 0'

		self.batch_size = self.num_examples // self.num_batches

		''' Session and Summary '''
		if save_path is None: 
			self.save_path = 'checkpoints/model_VAE_{}-{}_{}.cpkt'.format(learning_rate,self.batch_size,time.time())
		else:
			self.save_path = save_path

		np.random.seed(seed)
		tf.set_random_seed(seed)

		with self.G.as_default():

			self.optimiser = tf.train.AdamOptimizer( learning_rate = learning_rate, beta1 = beta1, beta2 = beta2 )
			self.train_op = self.optimiser.minimize( self.cost )
			init = tf.initialize_all_variables()
			self._test_vars = None

		with self.session as sess:

			sess.run(init)
			if load_path == 'default': self.saver.restore( sess, self.save_path )
			elif load_path is not None: self.saver.restore( sess, load_path )	

			training_cost = 0.
			best_eval_log_lik = - np.inf
			stop_counter = 0

			for epoch in range(epochs):

				''' Shuffle Data '''
				np.random.shuffle( x )

				''' Training '''
				
				for x_batch in utils.feed_numpy( self.batch_size, x ):

					training_result = sess.run( [self.train_op, self.cost],
											feed_dict = { self.x: x_batch } )

					training_cost = training_result[1]

				''' Evaluation '''

				stop_counter += 1

				if epoch % print_every == 0:

					test_vars = tf.get_collection(bookkeeper.GraphKeys.TEST_VARIABLES)
					if test_vars:
						if test_vars != self._test_vars:
							self._test_vars = list(test_vars)
							self._test_var_init_op = tf.initialize_variables(test_vars)
						self._test_var_init_op.run()


					eval_log_lik, x_recon_eval = \
						sess.run( [self.eval_log_lik, self.x_recon_eval],
									feed_dict = { self.x: x_valid } )

					if eval_log_lik > best_eval_log_lik:

						best_eval_log_lik = eval_log_lik
						self.saver.save( sess, self.save_path )
						stop_counter = 0

					utils.print_metrics( 	epoch+1,
											['Training', 'cost', training_cost],
											['Validation', 'log-likelihood', eval_log_lik] )

					if draw_img > 0 and epoch % draw_img == 0:

						import matplotlib
						matplotlib.use('Agg')
						import pylab
						import seaborn as sns

						five_random = np.random.random_integers(x_valid.shape[0], size = 5)
						x_sample = x_valid[five_random]
						x_recon_sample = x_recon_eval[five_random]

						sns.set_style('white')
						f, axes = pylab.subplots(5, 2, figsize=(8,12))
						for i,row in enumerate(axes):

							row[0].imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1)
							im = row[1].imshow(x_recon_sample[i].reshape(28, 28), vmin=0, vmax=1, 
								cmap=sns.light_palette((1.0, 0.4980, 0.0549), input="rgb", as_cmap=True))

							pylab.setp([a.get_xticklabels() for a in row], visible=False)
							pylab.setp([a.get_yticklabels() for a in row], visible=False)
	
						f.subplots_adjust(left=0.0, right=0.9, bottom=0.0, top=1.0)
						cbar_ax = f.add_axes([0.9, 0.1, 0.04, 0.8])
						f.colorbar(im, cax=cbar_ax, use_gridspec=True)
		
						pylab.tight_layout()
						pylab.savefig('img/recon-'+str(epoch)+'.png', format='png')
						pylab.clf()
						pylab.close('all')

				if stop_counter >= stop_iter:
					print('Stopping VAE training')
					print('No change in validation log-likelihood for {} iterations'.format(stop_iter))
					print('Best validation log-likelihood: {}'.format(best_eval_log_lik))
					print('Model saved in {}'.format(self.save_path))
					break

	def encode( self, x, sample = False ):

		if sample:
			return self.session.run( [self.z_sample, self.z_mu, self.z_lsgms], feed_dict = { self.x: x } )
		else:
			return self.session.run( [self.z_mu, self.z_lsgms], feed_dict = { self.x: x } )

	def decode( self, z ):

		return self.session.run( 	[self.x_recon],
									feed_dict = { self.z_sample: z } )