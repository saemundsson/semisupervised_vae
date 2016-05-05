import tensorflow as tf
import prettytensor as pt
import numpy as np
import utils

class FullyConnected( object ):

	def __init__( 	self,
					dim_output,
					hidden_layers = [500],
					nonlinearity = tf.nn.softplus,
					l2loss = 0.0,
					name = 'FullyConnected'	):

		self.dim_output = dim_output
		self.hidden_layers = hidden_layers
		self.nonlinearity = nonlinearity
		self.l2loss = l2loss

	def output( self, inputs, phase = pt.Phase.train ):

		inputs = pt.wrap( inputs )
		with pt.defaults_scope( phase = phase, activation_fn = self.nonlinearity, l2loss = self.l2loss ):
			for layer in self.hidden_layers:
				inputs = inputs.fully_connected( layer )

			return inputs.fully_connected( self.dim_output, activation_fn = None )

	

