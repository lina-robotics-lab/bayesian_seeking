import jax
from jax import numpy as jnp

def rbf(x,xp,l=1):
	"""
		We assume x.shape = (n_data,x_dim)
	"""
	return jnp.exp(-jnp.linalg.norm(x-xp,axis=-1) ** 2/(2* l**2 ))

def rbf_KT(X,l=1):
	"""
		Calculate the KT matrix of based on data set X under rbf kernel
	"""
	p2p=X[:,jnp.newaxis]-X

	# calculate the pairwise distance
	r=jnp.linalg.norm(p2p,axis=-1)

	return jnp.exp(-r**2/(2* l**2))

class GaussianProcess:
	"""This is our own implementation of a GaussianProcess model."""
	def __init__(self, mu_0='zero',kernel='rbf',sigma=0.1):
		super(GaussianProcess, self).__init__()
		
		if kernel =='rbf':
			self.l=1
			self.k_0 = lambda x,xp:rbf(x,xp,l=self.l)
			self.KT = lambda X:rbf_KT(X,l=self.l)
		else:
			raise Exception('Kernel {} is not yet supported'.format(kernel))

		if mu_0 == 'zero':
			self.mu_0 = lambda x: 0
		else:
			raise Exception('mu_0 {} is not yet supported'.format(mu_0))			

		self.sigma = sigma

		self.COV_INV = 1/self.sigma**2 # This corresponds to (sigma^2 I + K_T)^{-1}
		self.y_deviation = 0 # This corresponds to y-self.mu_0(x)

		self.x_T = [] # The collected data x so far
		self.y_T = [] # The collected data y so far

	def update(self,X,y):
		assert(len(X)==len(y))
		if len(self.x_T)==0:
			self.x_T=jnp.array(X)
			self.y_T=jnp.array(y).reshape(-1)
		else:
			self.x_T = jnp.vstack([self.x_T,X])
			self.y_T = jnp.vstack([self.y_T,y])

		self.COV_INV = jnp.inv(self.sigma**2 * jnp.eye(len(self.y_T)) + self.KT(self.x_T)) 
		# The heaviest computation lies here.

		self.y_deviation = self.y_T - self.mu_0(x_T)

	def m_T(self,x):
		return self.k_0(self.x_T,x) 
		# pass

	def mu(self,x):
		"""
			Predict the mean value at location x.
		"""
		pass
