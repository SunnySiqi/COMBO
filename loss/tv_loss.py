from estimation.total_variation import *

class TVLoss(object):
	def __call__(self, output, label, gamma, transition, regularization):
		loss = transition.loss(output, label) - gamma * regularization(output)
		return loss
