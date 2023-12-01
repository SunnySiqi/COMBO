import torch
import numpy as np
import torch.nn.functional as F

from typing import Optional, Callable, Tuple
from torch.distributions import Dirichlet, Categorical

# adapted from: https://github.com/YivanZhang/lio/tree/master/ex/transition-matrix

default_activation = lambda t: F.softmax(t, dim=1)

def diag_matrix(n: int, diagonal: float, off_diagonal: float) -> torch.Tensor:
	return off_diagonal * torch.ones(n, n) + (diagonal - off_diagonal) * torch.eye(n, n)

def owndef_confusion_matrix(v1: torch.Tensor, v2: torch.Tensor,
					 n1: int = None, n2: int = None) -> torch.Tensor:
	if n1 is None:
		n1 = v1.max().item() + 1
	if n2 is None:
		n2 = v2.max().item() + 1
	matrix = torch.zeros(n1, n2).long().to(v1.device)
	pairs, counts = torch.unique(torch.stack((v1, v2)), dim=1, return_counts=True)
	matrix[pairs[0], pairs[1]] = counts
	return matrix

def tv_regularization(num_pairs: int, activation_output: Callable = None):
	activation_output = default_activation if activation_output is None else activation_output

	def reg(t: torch.Tensor):
		p = activation_output(t)
		idx1, idx2 = torch.randint(0, t.shape[0], (2, num_pairs)).to(t.device)
		tv = 0.5 * (p[idx1] - p[idx2]).abs().sum(dim=1).mean()
		return tv

	return reg

class Transition:
	params: Optional[torch.Tensor] = None
	matrix: Optional[torch.Tensor] = None
	update: Callable[[torch.Tensor, torch.Tensor], None] = lambda *_: None
	loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

def dirichlet_transition(device, num_classes, diagonal, off_diagonal, betas):
	init_matrix = diag_matrix(num_classes, diagonal=diagonal, off_diagonal=off_diagonal).to(device)
	return DirichletTransition(init_matrix, betas)

class DirichletTransition(Transition):
	def __init__(self,
				 init_matrix: torch.Tensor,
				 betas: Tuple[float, float],
				 activation_output: Callable = None,
				 ):
		self.concentrations = init_matrix
		self.betas = betas
		self.activation_output = default_activation if activation_output is None else activation_output

	@property
	def params(self):
		return self.concentrations

	@property
	def matrix(self):
		return self.concentrations / self.concentrations.sum(dim=1, keepdim=True)

	def _sample(self):
		return torch.stack([Dirichlet(c).sample() for c in self.concentrations])

	def loss(self, t, y):
		p_z = self.activation_output(t)
		p_y = p_z @ self._sample()
		return F.nll_loss(torch.log(p_y + 1e-32), y)

	def update(self, t, y):
		num_classes = self.concentrations.shape[0]
		# z = t.detach().argmax(dim=1)  # simplified version using argmax
		z = Categorical(probs=self.activation_output(t.detach())).sample()
		m = owndef_confusion_matrix(z, y, n1=num_classes, n2=num_classes)
		self.concentrations *= self.betas[0]  # decay
		self.concentrations += self.betas[1] * m  # update


