import torch as t
from typing import Iterable, Callable, Tuple
from torch.multiprocessing import Pool

from evostrat import Individual


class Population:
    """
    A parameterized distribution over individuals.
    Meant to be optimized with `torch.optim` optimizers, as follows:
    pop = PopulationImpl(...)
    optim = torch.optim.Adam(pop.parameters())
    for i in range(N):
        optim.zero_grads()
        pop.fitness_grads(n_samples=200)
        optim.step()
    """

    def parameters(self) -> Iterable[t.Tensor]:
        """
        :return: The parameters of this population distribution.
        """

        raise NotImplementedError

    def sample(self, n) -> Iterable[Tuple[Individual, t.Tensor]]:
        """
        Sample n individuals and compute their log probabilities. The log probability computation MUST be differentiable.
        :param n: How many individuals to sample
        :return: n individuals and their log probability of being sampled: [(ind_1, log_prob_1), ..., (ind_n, log_prob_n)]
        """
        raise NotImplementedError

    def fitness_grads(
            self,
            n_samples: int,
            pool: Pool = None,
            fitness_shaping_fn: Callable[[Iterable[float]], Iterable[float]] = lambda x: x
    ):
        """
        Computes the (approximate) gradients of the expected fitness of the population.
        Uses torch autodiff to compute the gradients. The Individual.fitness does NOT need to be differentiable,
        but the log probability computations in Population.sample MUST be.
        :param n_samples: How many individuals to sample to approximate the gradient
        :param pool: Optional process pool to use when computing the fitness of the sampled individuals.
        :param fitness_shaping_fn: Optional function to modify the fitness, e.g. normalization, etc. Input is a list of n raw fitness floats. Output must also be n floats.
        :return: A (n,) tensor containing the raw fitness (before fitness_shaping_fn) for the n individuals.
        """

        samples = self.sample(n_samples)  # Generator
        individuals = []
        grads = []
        for individual, log_prob in samples:  # Compute gradients one at a time so only one log prob computational graph needs to be kept in memory at a time.
            assert log_prob.ndim == 0 and log_prob.isfinite() and log_prob.grad_fn is not None, "log_probs must be differentiable finite scalars"
            individuals.append(individual)
            grads.append([g.cpu() for g in t.autograd.grad(log_prob, self.parameters())])

        if pool is not None:
            results = pool.map(_fitness_fn_no_grad, individuals)
        else:
            raw_fitness, data = list(map(_fitness_fn_no_grad, individuals))
        raw_fitness = []
        data = []
        timesteps = 0
        for r in results:
            raw_fitness.append(r[0])
            data.extend(r[1])
            timesteps += r[2]
        # raw_fitness, data = results
        fitness = fitness_shaping_fn(raw_fitness)

        for i, p in enumerate(self.parameters()):
            p.grad = -t.mean(t.stack([ind_fitness * grad[i] for grad, ind_fitness in zip(grads, fitness)]), dim=0).to(p.device)
        # print(f"data is {data[:10]}")
        # print(f"total steps: {timesteps}")
        # input()
        return t.tensor(raw_fitness), data, timesteps
    
def _fitness_fn_no_grad(ind: Individual):
    # print("vjnevjknfvkjnrjkvnjknfvnjfknvfnkj")
    with t.no_grad():
        return ind.fitness()
    
    
    
    
    
    
from typing import Iterable, Dict, Callable, Union

import torch as t
import torch.distributions as d

from evostrat import Individual


class NormalPopulation(Population):
    """
    A distribution over individuals whose parameters are sampled from normal distributions
    """

    def __init__(self,
                 individual_parameter_shapes: Dict[str, t.Size],
                 individual_constructor: Callable[[Dict[str, t.Tensor]], Individual],
                 std: Union[float, str],
                 mirror_sampling: bool = True,
                 device='cpu'
                 ):
        """
        A distribution over individuals whose parameters are sampled from normal distributions
        The individuals are sampled by sampling their parameters from independent normal distributions and then calling individual_constructor with the sampled parameters.
        :param individual_parameter_shapes: The shapes of the parameters of an individual.
        :param individual_constructor: A function that constructs an individual from parameters (with shapes equal to individual_parameter_shapes).
        :param std: The standard deviation of the normal distributions.
        If a float, it is treated as a constant hyper-parameter. Equivalent to OpenAI ES [1].
        If it's a str it must be either 'shared' or 'diagonal':
            If it's 'shared' all parameters will share a single learned std deviation.
            if it's 'diagonal' each parameter will have their own learned std deviation, similar to PEPG [2].
        :param mirror_sampling: Whether or not individuals are sampled in pairs symmetrically around the mean. See [1].
        [1] - Salimans, Tim, et al. "Evolution strategies as a scalable alternative to reinforcement learning." arXiv preprint arXiv:1703.03864 (2017).
        [2] - Sehnke, Frank, et al. "Parameter-exploring policy gradients." Neural Networks 23.4 (2010): 551-559.
        """
        assert type(std) in {float, str}, "std must be a float or str"
        if type(std) == float:
            assert std > 0.0, "std must be greater than 0"
            self.param_logstds = {k: t.log(t.scalar_tensor(std, device=device)) for k in individual_parameter_shapes.keys()}
        if type(std) == str:
            assert std in {'shared', 'diagonal'}, "std must be 'shared' or 'diagonal'"
            if std == 'shared':
                self.shared_log_std = t.scalar_tensor(0.0, requires_grad=True, device=device)
                self.param_logstds = {k: self.shared_log_std for k in individual_parameter_shapes.keys()}
            else:
                self.param_logstds = {k: t.zeros(shape, requires_grad=True, device=device) for k, shape in individual_parameter_shapes.items()}

        self.std = std
        self.param_means = {k: t.zeros(shape, requires_grad=True, device=device) for k, shape in individual_parameter_shapes.items()}
        self.constructor = individual_constructor
        self.mirror_sampling = mirror_sampling

    def parameters(self) -> Iterable[t.Tensor]:
        if type(self.std) == float:
            std_params = []
        else:
            if self.std == 'shared':
                std_params = [self.shared_log_std]
            else:
                std_params = list(self.param_logstds.values())

        mean_params = list(self.param_means.values())
        return mean_params + std_params

    def sample(self, n) -> Iterable[Individual]:
        assert not self.mirror_sampling or n % 2 == 0, "if mirror_sampling is true, n must be an even number"

        n_samples = n // 2 if self.mirror_sampling else n

        for _ in range(n_samples):
            noise = {k: d.Normal(loc=t.zeros_like(v), scale=t.exp(self.param_logstds[k])).sample() for k, v in self.param_means.items()}
            yield (
                self.constructor({k: self.param_means[k] + n for k, n in noise.items()}),
                sum([d.Normal(self.param_means[k], scale=t.exp(self.param_logstds[k])).log_prob((self.param_means[k] + n).detach()).sum() for k, n in noise.items()])
            )
            if self.mirror_sampling:
                yield (
                    self.constructor({k: self.param_means[k] - n for k, n in noise.items()}),
                    sum([d.Normal(self.param_means[k], scale=t.exp(self.param_logstds[k])).log_prob((self.param_means[k] - n).detach()).sum() for k, n in noise.items()])
                )