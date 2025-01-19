import torch as t
from collections import deque


type TimeStep = int
type Loss = float

# Intended to be an implementation of importance sampling as described in https://arxiv.org/pdf/2102.09672 
class TimeStepSampler:
    def __init__(self, num_time_steps: int, use_importance_sampling: bool=False, threshold_for_importance_sampling: int=10):
        self.losses_map: dict[TimeStep, deque[Loss]] = {ts: deque([], maxlen=threshold_for_importance_sampling) for ts in range(0,num_time_steps)}
        self.use_importance_sampling: bool = use_importance_sampling
        self.is_threshold_for_importance_sampling_met: bool = False
        self.num_time_steps: int = num_time_steps
    # Dont bother calling this if not using importance sampling
    def update_losses(self, time_steps: t.Tensor, losses: t.Tensor) -> None:
        if not self.use_importance_sampling:
            raise Exception("update_losses should not be called if not using importance sampling")
        for ts, loss in zip(time_steps, losses):
            self.losses_map[ts].append(float(loss))
        if not self.is_threshold_for_importance_sampling_met:
            self._check_threshold_for_importance_sampling_met()


    def _check_threshold_for_importance_sampling_met(self) -> None:
        if all([len(losses) >= 10 for losses in self.losses_map.values()]):
            self.is_threshold_for_importance_sampling_met = True


    def sample_time_steps(self, time_step_dim: int) -> t.Tensor:
        if not (self.use_importance_sampling and self.is_threshold_for_importance_sampling_met):
            return t.randint(0, self.num_time_steps, (time_step_dim,))
        weights = t.tensor([(sum(self.losses_map[ts]) / len(self.losses_map[ts])) for ts in range(len(self.losses_map))])
        
        weights = t.sqrt(weights) / weights.sum()  
        sampled_indices = t.multinomial(weights, time_step_dim, replacement=True)
        
        return sampled_indices

        


