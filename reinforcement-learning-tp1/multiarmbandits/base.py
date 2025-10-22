from abc import ABC, abstractmethod
from typing import List, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt


@runtime_checkable
class Arm(Protocol):
    mean: float
    
    def sample(self) -> bool:
        ...


class ArmBernoulli:
    
    def __init__(self, p: float, random_state: int = 0) -> None:
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0, 1], got {p}")
            
        self.mean = p
        self.local_random = np.random.RandomState(random_state)
        
    def sample(self) -> bool:
        return self.local_random.rand() < self.mean


class BanditAlgorithm(ABC):
    
    def __init__(self, T: int, K: int) -> None:
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
            
        self.T = T
        self.K = K
        self.rewards: List[int] = []
        self.actions: List[int] = []
        
    @abstractmethod
    def run(self, arms: List[Arm]) -> None:
        ...
        
    def calculate_regret_evolution(self, optimal_mean: float) -> npt.NDArray[np.float64]:
        if not self.rewards:
            raise ValueError("No rewards recorded. Run the algorithm first.")
            
        cumulative_regret = np.zeros(len(self.rewards))
        optimal_reward_so_far = 0.0
        actual_reward_so_far = 0.0
        
        for t, reward in enumerate(self.rewards):
            optimal_reward_so_far += optimal_mean
            actual_reward_so_far += reward
            cumulative_regret[t] = optimal_reward_so_far - actual_reward_so_far
            
        return cumulative_regret
        
    def get_arm_selection_counts(self) -> npt.NDArray[np.int64]:
        counts = np.zeros(self.K, dtype=np.int64)
        for action in self.actions:
            counts[action] += 1
        return counts
        
    def get_final_regret(self, optimal_mean: float) -> float:
        regret_evolution = self.calculate_regret_evolution(optimal_mean)
        return float(regret_evolution[-1])


def create_bernoulli_arms(means: npt.NDArray[np.float64], 
                         random_state: int = 0) -> List[ArmBernoulli]:
    return [ArmBernoulli(mean, random_state + i) for i, mean in enumerate(means)]
