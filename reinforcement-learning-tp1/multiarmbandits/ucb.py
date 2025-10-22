from typing import List

import numpy as np
import numpy.typing as npt

from .base import Arm, BanditAlgorithm


class UCBAlgorithm(BanditAlgorithm):
    
    def __init__(self, T: int = 100, K: int = 5) -> None:
        super().__init__(T, K)
        self.arm_counts = np.zeros(K, dtype=np.int64)
        self.arm_rewards = np.zeros(K)
        
    def run(self, arms: List[Arm]) -> None:
        if len(arms) != self.K:
            raise ValueError(f"Expected {self.K} arms, got {len(arms)}")
            
        self.rewards.clear()
        self.actions.clear()
        self.arm_counts.fill(0)
        self.arm_rewards.fill(0)
        
        for t in range(1, self.T + 1):
            ucb_values = np.array([self._get_ucb_value(arm, t) for arm in range(self.K)])
            chosen_arm = int(np.argmax(ucb_values))
            reward = int(arms[chosen_arm].sample())

            self.arm_counts[chosen_arm] += 1
            self.arm_rewards[chosen_arm] += reward
            self.rewards.append(reward)
            self.actions.append(chosen_arm)
    
    def _get_empirical_mean(self, arm: int) -> float:
        if self.arm_counts[arm] == 0:
            return 0.0
        return self.arm_rewards[arm] / self.arm_counts[arm]
    
    def _get_confidence_bound(self, arm: int, t: int) -> float:
        if self.arm_counts[arm] == 0:
            return np.inf
        return np.sqrt(2.0 * np.log(t) / self.arm_counts[arm])
    
    def _get_ucb_value(self, arm: int, t: int) -> float:
        return self._get_empirical_mean(arm) + self._get_confidence_bound(arm, t)
    
    def get_empirical_means(self) -> npt.NDArray[np.float64]:
        means = np.zeros(self.K)
        for arm in range(self.K):
            means[arm] = self._get_empirical_mean(arm)
        return means
    
    def get_confidence_bounds(self, t: int) -> npt.NDArray[np.float64]:
        bounds = np.zeros(self.K)
        for arm in range(self.K):
            bounds[arm] = self._get_confidence_bound(arm, t)
        return bounds


class UCBAlgorithmWithHistory(UCBAlgorithm):
    
    def __init__(self, T: int = 100, K: int = 5) -> None:
        super().__init__(T, K)
        self.empirical_means_history: List[npt.NDArray[np.float64]] = []
        self.confidence_bounds_history: List[npt.NDArray[np.float64]] = []
        self.chosen_arms: List[int] = []
        
    def run(self, arms: List[Arm]) -> None:
        if len(arms) != self.K:
            raise ValueError(f"Expected {self.K} arms, got {len(arms)}")
            
        self.rewards.clear()
        self.actions.clear()
        self.arm_counts.fill(0)
        self.arm_rewards.fill(0)
        self.empirical_means_history.clear()
        self.confidence_bounds_history.clear()
        self.chosen_arms.clear()
        
        for t in range(1, self.T + 1):
            empirical_means = self.get_empirical_means()
            confidence_bounds = self.get_confidence_bounds(t)
            self.empirical_means_history.append(empirical_means.copy())
            self.confidence_bounds_history.append(confidence_bounds.copy())
            
            ucb_values = empirical_means + confidence_bounds
            chosen_arm = int(np.argmax(ucb_values))
            self.chosen_arms.append(chosen_arm)
            
            reward = int(arms[chosen_arm].sample())
            self.arm_counts[chosen_arm] += 1
            self.arm_rewards[chosen_arm] += reward
            self.rewards.append(reward)
            self.actions.append(chosen_arm)
