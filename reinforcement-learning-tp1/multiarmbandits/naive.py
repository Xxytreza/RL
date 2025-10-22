from typing import List

import numpy as np

from .base import Arm, BanditAlgorithm


class NaiveAlgorithm(BanditAlgorithm):
    
    def __init__(self, T: int = 100, N: int = 20, K: int = 5) -> None:
        super().__init__(T, K)
        
        if N >= T:
            raise ValueError(f"N must be less than T, got N={N}, T={T}")
        if N < K:
            raise ValueError(f"N must be at least K to test each arm once, got N={N}, K={K}")
            
        self.N = N
        self.efficacy = np.zeros(K)
        self.arm_counts = np.zeros(K)
        self.best_arm: int = 0
        
    def run(self, arms: List[Arm]) -> None:
        if len(arms) != self.K:
            raise ValueError(f"Expected {self.K} arms, got {len(arms)}")
            
        self.rewards.clear()
        self.actions.clear()
        self.efficacy.fill(0)
        self.arm_counts.fill(0)
        
        self._train(arms)
        
        self._exploit(arms)
    
    def _train(self, arms: List[Arm]) -> None:
        tests_per_arm = self.N // self.K
        for arm_idx in range(self.K):
            for _ in range(tests_per_arm):
                reward = int(arms[arm_idx].sample())
                self.efficacy[arm_idx] += reward
                self.arm_counts[arm_idx] += 1
                self.rewards.append(reward)
                self.actions.append(arm_idx)
                
    def _exploit(self, arms: List[Arm]) -> None:
        self.best_arm = int(self.efficacy.argmax())
        for _ in range(self.T - self.N):
            reward = int(arms[self.best_arm].sample())
            self.arm_counts[self.best_arm] += 1
            self.rewards.append(reward)
            self.actions.append(self.best_arm)
    
    def get_empirical_means(self) -> np.ndarray:
        means = np.zeros(self.K)
        for i in range(self.K):
            if self.arm_counts[i] > 0:
                means[i] = self.efficacy[i] / self.arm_counts[i]
        return means
    
    def get_best_arm(self) -> int:
        return self.best_arm
