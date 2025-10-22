from typing import List

import numpy as np
import numpy.typing as npt
from scipy import stats

from .base import Arm, BanditAlgorithm


class ThompsonSamplingAlgorithm(BanditAlgorithm):
   
    def __init__(self, T: int = 100, K: int = 5, 
                 alpha_prior: float = 1.0, beta_prior: float = 1.0) -> None:
        super().__init__(T, K)
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.alpha = np.full(K, alpha_prior)
        self.beta = np.full(K, beta_prior)
        
    def run(self, arms: List[Arm]) -> None:
        if len(arms) != self.K:
            raise ValueError(f"Expected {self.K} arms, got {len(arms)}")
            
        self.rewards.clear()
        self.actions.clear()
        self.alpha.fill(self.alpha_prior)
        self.beta.fill(self.beta_prior)
        
        for _ in range(self.T):
            theta_samples = np.array([
                np.random.beta(self.alpha[arm], self.beta[arm])
                for arm in range(self.K)
            ])
            
            chosen_arm = int(np.argmax(theta_samples))
            
            reward = int(arms[chosen_arm].sample())
            
            if reward == 1:
                self.alpha[chosen_arm] += 1
            else:
                self.beta[chosen_arm] += 1
                
            self.rewards.append(reward)
            self.actions.append(chosen_arm)
    
    def get_posterior_means(self) -> npt.NDArray[np.float64]:
        return self.alpha / (self.alpha + self.beta)
    
    def get_posterior_variance(self) -> npt.NDArray[np.float64]:
        return (self.alpha * self.beta) / ((self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1))
    
    def get_credible_interval(self, confidence: float = 0.95) -> npt.NDArray[np.float64]:
        alpha = 1 - confidence
        intervals = np.zeros((self.K, 2))
        
        for arm in range(self.K):
            intervals[arm, 0] = stats.beta.ppf(alpha / 2, self.alpha[arm], self.beta[arm])
            intervals[arm, 1] = stats.beta.ppf(1 - alpha / 2, self.alpha[arm], self.beta[arm])
            
        return intervals
    
    def sample_posterior(self, n_samples: int = 1000) -> npt.NDArray[np.float64]:
        samples = np.zeros((n_samples, self.K))
        for arm in range(self.K):
            samples[:, arm] = np.random.beta(self.alpha[arm], self.beta[arm], n_samples)
        return samples


class ThompsonSamplingAlgorithmWithHistory(ThompsonSamplingAlgorithm):
    
    def __init__(self, T: int = 100, K: int = 5,
                 alpha_prior: float = 1.0, beta_prior: float = 1.0) -> None:
        super().__init__(T, K, alpha_prior, beta_prior)
        self.alpha_history: List[npt.NDArray[np.float64]] = []
        self.beta_history: List[npt.NDArray[np.float64]] = []
        self.chosen_arms: List[int] = []
        
    def run(self, arms: List[Arm]) -> None:
        if len(arms) != self.K:
            raise ValueError(f"Expected {self.K} arms, got {len(arms)}")
            
        self.rewards.clear()
        self.actions.clear()
        self.alpha.fill(self.alpha_prior)
        self.beta.fill(self.beta_prior)
        self.alpha_history.clear()
        self.beta_history.clear()
        self.chosen_arms.clear()
        
        for _ in range(self.T):
            self.alpha_history.append(self.alpha.copy())
            self.beta_history.append(self.beta.copy())
            
            theta_samples = np.array([
                np.random.beta(self.alpha[arm], self.beta[arm])
                for arm in range(self.K)
            ])
            
            chosen_arm = int(np.argmax(theta_samples))
            self.chosen_arms.append(chosen_arm)
            
            reward = int(arms[chosen_arm].sample())
            
            if reward == 1:
                self.alpha[chosen_arm] += 1
            else:
                self.beta[chosen_arm] += 1
                
            self.rewards.append(reward)
            self.actions.append(chosen_arm)
    
    def get_distribution_at_time(self, t: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        if t < 0 or t >= len(self.alpha_history):
            raise ValueError(f"Time step {t} out of range [0, {len(self.alpha_history)-1}]")
        return self.alpha_history[t], self.beta_history[t]
