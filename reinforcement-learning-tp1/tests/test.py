import pytest
import numpy as np
from unittest.mock import Mock

from multiarmbandits import (
    ArmBernoulli,
    create_bernoulli_arms, 
    NaiveAlgorithm,
    UCBAlgorithm,
    UCBAlgorithmWithHistory,
    ThompsonSamplingAlgorithm,
    ThompsonSamplingAlgorithmWithHistory,
)


class TestArmBernoulli:
    
    def test_initialization(self):
        arm = ArmBernoulli(0.5, random_state=42)
        assert arm.mean == 0.5
        assert arm.local_random is not None
        
    def test_invalid_probability(self):
        with pytest.raises(ValueError):
            ArmBernoulli(-0.1)
        with pytest.raises(ValueError):
            ArmBernoulli(1.1)
            
    def test_sampling_deterministic(self):
        arm1 = ArmBernoulli(0.5, random_state=42)
        arm2 = ArmBernoulli(0.5, random_state=42)
        
        samples1 = [arm1.sample() for _ in range(100)]
        samples2 = [arm2.sample() for _ in range(100)]
        
        assert samples1 == samples2
        
    def test_sampling_bounds(self):
        arm = ArmBernoulli(0.5, random_state=42)
        for _ in range(10):
            sample = arm.sample()
            assert isinstance(sample, (bool, np.bool_))


class TestCreateBernoulliArms:
    
    def test_create_arms(self):
        means = np.array([0.1, 0.5, 0.9])
        arms = create_bernoulli_arms(means, random_state=42)
        
        assert len(arms) == 3
        assert all(isinstance(arm, ArmBernoulli) for arm in arms)
        assert [arm.mean for arm in arms] == [0.1, 0.5, 0.9]


class TestNaiveAlgorithm:
    
    def test_initialization(self):
        alg = NaiveAlgorithm(T=100, N=20, K=5)
        assert alg.T == 100
        assert alg.N == 20
        assert alg.K == 5
        
    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            NaiveAlgorithm(T=20, N=25, K=5)
        with pytest.raises(ValueError):
            NaiveAlgorithm(T=100, N=3, K=5)
        with pytest.raises(ValueError):
            NaiveAlgorithm(T=0, N=20, K=5)
        with pytest.raises(ValueError):
            NaiveAlgorithm(T=100, N=20, K=0)
            
    def test_run_with_mocks(self):
        arms = [Mock() for _ in range(3)]
        for i, arm in enumerate(arms):
            arm.sample.return_value = i % 2
            
        alg = NaiveAlgorithm(T=30, N=15, K=3)
        alg.run(arms)
        
        assert len(alg.rewards) == 30
        assert len(alg.actions) == 30
        assert alg.best_arm in [0, 1, 2]
        
    def test_regret_calculation(self):
        arms = create_bernoulli_arms(np.array([0.1, 0.5, 0.9]), random_state=42)
        alg = NaiveAlgorithm(T=50, N=15, K=3)
        alg.run(arms)
        
        regret = alg.calculate_regret_evolution(0.9)
        assert len(regret) == 50
        assert all(r >= 0 for r in regret)
        assert regret[-1] >= regret[0]


class TestUCBAlgorithm:
    
    def test_initialization(self):
        alg = UCBAlgorithm(T=100, K=5)
        assert alg.T == 100
        assert alg.K == 5
        
    def test_run_with_mocks(self):
        arms = [Mock() for _ in range(3)]
        for arm in arms:
            arm.sample.return_value = 1
            
        alg = UCBAlgorithm(T=30, K=3)
        alg.run(arms)
        
        assert len(alg.rewards) == 30
        assert len(alg.actions) == 30
        assert all(reward == 1 for reward in alg.rewards)
        
    def test_confidence_bounds(self):
        alg = UCBAlgorithm(T=10, K=3)
        
        assert alg._get_confidence_bound(0, 1) == np.inf
        
        alg.arm_counts[0] = 5
        bound = alg._get_confidence_bound(0, 10)
        assert bound > 0 and bound < np.inf


class TestUCBAlgorithmWithHistory:
    
    def test_history_tracking(self):
        arms = create_bernoulli_arms(np.array([0.3, 0.7]), random_state=42)
        alg = UCBAlgorithmWithHistory(T=10, K=2)
        alg.run(arms)
        
        assert len(alg.empirical_means_history) == 10
        assert len(alg.confidence_bounds_history) == 10
        assert len(alg.chosen_arms) == 10


class TestThompsonSamplingAlgorithm:
    
    def test_initialization(self):
        alg = ThompsonSamplingAlgorithm(T=100, K=5)
        assert alg.T == 100
        assert alg.K == 5
        assert np.all(alg.alpha == 1.0)
        assert np.all(alg.beta == 1.0)
        
    def test_custom_priors(self):
        alg = ThompsonSamplingAlgorithm(T=100, K=3, alpha_prior=2.0, beta_prior=3.0)
        assert np.all(alg.alpha == 2.0)
        assert np.all(alg.beta == 3.0)
        
    def test_parameter_updates(self):
        arms = [Mock() for _ in range(2)]
        arms[0].sample.return_value = True
        arms[1].sample.return_value = False
        
        alg = ThompsonSamplingAlgorithm(T=2, K=2)
        
        np.random.seed(42)
        alg.run(arms)
        
        assert np.sum(alg.alpha) >= 2.0 
        assert np.sum(alg.beta) >= 2.0
        total_updates = np.sum(alg.alpha) + np.sum(alg.beta) 
        assert total_updates > 4.0
        
    def test_posterior_statistics(self):
        alg = ThompsonSamplingAlgorithm(T=1, K=2)
        alg.alpha = np.array([3.0, 2.0])
        alg.beta = np.array([2.0, 5.0])
        
        means = alg.get_posterior_means()
        variances = alg.get_posterior_variance()
        intervals = alg.get_credible_interval()
        
        assert len(means) == 2
        assert len(variances) == 2
        assert intervals.shape == (2, 2)
        assert np.all(intervals[:, 0] <= intervals[:, 1])


class TestThompsonSamplingWithHistory:
    
    def test_history_tracking(self):
        arms = create_bernoulli_arms(np.array([0.3, 0.7]), random_state=42)
        alg = ThompsonSamplingAlgorithmWithHistory(T=10, K=2)
        alg.run(arms)
        
        assert len(alg.alpha_history) == 10
        assert len(alg.beta_history) == 10
        assert len(alg.chosen_arms) == 10
        
    def test_get_distribution_at_time(self):
        arms = create_bernoulli_arms(np.array([0.5, 0.5]), random_state=42)
        alg = ThompsonSamplingAlgorithmWithHistory(T=5, K=2)
        alg.run(arms)
        
        alpha, beta = alg.get_distribution_at_time(0)
        assert len(alpha) == 2
        assert len(beta) == 2
        
        with pytest.raises(ValueError):
            alg.get_distribution_at_time(10)


class TestIntegration:
    
    def test_algorithm_comparison(self):
        means = np.array([0.2, 0.5, 0.8])
        arms = create_bernoulli_arms(means, random_state=42)
        optimal_mean = np.max(means)
        
        naive = NaiveAlgorithm(T=50, N=15, K=3)
        ucb = UCBAlgorithm(T=50, K=3)
        thompson = ThompsonSamplingAlgorithm(T=50, K=3)
        
        algorithms = [naive, ucb, thompson]
        
        for alg in algorithms:
            alg.run(arms)
            
        for alg in algorithms:
            assert len(alg.rewards) == 50
            assert len(alg.actions) == 50
            
        for alg in algorithms:
            regret = alg.calculate_regret_evolution(optimal_mean)
            assert len(regret) == 50
            final_regret = regret[-1]
            assert final_regret >= -10
            
        thompson_regret = thompson.get_final_regret(optimal_mean)
        assert thompson_regret >= 0


if __name__ == "__main__":
    pytest.main([__file__])
