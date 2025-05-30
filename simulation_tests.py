import pytest
import numpy as np
import pandas as pd
import simpy
import scipy.stats
import copy
from pathlib import Path

from simulation_core import (
    truncated_erlang, mixed_lognormal_gpd, MixedLognormalGPD, Patient,
    Scenario, DISTRIBUTION_PARAMS, DEFAULT_RANDOM_SEED, 
    create_distributions, routine_probability, urgent_probability,
    DEFAULT_N_REPS, DEFAULT_RESULTS_COLLECTION_PERIOD,
    calculate_differential_parameters
)

from simulation_engine import (
    PatientGenerator, IndependentStreamsGenerator, ExclusivePatientGenerator,
    Auditor, single_run_to_dataframe, multiple_simulation_runs
)
RANDOM_SEED = 42


@pytest.fixture
def fitted_patient_types():
    """Test fixture with simulation parameters"""
    return pd.DataFrame({
        'Shape': [0.638, 3.447, 2.626, 1.23, 1.4, 1.3],
        'Location': [0, 0, 0, 0, 0, 0],
        'Scale': [154.52, 40.0, 38.0, 7.5, 8.0, 3.0]
    }, index=['all_patients_los', 'ed_patients_los', 'non_ed_patients_los',
              'all_patients_iat', 'ed_patients_iat', 'non_ed_patients_iat'])


class TestBasicSimulation:
    """Test basic simulation functionality"""
    
    def test_simulation_runs(self, fitted_patient_types):
        """Test that the simulation runs without errors"""
        scenario = Scenario(fitted_patient_types, random_seed=DEFAULT_RANDOM_SEED)
        auditor = single_run_to_dataframe(scenario, rc_period=100)
        assert len(auditor.los_data) > 0, "Simulation did not generate any patients"
    
    def test_patient_lifecycle(self, fitted_patient_types):
        """Test patient lifecycle from creation through departure"""
        env = simpy.Environment()
        scenario = Scenario(fitted_patient_types, random_seed=RANDOM_SEED, verbose=False)
        auditor = Auditor(env, run_length=500, fitted_patient_types=fitted_patient_types, scenario=scenario)
        
        patient = Patient(1, env, auditor, scenario, 'ed_patients')
        
        assert patient.patient_id == 1
        assert patient.patient_group == 'ed_patients'
        assert 21 <= patient.los <= 750
        assert patient.referral_type in ['urgent', 'routine']
    
    def test_auditor_tracking(self, fitted_patient_types):
        """Test that auditor correctly tracks patient statistics"""
        env = simpy.Environment()
        scenario = Scenario(fitted_patient_types, random_seed=RANDOM_SEED, verbose=False)
        auditor = Auditor(env, run_length=100, fitted_patient_types=fitted_patient_types, scenario=scenario)
        
        auditor.patient_entered('urgent', 150, 'ed_patients')
        assert auditor.total_admissions['ed_patients'] == 1
        assert auditor.current_patients['ed_patients'] == 1
        
        auditor.patient_left('urgent', 'ed_patients')
        assert auditor.current_patients['ed_patients'] == 0


class TestDistributions:
    """Test distribution properties and bounds"""
    
    def test_iat_distribution_bounds(self, fitted_patient_types):
        """Test bounds for all IAT Distributions"""
        distributions = {
            'ed_patients_iat': (0.8, 40.0),
            'non_ed_patients_iat': (0.8, 35.0),
            'all_patients_iat': (0.1, 40.0)
        }
        
        for dist_type, (lower, upper) in distributions.items():
            dist = truncated_erlang(dist_type, fitted_patient_types, random_seed=RANDOM_SEED)
            samples = np.array([dist.rvs() for _ in range(1000)])
            
            assert np.all(samples >= lower), f'{dist_type} has values below {lower}'
            assert np.all(samples <= upper), f'{dist_type} has values above {upper}'
    
    def test_los_distribution_properties(self, fitted_patient_types):
        """Test that the LOS distribution has expected statistical properties"""
        scenario = Scenario(fitted_patient_types, random_seed=DEFAULT_RANDOM_SEED)
        auditor = single_run_to_dataframe(scenario, rc_period=1460, warm_up=365)
        
        los_values = auditor.los_data['los_days']
        assert len(los_values) > 0, "No LOS values generated"
        
        # Check threshold proportions
        below_threshold = los_values[los_values <= 200]
        above_threshold = los_values[los_values > 200]
        
        assert len(below_threshold) > 0 and len(above_threshold) > 0
        
        expected_prop = DISTRIBUTION_PARAMS['los']['all_patients']['prob_below_threshold']
        actual_prop = len(below_threshold) / len(los_values)
        assert abs(expected_prop - actual_prop) < 0.1, \
            f"Proportion below threshold {actual_prop:.2f} too far from expected {expected_prop:.2f}"
    
    def test_mixed_distribution_randomness(self):
        """Test distribution randomness and reproducibility"""
        dist = mixed_lognormal_gpd('all_patients', DISTRIBUTION_PARAMS)
        
        # Test different random states produce different results
        samples1 = [dist.rvs(random_state=i) for i in range(10)]
        samples2 = [dist.rvs(random_state=i+100) for i in range(10)]
        assert not all(abs(a - b) < 0.001 for a, b in zip(samples1, samples2))
        
        # Test same random state produces same results
        assert dist.rvs(random_state=42) == dist.rvs(random_state=42)


class TestReproducibility:
    """Test simulation reproducibility"""
    
    def test_same_seed_reproducibility(self, fitted_patient_types):
        """Test that same seed produces same results"""
        results = []
        for _ in range(2):
            scenario = Scenario(fitted_patient_types, random_seed=42)
            auditor = single_run_to_dataframe(scenario, rc_period=500)
            results.append(auditor.los_data['los_days'].values)
        
        np.testing.assert_almost_equal(results[0], results[1], decimal=10)
    
    def test_different_seeds_produce_different_results(self, fitted_patient_types):
        """Test that different seeds produce different results"""
        scenario1 = Scenario(fitted_patient_types, random_seed=42)
        scenario2 = Scenario(fitted_patient_types, random_seed=43)
        
        auditor1 = single_run_to_dataframe(scenario1, rc_period=500)
        auditor2 = single_run_to_dataframe(scenario2, rc_period=500)
        
        los1 = auditor1.los_data['los_days'].values
        los2 = auditor2.los_data['los_days'].values
        
        # Check at least one value is different
        min_len = min(len(los1), len(los2))
        assert not np.array_equal(los1[:min_len], los2[:min_len])


class TestGeneratorIndependence:
    """Test independence of patient arrival streams"""
    
    def test_stream_independence(self, fitted_patient_types):
        """Test that IndependentStreamsGenerator creates truly independent streams"""
        original_params = copy.deepcopy(DISTRIBUTION_PARAMS)
        
        try:
            # Test with modified ED parameters
            DISTRIBUTION_PARAMS['iat']['ed_patients']['scale'] *= 3
            modified_fitted_types = create_distributions()
            
            # Run with independent streams
            env = simpy.Environment()
            scenario = Scenario(modified_fitted_types, separate_patient_groups=True,
                              independent_streams=True, random_seed=42)
            auditor = Auditor(env, run_length=600, scenario=scenario, 
                            fitted_patient_types=modified_fitted_types)
            IndependentStreamsGenerator(env, scenario, auditor)
            env.run(until=600)
            
            ind_ed = auditor.total_admissions['ed_patients']
            ind_non_ed = auditor.total_admissions['non_ed_patients']
            
            # Run with shared stream
            DISTRIBUTION_PARAMS['iat']['ed_patients']['scale'] = original_params['iat']['ed_patients']['scale']
            DISTRIBUTION_PARAMS['iat']['all_patients']['scale'] *= 3
            modified_fitted_types = create_distributions()
            
            env = simpy.Environment()
            scenario = Scenario(modified_fitted_types, separate_patient_groups=True,
                              independent_streams=False, random_seed=42)
            auditor = Auditor(env, run_length=600, scenario=scenario,
                            fitted_patient_types=modified_fitted_types)
            PatientGenerator(env, scenario, auditor)
            env.run(until=600)
            
            shared_ed = auditor.total_admissions['ed_patients']
            shared_non_ed = auditor.total_admissions['non_ed_patients']
            
        finally:
            for key in original_params:
                DISTRIBUTION_PARAMS[key] = original_params[key]
        
        # Independent streams should show less coupling
        # When we modify ED params, non-ED should be less affected in independent streams
        assert ind_ed < shared_ed * 0.8  # ED should be more affected in independent
        assert abs(ind_non_ed - shared_non_ed) < shared_non_ed * 0.3  # Non-ED less affected


class TestStatisticalProperties:
    """Test statistical properties of distributions"""
    
    def test_referral_type_probabilities(self):
        """Test urgent vs routine referral probabilities"""
        routine_prob = 64 / 106
        urgent_prob = 42 / 106
        
        np.random.seed(42)
        referrals = np.random.choice(['routine', 'urgent'], size=10000, 
                                   p=[routine_prob, urgent_prob])
        
        unique, counts = np.unique(referrals, return_counts=True)
        observed = dict(zip(unique, counts / len(referrals)))
        
        assert np.isclose(observed['routine'], routine_prob, atol=0.02)
        assert np.isclose(observed['urgent'], urgent_prob, atol=0.02)
    
    def test_patient_group_proportions(self):
        """Test ED vs Non-ED patient group proportions"""
        ed_prob = 0.521
        
        np.random.seed(42)
        groups = np.random.choice(['ed', 'non_ed'], size=10000, 
                                p=[ed_prob, 1-ed_prob])
        
        unique, counts = np.unique(groups, return_counts=True)
        observed = dict(zip(unique, counts / len(groups)))
        
        assert np.isclose(observed['ed'], ed_prob, atol=0.02)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_invalid_distribution_type(self):
        """Test handling of invalid distribution types"""
        data = pd.DataFrame({
            'Shape': [1.0], 'Location': [0], 'Scale': [1.0]
        }, index=['test_los'])
        
        with pytest.raises(KeyError):
            truncated_erlang('invalid_type', data)
    
    def test_complete_simulation_run(self, fitted_patient_types):
        """Test a full simulation run with all components"""
        scenario = Scenario(fitted_patient_types, random_seed=RANDOM_SEED, verbose=False)
        env = simpy.Environment()
        auditor = Auditor(env, run_length=500, scenario=scenario, 
                         fitted_patient_types=fitted_patient_types,
                         first_obs=50, interval=30)
        generator = PatientGenerator(env, scenario, auditor)
        env.run(until=500)
        
        assert auditor.total_admissions['all_patients'] > 0
        assert len(auditor.los_data) > 0


# Utility function for parameter sensitivity testing
def test_parameter_sensitivity(fitted_patient_types):
    """Test that changing distribution parameters affects results as expected"""
    params1 = copy.deepcopy(DISTRIBUTION_PARAMS)
    params2 = copy.deepcopy(DISTRIBUTION_PARAMS)
    
    params1['los']['all_patients']['prob_below_threshold'] = 0.8
    params2['los']['all_patients']['prob_below_threshold'] = 0.4
    
    samples1 = []
    samples2 = []
    
    for i in range(1000):
        dist1 = mixed_lognormal_gpd('all_patients', params1, random_seed=i+1000)
        dist2 = mixed_lognormal_gpd('all_patients', params2, random_seed=i+1000)
        samples1.append(dist1.rvs(random_state=i+1000))
        samples2.append(dist2.rvs(random_state=i+1000))
    
    below1 = sum(1 for v in samples1 if v <= 200) / len(samples1)
    below2 = sum(1 for v in samples2 if v <= 200) / len(samples2)
    
    assert below1 > below2  # Higher prob_below_threshold = more values below
    assert abs(below1 - 0.8) < 0.1  # Within 10% of target
    assert abs(below2 - 0.4) < 0.1  # Within 10% of target