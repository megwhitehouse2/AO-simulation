#!/usr/bin/env python3
"""
Assertive Outreach Service Simulation - Core Components
Contains core simulation classes and distribution implementations

AO (Assessment and Outcome) Unit Simulation:
Models patient flow through Assessment and Outcome services that wrap around 
core CAMHS treatments in the community.
"""

import numpy as np
import pandas as pd
import simpy
from scipy.stats import erlang
from typing import Dict, Optional, Tuple


# CONFIGURATION AND PARAMETERS

DEBUG_MODE = False  # Set to True to see debug prints

# Set the probabilities for referrals
routine_probability = 64 / (64 + 42)  # raw counts from historical data
urgent_probability = 42 / (64 + 42)   # raw counts from historical data

DEFAULT_RANDOM_SEED = 42
WARMUP_PERIOD = 365
DEFAULT_N_REPS = 50
DEFAULT_RESULTS_COLLECTION_PERIOD = 365
DEFAULT_RNG_SET = None

# Pre-fitted distribution parameters from analysis
DISTRIBUTION_PARAMS = {
    'los': {  # Length of Stay parameters
        'all_patients': {
            'lognormal': {'mu': 4.65, 'sigma': 0.35}, # Adjusted from mu: 4.8, sigma: 0.25
            'gpd': {'shape': 0.164, 'scale': 130.0}, # fitted to upper tail (>200 days) of LOS distribution
            'prob_below_threshold': 0.67,  # percentage of stays below 200 days based on empirical data
            'threshold': 200  # Days
        },
        'ed_patients': {
            'lognormal': {'mu': 4.64, 'sigma': 0.46}, # reduced from 4.62 and sigma reduced from 0.52
            'gpd': {'shape': 0.213, 'scale': 125.3}, # fitted to upper tail (>200 days) of LOS distribution
            'prob_below_threshold': 0.615,  # increased lognormal proportion (short stay) from 0.60000
            'threshold': 200  # Days
        },
        'non_ed_patients': {
            'lognormal': {'mu': 4.5, 'sigma': 0.58}, # from 4.48, 0.57
            'gpd': {'shape': 0.095, 'scale': 130.0}, # increased from shape : 0.082, 'scale': 122.2, from 0.11, 140.0
            'prob_below_threshold': 0.76,  # increased from 0.77 for more samples below threshold
            'threshold': 220  # Days - Note the different threshold
        }
    },
    'iat': {  
        'all_patients': {
            'shape': 0.93,
            'location': 0.86,
            'scale': 6.5  # Increased from 4.0
        },
        'ed_patients': {
            'shape': 1.11,
            'location': 0.0,
            'scale': 8.0  # Increased from 4.0
        },
        'non_ed_patients': {
            'shape': 2.21,
            'location': 0.0,
            'scale': 7.5  # Increased from 2.0
        }
    }
}


# DISTRIBUTION CLASSES 

class ErlangWrapper:
    def __init__(self, shape, loc, scale, patient_type_key, random_seed=None):  
        self.shape = round(shape) if shape > 0 else 1  # Ensure positive shape for Erlang (not Gamma)
        self.loc = loc
        self.scale = scale  # Use scale directly without multipliers
        self.patient_type = patient_type_key
        self.rng = np.random.RandomState() 
        
        if random_seed is not None:  # Use the random seed if provided
            self.rng.seed(random_seed)

        if self.patient_type == 'ed_patients_iat':
            self.scale = scale  
            self.lower_bound = 0.8
            self.upper_bound = 40.0
            
        elif self.patient_type == 'non_ed_patients_iat':
            self.scale = scale 
            self.lower_bound = 0.8 # from 0.6
            self.upper_bound = 35.0 # increased for more gaps in arrivals- simulation currently overestimates
            
        else:  # all_patients_iat
            self.scale = scale
            self.lower_bound = 0.01
            self.upper_bound = 40.0
            
        self.lower_cdf = erlang.cdf(self.lower_bound, a=self.shape, loc=self.loc, scale=self.scale)
        self.upper_cdf = erlang.cdf(self.upper_bound, a=self.shape, loc=self.loc, scale=self.scale)

    def rvs(self, size=1, random_state=None):
        if random_state is not None:
            self.rng.seed(random_state)
        u = self.rng.uniform(self.lower_cdf, self.upper_cdf, size=size)
        return erlang.ppf(u, a=self.shape, loc=self.loc, scale=self.scale)


class MixedLognormalGPD:
    """
    Mixed distribution using lognormal for values below threshold and 
    Generalized Pareto Distribution (GPD) for values above threshold.
    """
    def __init__(self, lognormal_params, gpd_params, threshold, prob_below_threshold, 
                 lower=30, upper=750, random_seed=42, dev_mode=True, verbose=False):
        self.lognormal_params = lognormal_params
        self.gpd_params = gpd_params
        self.threshold = threshold
        self.prob_below_threshold = prob_below_threshold
        self.lower = lower
        self.upper = upper
        self.dev_mode = dev_mode
        self.verbose = verbose
        self.random_seed = random_seed
        
        # Create a consistent random state for this instance
        self.rng = np.random.RandomState(self.random_seed)
        
    def rvs(self, size=1, random_state=None):
        """Generate random samples from the mixed distribution"""
        # Create a local random state for this call
        local_rng = np.random.RandomState(random_state) if random_state is not None else self.rng
        
        # Determine counts for each distribution component
        n_below = local_rng.binomial(n=size, p=self.prob_below_threshold)
        n_above = size - n_below
        
        if self.verbose:
            print(f"Generating {size} samples: {n_below} below threshold, {n_above} above threshold")
            print(f"Parameters: threshold={self.threshold}, prob_below={self.prob_below_threshold}")
            print(f"Lognormal params: {self.lognormal_params}")
            print(f"GPD params: {self.gpd_params}")

        # Initialise output array
        samples = np.zeros(size)
        
        # Generate samples below threshold with lognormal
        if n_below > 0:
            # Generate with buffer for rejection sampling
            buffer_factor = 1.5
            
            # Generate in batches efficiently
            below_samples = []
            remaining = n_below
            
            while remaining > 0:
                batch_size = int(remaining * buffer_factor)
                batch = np.exp(local_rng.normal(
                    loc=self.lognormal_params['mu'],
                    scale=self.lognormal_params['sigma'],
                    size=batch_size
                ))
                
                # Filter valid samples
                valid = batch[(batch >= self.lower) & (batch <= self.threshold)]
                if len(valid) > 0:
                    below_samples.append(valid[:min(len(valid), remaining)])
                    remaining -= min(len(valid), remaining)
                
                # Avoid infinite loops
                if len(valid) == 0:
                    buffer_factor *= 2
            
            # Combine all batches
            if below_samples:
                all_below_samples = np.concatenate(below_samples)
                samples[:n_below] = all_below_samples
        
        # Generate samples above threshold with GPD
        if n_above > 0:
            # GPD + threshold for samples above threshold
            shape = self.gpd_params['shape']
            scale = self.gpd_params['scale']
            
            # Generate with buffer for upper limit rejection
            buffer_factor = 1.5
            above_samples = []
            remaining = n_above
            
            while remaining > 0:
                batch_size = int(remaining * buffer_factor)
                # Generate GPD samples above threshold
                u = local_rng.uniform(0, 1, size=batch_size)
                
                if abs(shape) < 1e-6:  # Shape ≈ 0, use exponential approximation
                    batch = self.threshold + scale * (-np.log(1 - u))
                else:
                    batch = self.threshold + scale * ((1 - u) ** (-shape) - 1) / shape
                
                # Filter valid samples
                valid = batch[(batch > self.threshold) & (batch <= self.upper)]
                if len(valid) > 0:
                    above_samples.append(valid[:min(len(valid), remaining)])
                    remaining -= min(len(valid), remaining)
                
                # Avoid infinite loops
                if len(valid) == 0:
                    buffer_factor *= 2
            
            # Combine all batches
            if above_samples:
                all_above_samples = np.concatenate(above_samples)
                samples[n_below:] = all_above_samples
        
        # Shuffle to ensure random order
        if size > 1:
            local_rng.shuffle(samples)
        
        # Apply clipping as a safety measure
        samples = np.clip(samples, self.lower, self.upper)

        return samples if size > 1 else samples[0]



# DISTRIBUTION FACTORY FUNCTIONS 

def create_distributions(apply_calibration=True):
    """
    Create statistical distributions for both LOS and IAT.
    Returns DataFrame with fitted parameters.
    
    Parameters:
    -----------
    apply_calibration : bool
        If True, applies calibration factors to IAT scales
    """
    # Calibration factors based on simulation results
    calibration_factors = {
        'all_patients': 1.0,  # Changed from 0.4
        'ed_patients': 1.3,   # Changed from 0.4
        'non_ed_patients': 1.2  # Changed from 0.4
    }
    
    # Initialise empty DataFrame
    data = []
    
    # Create LOS distributions
    for patient_type in ['all_patients', 'ed_patients', 'non_ed_patients']:
        # For lognormal component
        lognormal_params = DISTRIBUTION_PARAMS['los'][patient_type]['lognormal']
        
        # For GPD component
        gpd_params = DISTRIBUTION_PARAMS['los'][patient_type]['gpd']
        
        # Add combined entry for backward compatibility
        data.append({
            'index': f'{patient_type}_los',
            'Shape': lognormal_params['sigma'],  # Use lognormal sigma as shape
            'Location': 0.0,  # Fixed for all
            'Scale': np.exp(lognormal_params['mu'])  # Convert mu to scale
        })
        
        # IAT parameters
        iat_params = DISTRIBUTION_PARAMS['iat'][patient_type]
        
        # Apply calibration if requested
        scale = iat_params['scale']
        if apply_calibration:
            scale *= calibration_factors[patient_type]
            if DEBUG_MODE:
                print(f"Calibrating {patient_type}_iat: {iat_params['scale']} * {calibration_factors[patient_type]} = {scale}")
        else:
            if DEBUG_MODE:
                print(f"NOT calibrating {patient_type}_iat: scale = {scale}")
            
        data.append({
            'index': f'{patient_type}_iat',
            'Shape': iat_params['shape'],
            'Location': iat_params['location'],
            'Scale': scale
        })
    
    # Convert to DataFrame
    fitted_df = pd.DataFrame(data)
    fitted_df.set_index('index', inplace=True)
    
    return fitted_df


def truncated_erlang(patient_type_key, fitted_patient_types, shape=None, loc=None,
    scale=None, random_seed=None):
    """
    Returns a callable object with an `rvs` method for generating samples from a truncated Erlang distribution.
    
    Parameters:
    -----------
    patient_type_key: str
        Key to access specific patient type parameters (e.g., 'all_patients_iat').
    fitted_patient_types: pd.DataFrame
        DataFrame containing shape, location, and scale parameters for each patient type.
    """
    # Get parameters for the given patient type
    shape = shape if shape is not None else fitted_patient_types.loc[patient_type_key, 'Shape']
    loc = loc if loc is not None else fitted_patient_types.loc[patient_type_key, 'Location']
    scale = scale if scale is not None else fitted_patient_types.loc[patient_type_key, 'Scale']
    
    # Return an ErlangWrapper object with patient-specific truncation
    return ErlangWrapper(shape, loc, scale, patient_type_key, random_seed=random_seed)


def mixed_lognormal_gpd(patient_type, distribution_params=None, 
                      lower=None, upper=None, threshold=None, random_seed=None, 
                      dev_mode=True, verbose=False):
    """
    Factory function that creates a MixedLognormalGPD distribution for the specified patient type.
    
    Parameters:
    -----------
    patient_type : str
        Type of patient ('ed_patients', 'non_ed_patients', or 'all_patients')
    distribution_params : dict or DataFrame, optional
        Dictionary with distribution parameters or DataFrame with fitted parameters
    threshold : float, optional
        Threshold value to use instead of the default for the patient type
    ...
    """
    # Set default bounds
    lower = 30 if lower is None else lower
    
    # Set defaults based on patient type
    if upper is None:
        upper = 750
    
    # Use provided random seed or create a deterministic one based on patient type
    if random_seed is not None:
        # Ensure different seeds for each patient type
        if patient_type == 'ed_patients':
            seed = random_seed + 101
        elif patient_type == 'non_ed_patients':
            seed = random_seed + 102
        else:  # all_patients
            seed = random_seed + 100
    else:
        # Default seeds for when none is provided
        if patient_type == 'ed_patients':
            seed = 42
        elif patient_type == 'non_ed_patients':
            seed = 43
        else:  # all_patients
            seed = 44
    
    # Check if distribution_params is a DataFrame or None
    if distribution_params is None or isinstance(distribution_params, pd.DataFrame):
        # Use the default parameters based on patient type
        if patient_type == 'ed_patients':
            lognormal_params = DISTRIBUTION_PARAMS['los']['ed_patients']['lognormal']
            gpd_params = DISTRIBUTION_PARAMS['los']['ed_patients']['gpd']
            prob_below_threshold = DISTRIBUTION_PARAMS['los']['ed_patients']['prob_below_threshold']
            default_threshold = DISTRIBUTION_PARAMS['los']['ed_patients']['threshold']
        elif patient_type == 'non_ed_patients':
            lognormal_params = DISTRIBUTION_PARAMS['los']['non_ed_patients']['lognormal']
            gpd_params = DISTRIBUTION_PARAMS['los']['non_ed_patients']['gpd']
            prob_below_threshold = DISTRIBUTION_PARAMS['los']['non_ed_patients']['prob_below_threshold']
            default_threshold = DISTRIBUTION_PARAMS['los']['non_ed_patients']['threshold']
        else:  # all_patients
            lognormal_params = DISTRIBUTION_PARAMS['los']['all_patients']['lognormal']
            gpd_params = DISTRIBUTION_PARAMS['los']['all_patients']['gpd']
            prob_below_threshold = DISTRIBUTION_PARAMS['los']['all_patients']['prob_below_threshold']
            default_threshold = DISTRIBUTION_PARAMS['los']['all_patients']['threshold']
    else:
        # Use provided dictionary distribution parameters
        lognormal_params = distribution_params['los'][patient_type]['lognormal']
        gpd_params = distribution_params['los'][patient_type]['gpd']
        prob_below_threshold = distribution_params['los'][patient_type]['prob_below_threshold']
        default_threshold = distribution_params['los'][patient_type]['threshold']
    
    # Use provided threshold if available, otherwise use default
    threshold_value = threshold if threshold is not None else default_threshold
    
    return MixedLognormalGPD(
        lognormal_params=lognormal_params,
        gpd_params=gpd_params,
        threshold=threshold_value,
        prob_below_threshold=prob_below_threshold,
        lower=lower,
        upper=upper,
        random_seed=seed,
        dev_mode=dev_mode,
        verbose=verbose
    )


# CORE SIMULATION CLASSES

class Scenario:
    def __init__(self, fitted_patient_types, random_seed=None, separate_patient_groups=False, 
                 independent_streams=False, verbose=False):
        """
        :param fitted_patient_types: The fitted distributions for LOS & IAT (all, ed, non_ed).
        :param random_seed: Optional random seed for reproducibility.
        :param separate_patient_groups: If True, generate ED vs. Non-ED separately.
        :param independent_streams: If True, use completely independent arrival processes.
        :param verbose: Print verbose debugging info.
        """
        self.fitted_patient_types = fitted_patient_types
        self.separate_patient_groups = separate_patient_groups
        self.independent_streams = independent_streams  
        self.verbose = verbose
        # Set a different random seed for each instance
        self.random_seed = random_seed if random_seed is not None else DEFAULT_RANDOM_SEED
        self.rng = np.random.RandomState(self.random_seed)  # Create RandomState instance

    def get_distributions(self, dist_type, patient_group='all_patients'):
        """Returns the distribution object for either the LOS or IAT"""
        # Generate new seed for each distribution using the instance's RNG
        dist_seed = self.rng.randint(1, 10000)
        
        if dist_type == 'los':
            if self.verbose:
                print(f"Creating LOS distribution for {patient_group} with seed {dist_seed}")
            dist = mixed_lognormal_gpd(patient_group, DISTRIBUTION_PARAMS, 
                                    random_seed=dist_seed, verbose=self.verbose)
            
            # Test the distribution by sampling a few values with different random states
            if self.verbose:
                test_samples = [dist.rvs(random_state=dist_seed+i) for i in range(5)]
                print(f"Test samples: {test_samples}")
            
            return dist
        elif dist_type == 'iat':
            return truncated_erlang(f'{patient_group}_iat', self.fitted_patient_types, random_seed=dist_seed)


class Patient:
    def __init__(self, patient_id, env, auditor, scenario, patient_group):
        self.patient_id = patient_id
        self.env = env
        self.auditor = auditor
        self.scenario = scenario
        self.patient_group = patient_group
        self.los_distribution_key = patient_group  # Default to patient group
        self.referral_type, self.los = self.sample_referral_type_and_los()
        self.env.process(self.lifecycle())

    def sample_referral_type_and_los(self):
        # Use scenario's RNG
        if self.scenario.rng.rand() < urgent_probability:
            referral_type = 'urgent'
        else:
            referral_type = 'routine'
        
        # Get LOS distribution - use override if set
        distribution_key = getattr(self, 'los_distribution_key', self.patient_group)
        los_distribution = self.scenario.get_distributions(
            dist_type='los',
            patient_group=distribution_key
        )
        
        # Generate a unique random state for this patient
        patient_random_state = self.patient_id * 100 + self.scenario.random_seed
        
        # Sample LOS
        base_los = los_distribution.rvs(random_state=patient_random_state)
        
        # Add slight variation to prevent clustering (only for non-ED, due to small samples)
        if self.patient_group == 'non_ed_patients':
            # Add small random variation (±10%) to reduce clustering
            variation = 0.9 + 0.2 * self.scenario.rng.random()
            los = base_los * variation
        else:
            los = base_los

        return referral_type, los

    def lifecycle(self):
        # Add delay before admission based on referral type
        if self.referral_type == 'routine':
            delay = 5  # 5 days 
        elif self.referral_type == 'urgent':
            delay = 1 # 1 day
        else:
            delay = 0

        # Wait for the referral delay before entering the system
        yield self.env.timeout(delay)

        # Patient enters the system
        self.auditor.patient_entered(self.referral_type, self.los, self.patient_group)

        # Simulate length of stay
        yield self.env.timeout(self.los)

        # Patient leaves the system
        self.auditor.patient_left(self.referral_type, self.patient_group)



# UTILITY FUNCTIONS

def calculate_differential_parameters(target_group, volume_change):
    """Calculate correct parameters for differential scenarios"""
    
    # CHANGE THESE TWO LINES to use simulation steady-state proportions:
    baseline_ed_prop = 0.559    # Was 0.521 - now using simulation reality
    baseline_non_ed_prop = 0.441 # Was 0.479 - now using simulation reality
    
    # Everything else stays exactly the same
    if target_group == 'ed_patients':
        total_multiplier = 1 + (volume_change * baseline_ed_prop)
        new_ed_probability = ((1 + volume_change) * baseline_ed_prop) / total_multiplier
    else:  # non_ed_patients
        total_multiplier = 1 + (volume_change * baseline_non_ed_prop)
        new_ed_probability = baseline_ed_prop / total_multiplier
    
    return total_multiplier, new_ed_probability

