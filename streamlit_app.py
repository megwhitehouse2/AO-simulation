#!/usr/bin/env python3
"""
Assertive Outreach Service Discrete Event Simulation - Streamlit Interface
Main application file using modular components

This DES study follows the STRESS guidelines (Monks et al., 2019) for methodological 
transparency and reproducibility in simulation frameworks.

The simulation models patient flow through an Assertive Outreach (AO) service,
addressing two critical operational questions:
1. Whether separating ED and Non-ED pathways could improve service efficiency
2. The service's resilience to future demand increases

Author: Megan Whitehouse
Institution: University of Exeter
Date: 28/05/2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import simpy
import time
from typing import Dict, List, Optional

# Import simulation modules
from simulation_core import (
    Scenario, create_distributions, calculate_differential_parameters,
    DEFAULT_RANDOM_SEED
)
from simulation_engine import (
    PatientGenerator, ExclusivePatientGenerator, FixedPatientGenerator,
    Auditor, multiple_simulation_runs
)
from visualisation import (
    create_results_summary_table, create_census_comparison_chart,
    create_los_distribution_chart, create_change_analysis_chart,
    create_differential_validation_chart, create_metrics_comparison_table,
    create_export_dataframe, get_scenario_description, validate_results_structure
)

# Configure Streamlit page settings
st.set_page_config(
    page_title="Assertive Outreach Service Simulation",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #005EB8;  /* NHS Blue */
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    .scenario-section {
        background-color: #F0F4F5;  /* NHS Light Grey */
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #005EB8;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #005EB8;  /* NHS Blue */
        margin: 0.5rem 0;
    }
    .clinical-rationale {
        background-color: #F0F4F5;  /* NHS Light Grey */
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #005EB8;  /* NHS Blue */
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Scenario Runner Functions
def extract_serialisable_results(auditor):
    """
    Extract only serialisable data from Auditor object for storage in session state.
    
    The Auditor object contains SimPy environment references that cannot be serialised.
    This function extracts only the data needed for results display and analysis.
    
    Parameters
    ----------
    auditor : Auditor
        The simulation auditor containing results data
        
    Returns
    -------
    dict
        Serialisable dictionary containing:
        - summary_frame: Patient group statistics (census, LOS metrics)
        - referral_proportions: Urgent vs routine referral splits
        - total_admissions: Admission counts by patient group
        - yearly_admissions: Annualised admission rates
    """
    if auditor is None:
        return None
        
    # Create a serialisable version of the results
    serialisable_results = {
        'summary_frame': auditor.summary_frame.copy() if hasattr(auditor, 'summary_frame') else {},
        'referral_proportions': getattr(auditor, 'referral_proportions', {'urgent': 0, 'routine': 0}),
        'total_admissions': auditor.total_admissions.copy() if hasattr(auditor, 'total_admissions') else {},
        'yearly_admissions': auditor.yearly_admissions.copy() if hasattr(auditor, 'yearly_admissions') else {},
    }
    return serialisable_results


def run_scenario_analysis(scenario_type: str, parameters: Dict, 
                         warm_up: int, run_length: int, n_reps: int) -> Dict:
    """
    Run scenario analysis with caching for different operational scenarios.
    
    This is the main entry point for running all five scenarios defined in the methodology:
    - Baseline models (all patients vs separate groups)
    - Service inclusion criteria (1a: Non-ED only, 1b: ED only)
    - Uniform volume changes (2a: -30% to +30%)
    - Differential volume changes (2b: ED increase, 2c: Non-ED increase)
    
    Parameters
    ----------
    scenario_type : str
        Type of scenario to run (baseline_all, baseline_separate, exclusive, 
        uniform_volume, differential_volume)
    parameters : dict
        Scenario-specific parameters (varies by scenario type)
    warm_up : int
        Warm-up period in days (typically 365 to match service establishment period)
    run_length : int
        Simulation run length in days (typically 1825 = 5 years)
    n_reps : int
        Number of replications for statistical validity (typically 10-30)
        
    Returns
    -------
    dict
        Results dictionary containing simulation outputs and metadata
    """
    
    if scenario_type == "baseline_all":
        # Baseline model treating all patients as single stream
        # Used for validation against historical data (2021-2023)
        scenario = Scenario(
            fitted_patient_types=create_distributions(apply_calibration=True),
            separate_patient_groups=False,  # Single patient stream
            random_seed=DEFAULT_RANDOM_SEED
        )
        results = multiple_simulation_runs(scenario, rc_period=run_length, 
                                         warm_up=warm_up, n_reps=n_reps)
        return {'results': results, 'scenario_type': scenario_type}
        
    elif scenario_type == "baseline_separate":
        # Baseline model with ED and Non-ED tracked separately
        # Preserves group-level characteristics for more detailed analysis
        scenario = Scenario(
            fitted_patient_types=create_distributions(apply_calibration=True),
            separate_patient_groups=True,  # Track ED/Non-ED separately
            random_seed=DEFAULT_RANDOM_SEED
        )
        results = multiple_simulation_runs(scenario, rc_period=run_length, 
                                         warm_up=warm_up, n_reps=n_reps)
        return {'results': results, 'scenario_type': scenario_type}
        
    elif scenario_type == "exclusive":
        # Scenarios 1a and 1b: Service inclusion criteria testing
        return run_exclusive_scenario(
            parameters['exclusive_type'], 
            parameters['description'],
            warm_up, run_length, n_reps
        )
        
    elif scenario_type == "uniform_volume":
        # Scenario 2a: Uniform referral rate changes
        return run_uniform_volume_scenario(
            parameters['volume_change'],
            parameters['scenario_name'],
            parameters['description'],
            warm_up, run_length, n_reps
        )
        
    elif scenario_type == "differential_volume":
        # Scenarios 2b and 2c: Differential referral increases
        return run_differential_scenario(
            parameters['target_group'],
            parameters['volume_change'],
            parameters['scenario_name'],
            parameters['description'],
            warm_up, run_length, n_reps
        )
    else:
        raise ValueError(f"Unknown scenario type: {scenario_type}")


def run_exclusive_scenario(exclusive_type: str, description: str, 
                         warm_up: int, run_length: int, n_reps: int) -> Dict:
    """
    Run exclusive service scenarios (1a and 1b) testing service inclusion criteria.
    
    This simulates operating the service exclusively for one patient group:
    - Scenario 1a (non_ed_only): Evaluates service without ED patients, who typically 
      have longer stays. Determines capacity implications if ED referrals were 
      redirected to specialised services.
    - Scenario 1b (ed_only): Examines service functioning with ED patients only. 
      Despite shorter statistical LOS, this group may require disproportionate 
      clinical resources due to less established pathways.
    
    Note: The historical arrival rate remains constant (from CAMHS). As ED and Non-ED 
    are enforced by probabilities, exclusion effectively halves arrival rates, 
    mirroring acceptance/rejection rates for exclusive services.
    
    Parameters
    ----------
    exclusive_type : str
        Either 'ed_only' or 'non_ed_only'
    description : str
        Scenario description for results tracking
    warm_up : int
        Warm-up period in days
    run_length : int
        Simulation run length in days
    n_reps : int
        Number of replications
        
    Returns
    -------
    dict
        Results with excluded group metrics zeroed out
    """
    try:
        # Create scenario with separate group tracking enabled
        scenario = Scenario(
            fitted_patient_types=create_distributions(apply_calibration=True),
            separate_patient_groups=True,  # Required for exclusive scenarios
            random_seed=DEFAULT_RANDOM_SEED
        )
        
        all_results = []
        
        # Run multiple replications with different random seeds
        for rep in range(n_reps):
            # Each replication gets unique seed for statistical independence
            rep_seed = DEFAULT_RANDOM_SEED + rep
            scenario.random_seed = rep_seed
            scenario.rng = np.random.RandomState(rep_seed)
            
            # Create simulation environment
            env = simpy.Environment()
            total_run_length = run_length + warm_up + 5  # +5 days buffer
            
            # Initialise auditor to track metrics
            auditor = Auditor(env,
                            run_length=total_run_length,
                            fitted_patient_types=scenario.fitted_patient_types,
                            scenario=scenario,
                            first_obs=warm_up,  # Start recording after warm-up
                            interval=30)  # Record metrics every 30 days
            
            # Use ExclusivePatientGenerator to filter patient types
            patient_generator = ExclusivePatientGenerator(
                env, scenario, auditor, exclusive_type
            )
            
            # Run simulation
            env.run(until=total_run_length)
            all_results.append(auditor)
        
        # Process results - use final replication as template
        final_auditor = all_results[-1]
        
        # Zero out metrics for excluded group to reflect service design
        if exclusive_type == 'ed_only':
            # Scenario 1b: Zero out non-ED metrics
            if 'non_ed_patients' in final_auditor.summary_frame:
                for metric in final_auditor.summary_frame['non_ed_patients']:
                    if metric != 'yearly_admissions':
                        final_auditor.summary_frame['non_ed_patients'][metric] = 0
                    else:
                        final_auditor.summary_frame['non_ed_patients'][metric] = []
        elif exclusive_type == 'non_ed_only':
            # Scenario 1a: Zero out ED metrics
            if 'ed_patients' in final_auditor.summary_frame:
                for metric in final_auditor.summary_frame['ed_patients']:
                    if metric != 'yearly_admissions':
                        final_auditor.summary_frame['ed_patients'][metric] = 0
                    else:
                        final_auditor.summary_frame['ed_patients'][metric] = []
        
        # Average metrics across all replications for statistical validity
        for group in final_auditor.summary_frame:
            for metric in final_auditor.summary_frame[group]:
                if (metric != 'yearly_admissions' and 
                    final_auditor.summary_frame[group][metric] != 0):
                    # Extract values from all replications
                    values = [result.summary_frame[group][metric] 
                             for result in all_results 
                             if (group in result.summary_frame and 
                                 metric in result.summary_frame[group])]
                    if values:
                        # Use mean for point estimates
                        final_auditor.summary_frame[group][metric] = np.mean(values)
        
        return {
            'results': extract_serialisable_results(final_auditor),
            'description': description,
            'type': 'exclusive',
            'exclusive_type': exclusive_type
        }
        
    except Exception as e:
        return {'results': None, 'error': str(e)}


def run_uniform_volume_scenario(volume_change: float, scenario_name: str, 
                              description: str, warm_up: int, 
                              run_length: int, n_reps: int) -> Dict:
    """
    Run Scenario 2a: Uniform referral rate changes across all patient groups.
    
    Tests system resilience to population-level demand fluctuations by uniformly 
    changing all referral rates. This examines service sustainability under varying 
    demand conditions (e.g., population growth, service awareness changes).
    
    Volume changes are implemented by modifying the Inter-Arrival Time (IAT) scale 
    parameter. A positive volume change decreases IAT (more frequent arrivals).
    
    Parameters
    ----------
    volume_change : float
        Proportional change in referrals (-0.30 to +0.30, i.e., -30% to +30%)
    scenario_name : str
        Identifier for results tracking (e.g., "2a_uniform_+20%")
    description : str
        Full scenario description
    warm_up : int
        Warm-up period in days
    run_length : int
        Simulation run length in days
    n_reps : int
        Number of replications
        
    Returns
    -------
    dict
        Results showing impact of uniform volume change on all metrics
    """
    try:
        # Get calibrated distribution parameters
        params = create_distributions(apply_calibration=True).copy()
        
        if volume_change != 0:
            # Modify IAT to achieve desired volume change
            # volume_multiplier > 1 means more arrivals (shorter IAT)
            original_scale = params.loc['all_patients_iat', 'Scale']
            volume_multiplier = 1.0 + volume_change
            new_scale = original_scale / volume_multiplier  # Inverse relationship
            params.loc['all_patients_iat', 'Scale'] = new_scale
        
        # Create scenario using all_patients model (single stream)
        scenario = Scenario(
            fitted_patient_types=params,
            separate_patient_groups=False,  # Use combined model for uniform changes
            random_seed=DEFAULT_RANDOM_SEED
        )
        
        # Run simulation with modified parameters
        results = multiple_simulation_runs(
            scenario,
            rc_period=run_length,
            warm_up=warm_up,
            n_reps=n_reps
        )
        
        return {
            'results': extract_serialisable_results(results),
            'description': description,
            'type': 'uniform_volume',
            'volume_change': volume_change
        }
        
    except Exception as e:
        return {'results': None, 'error': str(e)}


def run_differential_scenario(target_group: str, volume_change: float, 
                            scenario_name: str, description: str, 
                            warm_up: int, run_length: int, n_reps: int) -> Dict:
    """
    Run Scenarios 2b and 2c: Differential volume changes for specific patient groups.
    
    Tests targeted volume changes in one patient group while keeping the other constant:
    - Scenario 2b (ED increase): Given documented 84% increase in ED presentations 
      over 5 years, evaluates impact on service capacity and resource allocation
    - Scenario 2c (Non-ED increase): Examines implications of increased complex 
      non-ED referrals, testing service adaptability to case mix changes
    
    Uses modified Bernoulli probabilities to achieve group-specific changes while 
    maintaining total arrival rate adjustments.
    
    Parameters
    ----------
    target_group : str
        Group to increase ('ed_patients' or 'non_ed_patients')
    volume_change : float
        Proportional change for target group (-0.30 to +0.30)
    scenario_name : str
        Identifier for results (e.g., "2b_ed_+20%")
    description : str
        Full scenario description
    warm_up : int
        Warm-up period in days
    run_length : int
        Simulation run length in days
    n_reps : int
        Number of replications
        
    Returns
    -------
    dict
        Results showing differential impact on patient groups
    """
    try:
        # Calculate required parameters for differential changes
        # This maintains one group constant while changing the other
        total_multiplier, new_ed_probability = calculate_differential_parameters(
            target_group, volume_change
        )
        
        # Get calibrated parameters and modify overall IAT
        params = create_distributions(apply_calibration=True).copy()
        original_scale = params.loc['all_patients_iat', 'Scale']
        new_scale = original_scale / total_multiplier  # Adjust total arrival rate
        params.loc['all_patients_iat', 'Scale'] = new_scale
        
        # Create scenario with separate groups to track differential effects
        scenario = Scenario(
            fitted_patient_types=params,
            separate_patient_groups=True,  # Required to track group differences
            random_seed=DEFAULT_RANDOM_SEED
        )
        
        # Run multiple replications
        all_results = []
        
        for rep in range(n_reps):
            # Unique seed per replication
            rep_seed = DEFAULT_RANDOM_SEED + rep
            scenario.random_seed = rep_seed
            scenario.rng = np.random.RandomState(rep_seed)
            
            env = simpy.Environment()
            total_run_length = run_length + warm_up + 5
            
            # Initialise auditor
            auditor = Auditor(env,
                            run_length=total_run_length,
                            fitted_patient_types=scenario.fitted_patient_types,
                            scenario=scenario,
                            first_obs=warm_up,
                            interval=30)
            
            # Use FixedPatientGenerator with modified ED probability
            # This achieves differential changes by altering patient mix
            patient_generator = FixedPatientGenerator(
                env, scenario, auditor, 
                ed_probability_override=new_ed_probability
            )
            
            env.run(until=total_run_length)
            all_results.append(auditor)
        
        # Process and average results across replications
        final_result = all_results[-1]
        for group in final_result.summary_frame:
            for metric in final_result.summary_frame[group]:
                if metric != 'yearly_admissions':
                    values = [result.summary_frame[group][metric] 
                             for result in all_results 
                             if (group in result.summary_frame and 
                                 metric in result.summary_frame[group] and
                                 result.summary_frame[group][metric] is not None)]
                    if values:
                        final_result.summary_frame[group][metric] = np.mean(values)
        
        return {
            'results': extract_serialisable_results(final_result),
            'description': description,
            'type': 'differential_volume',
            'target_group': target_group,
            'volume_change': volume_change,
            'total_multiplier': total_multiplier,
            'new_ed_probability': new_ed_probability
        }
        
    except Exception as e:
        return {'results': None, 'error': str(e)}


# MAIN STREAMLIT APPLICATION

def main():
    """
    Main Streamlit application implementing the STRESS guidelines for DES transparency.
    
    The interface provides:
    1. Parameter configuration (warm-up, run length, replications)
    2. Scenario selection and execution
    3. Results visualisation and comparison
    4. Export functionality for further analysis
    
    All scenarios measure both:
    - Point-in-time metrics: Occupancy statistics (min, max, median, SD)
    - Time-aggregated metrics: Admissions, length of stay distributions
    """
    
    # Title and introduction
    st.markdown(
        '<h1 class="main-header">Assertive Outreach Service Simulation</h1>', 
        unsafe_allow_html=True
    )
    
    st.markdown("""
    **Discrete Event Simulation for Assertive Outreach Service Planning**
    
    This simulation models patient flow through an assertive outreach service, 
    supporting scenario analysis for service inclusion criteria and referral 
    volume changes based on calibrated statistical distributions from historical data (2021-2023).
    """)
    
    # Sidebar configuration
    st.sidebar.header("Simulation Configuration")
    
    with st.sidebar.expander("Parameters", expanded=True):
        # Warm-up period: Typically 365 days to match service establishment
        warm_up_days = st.slider("Warm-up Period (days)", 180, 730, 365, 30,
                                help="Period to reach steady state (typically 1 year)")
        
        # Run length: 5 years (1825 days) provides stable long-term metrics
        run_length_days = st.slider("Run Length (days)", 730, 3650, 1825, 365,
                                   help="Simulation duration after warm-up (typically 5 years)")
        
        # Replications: 10-30 for statistical validity
        n_replications = st.slider("Number of Replications", 5, 100, 30, 5,
                                  help="More replications = more reliable results")
        
        # Random seed for reproducibility (STRESS guideline requirement)
        random_seed = st.number_input("Random Seed", value=42, min_value=1,
                                     help="Set for reproducible results")
    
    # Scenario selection aligned with methodology objectives
    st.sidebar.header("Scenario Selection")
    
    scenario_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        [
            "Baseline Models",
            "Service Inclusion (1a, 1b)",
            "Uniform Volume Changes (2a)",
            "Differential Volume Changes (2b, 2c)"
        ]
    )
    
    # Initialise session state for results persistence
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'baseline_results' not in st.session_state:
        st.session_state.baseline_results = None
    

    # BASELINE MODELS - For validation against historical data

    if scenario_type == "Baseline Models":
        st.header("Baseline Model Analysis")
        
        st.markdown("""
        **Purpose:** Establish baseline performance metrics for model 
        validation and comparison against historical data (2021-2023).
        
        - **All Patients Model:** Single patient stream (original approach)
        - **Separate Groups Model:** ED and Non-ED patients tracked separately
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Run All Patients Model", type="primary"):
                with st.spinner("Running All Patients baseline..."):
                    start_time = time.time()
                    result = run_scenario_analysis(
                        "baseline_all", {}, warm_up_days, 
                        run_length_days, n_replications
                    )
                    duration = time.time() - start_time
                    
                    # Store results in session state
                    st.session_state.results['baseline_all'] = result
                    st.session_state.baseline_results = result['results']
                    
                    st.success(f"Completed in {duration:.1f} seconds")
        
        with col2:
            if st.button("Run Separate Groups Model", type="primary"):
                with st.spinner("Running Separate Groups baseline..."):
                    start_time = time.time()
                    result = run_scenario_analysis(
                        "baseline_separate", {}, warm_up_days, 
                        run_length_days, n_replications
                    )
                    duration = time.time() - start_time
                    
                    st.session_state.results['baseline_separate'] = result
                    if st.session_state.baseline_results is None:
                        st.session_state.baseline_results = result['results']
                    
                    st.success(f"Completed in {duration:.1f} seconds")
        
        # Display baseline results with validation metrics
        if ('baseline_all' in st.session_state.results or 
            'baseline_separate' in st.session_state.results):
            
            st.subheader("Baseline Results")
            
            tabs = st.tabs(["All Patients", "Separate Groups"])
            
            with tabs[0]:
                if 'baseline_all' in st.session_state.results:
                    result = st.session_state.results['baseline_all']
                    if validate_results_structure(result['results']):
                        # Display summary statistics
                        summary_df = create_results_summary_table(result['results'])
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Show LOS distribution (key validation metric)
                        los_chart = create_los_distribution_chart(result['results'])
                        st.plotly_chart(los_chart, use_container_width=True)
                    else:
                        st.error("Invalid results structure")
                else:
                    st.info("Run All Patients Model to see results")
            
            with tabs[1]:
                if 'baseline_separate' in st.session_state.results:
                    result = st.session_state.results['baseline_separate']
                    if validate_results_structure(result['results']):
                        summary_df = create_results_summary_table(result['results'])
                        st.dataframe(summary_df, use_container_width=True)
                        
                        los_chart = create_los_distribution_chart(result['results'])
                        st.plotly_chart(los_chart, use_container_width=True)
                    else:
                        st.error("Invalid results structure")
                else:
                    st.info("Run Separate Groups Model to see results")
    

    # SERVICE INCLUSION SCENARIOS (1a, 1b) - Testing specialisation
 
    elif scenario_type == "Service Inclusion (1a, 1b)":
        st.header("Service Inclusion Analysis")
        
        st.markdown("""
        **Research Question:** How would service dynamics change if certain 
        patient groups were excluded?
        
        - **Scenario 1a:** Non-ED patients only (ED patients redirected to 
          specialized services)
        - **Scenario 1b:** ED patients only (Non-ED patients excluded from 
          inpatient care)
        """)
        
        exclusive_type = st.radio(
            "Select Exclusive Service Model:",
            ["non_ed_only", "ed_only"],
            format_func=lambda x: {
                "non_ed_only": "1a: Non-ED Patients Only", 
                "ed_only": "1b: ED Patients Only"
            }[x]
        )
        
        scenario_name = f"1{'a' if exclusive_type == 'non_ed_only' else 'b'}_{exclusive_type}"
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("Run Exclusive Scenario", type="primary"):
                with st.spinner(f"Running {scenario_name}..."):
                    start_time = time.time()
                    result = run_scenario_analysis(
                        "exclusive", 
                        {
                            "exclusive_type": exclusive_type,
                            "description": f"Scenario {scenario_name}"
                        },
                        warm_up_days, run_length_days, n_replications
                    )
                    duration = time.time() - start_time
                    
                    st.session_state.results[scenario_name] = result
                    st.success(f"Completed in {duration:.1f} seconds")
        
        with col2:
            # Clinical rationale from methodology
            clinical_rationale = {
                "non_ed_only": """
                **Clinical Rationale:** Evaluates service without ED patients who 
                typically have longer stays (median 141.5 vs 105.9 days). Determines 
                capacity implications if ED referrals are redirected to specialized 
                services, acknowledging treatment progression challenges.
                """,
                "ed_only": """
                **Clinical Rationale:** Examines service functioning with ED patients 
                only. Despite shorter LOS, this group may require disproportionate 
                clinical resources due to less established pathways and higher 
                comorbidity rates (>50% with autism spectrum conditions).
                """
            }
            st.markdown(
                f'<div class="clinical-rationale">{clinical_rationale[exclusive_type]}</div>',
                unsafe_allow_html=True
            )
        
        # Display scenario results
        if scenario_name in st.session_state.results:
            st.subheader("Exclusive Scenario Results")
            
            result = st.session_state.results[scenario_name]
            
            if result.get('results') and validate_results_structure(result['results']):
                # Summary statistics table
                summary_df = create_results_summary_table(result['results'])
                st.dataframe(summary_df, use_container_width=True)
                
                # Compare with baseline if available
                if (st.session_state.baseline_results and 
                    validate_results_structure(st.session_state.baseline_results)):
                    comparison_chart = create_census_comparison_chart(
                        st.session_state.baseline_results, 
                        result['results'], 
                        scenario_name
                    )
                    st.plotly_chart(comparison_chart, use_container_width=True)
                
                # LOS distribution for included group
                los_chart = create_los_distribution_chart(result['results'])
                st.plotly_chart(los_chart, use_container_width=True)
            else:
                st.error(f"Scenario failed: {result.get('error', 'Invalid results structure')}")
    

    # UNIFORM VOLUME CHANGES (2a) - Population-level demand changes
  
    elif scenario_type == "Uniform Volume Changes (2a)":
        st.header("Uniform Volume Change Analysis")
        
        st.markdown("""
        **Research Question:** How does the service respond to general changes 
        in referral volume?
        
        **Scenario 2a:** Tests system resilience to population-level demand 
        fluctuations by uniformly changing all referral rates. Critical for 
        capacity planning given current near-capacity operation (81-100%).
        """)
        
        volume_change = st.selectbox(
            "Select Volume Change:",
            [-0.30, -0.20, -0.10, 0.10, 0.20, 0.30],
            format_func=lambda x: f"{x:+.0%} Volume Change",
            index=3  # Default to +10%
        )
        
        scenario_name = f"2a_uniform_{volume_change:+.0%}"
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("Run Uniform Volume Scenario", type="primary"):
                with st.spinner(f"Running uniform {volume_change:+.0%} scenario..."):
                    start_time = time.time()
                    result = run_scenario_analysis(
                        "uniform_volume", 
                        {
                            "volume_change": volume_change,
                            "scenario_name": scenario_name,
                            "description": f"Scenario 2a: Uniform {volume_change:+.0%} referral volume change"
                        },
                        warm_up_days, run_length_days, n_replications
                    )
                    duration = time.time() - start_time
                    
                    st.session_state.results[scenario_name] = result
                    st.success(f"Completed in {duration:.1f} seconds")
        
        with col2:
            st.markdown(
                f"""
                <div class="clinical-rationale">
                <strong>Clinical Rationale:</strong><br>
                Examines service sustainability under varying demand conditions. 
                A {volume_change:+.0%} change tests whether the service can maintain 
                quality with baseline median occupancy of 24.3 patients (capacity ~26-27).
                Results indicate 10% increases are manageable, but 20-30% exceed capacity.
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Display uniform volume results
        if scenario_name in st.session_state.results:
            st.subheader("Uniform Volume Results")
            
            result = st.session_state.results[scenario_name]
            
            if result.get('results') and validate_results_structure(result['results']):
                # Summary metrics
                summary_df = create_results_summary_table(result['results'])
                st.dataframe(summary_df, use_container_width=True)
                
                if (st.session_state.baseline_results and 
                    validate_results_structure(st.session_state.baseline_results)):
                    # Census comparison chart
                    comparison_chart = create_census_comparison_chart(
                        st.session_state.baseline_results, 
                        result['results'], 
                        f"Uniform {volume_change:+.0%}"
                    )
                    st.plotly_chart(comparison_chart, use_container_width=True)
                    
                    # Change analysis chart
                    change_chart = create_change_analysis_chart(
                        st.session_state.baseline_results,
                        result['results'],
                        volume_change,
                        f"Uniform {volume_change:+.0%}"
                    )
                    st.plotly_chart(change_chart, use_container_width=True)
                    
                    # Detailed comparison table
                    comparison_df = create_metrics_comparison_table(
                        st.session_state.baseline_results,
                        result['results'],
                        f"Uniform {volume_change:+.0%}"
                    )
                    if not comparison_df.empty:
                        st.subheader("Detailed Comparison")
                        st.dataframe(comparison_df, use_container_width=True)
            else:
                st.error(f"Scenario failed: {result.get('error', 'Invalid results structure')}")
    

    # DIFFERENTIAL VOLUME CHANGES (2b, 2c) - Group-specific changes

    elif scenario_type == "Differential Volume Changes (2b, 2c)":
        st.header("Differential Volume Change Analysis")
        
        st.markdown("""
        **Research Question:** How does targeted volume change in one patient 
        group affect overall service dynamics?
        
        - **Scenario 2b:** ED referrals increase, Non-ED referrals stay constant
        - **Scenario 2c:** Non-ED referrals increase, ED referrals stay constant
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_group = st.radio(
                "Select Target Group:",
                ["ed_patients", "non_ed_patients"],
                format_func=lambda x: {
                    "ed_patients": "2b: ED Patients (Non-ED constant)", 
                    "non_ed_patients": "2c: Non-ED Patients (ED constant)"
                }[x]
            )
        
        with col2:
            volume_change = st.selectbox(
                "Select Volume Change:",
                [-0.30, -0.20, -0.10, 0.10, 0.20, 0.30],
                format_func=lambda x: f"{x:+.0%} Volume Change",
                index=4  # Default to +20%
            )
        
        scenario_name = (f"2{'b' if target_group == 'ed_patients' else 'c'}_"
                        f"{target_group.split('_')[0]}_{volume_change:+.0%}")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("Run Differential Scenario", type="primary"):
                with st.spinner(f"Running {scenario_name}..."):
                    start_time = time.time()
                    result = run_scenario_analysis(
                        "differential_volume", 
                        {
                            "target_group": target_group,
                            "volume_change": volume_change,
                            "scenario_name": scenario_name,
                            "description": f"Scenario {scenario_name}"
                        },
                        warm_up_days, run_length_days, n_replications
                    )
                    duration = time.time() - start_time
                    
                    st.session_state.results[scenario_name] = result
                    st.success(f"Completed in {duration:.1f} seconds")
        
        with col2:
            # Clinical rationale specific to each differential scenario
            clinical_rationale = {
                "ed_patients": """
                **Clinical Rationale:** Given documented 84% increase in ED 
                presentations over 5 years, evaluates impact of rising ED referrals 
                on service capacity. ED patients require 1.48x resources of non-ED 
                due to longer stays (38.5% exceed 200 days).
                """,
                "non_ed_patients": """
                **Clinical Rationale:** Examines implications of increased complex 
                non-ED referrals while ED remains stable. Tests service adaptability 
                to case mix changes, acknowledging treatment complexity from high 
                autism comorbidity rates.
                """
            }
            st.markdown(
                f'<div class="clinical-rationale">{clinical_rationale[target_group]}</div>',
                unsafe_allow_html=True
            )
        
        # Display differential scenario results
        if scenario_name in st.session_state.results:
            st.subheader("Differential Volume Results")
            
            result = st.session_state.results[scenario_name]
            
            if result.get('results') and validate_results_structure(result['results']):
                # Summary statistics
                summary_df = create_results_summary_table(result['results'])
                st.dataframe(summary_df, use_container_width=True)
                
                if (st.session_state.baseline_results and 
                    validate_results_structure(st.session_state.baseline_results)):
                    # Census comparison
                    comparison_chart = create_census_comparison_chart(
                        st.session_state.baseline_results, 
                        result['results'], 
                        scenario_name
                    )
                    st.plotly_chart(comparison_chart, use_container_width=True)
                    
                    # Differential validation chart
                    validation_chart = create_differential_validation_chart(
                        st.session_state.baseline_results,
                        result['results'],
                        target_group,
                        volume_change
                    )
                    st.plotly_chart(validation_chart, use_container_width=True)
                    
                    # Detailed comparison table
                    comparison_df = create_metrics_comparison_table(
                        st.session_state.baseline_results,
                        result['results'],
                        scenario_name
                    )
                    if not comparison_df.empty:
                        st.subheader("Detailed Comparison")
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Validation summary for differential effects
                        st.subheader("Differential Effect Validation")
                        
                        # Extract changes for validation
                        if (isinstance(st.session_state.baseline_results, dict) and 
                            'summary_frame' in st.session_state.baseline_results):
                            baseline_sf = st.session_state.baseline_results['summary_frame']
                        elif hasattr(st.session_state.baseline_results, 'summary_frame'):
                            baseline_sf = st.session_state.baseline_results.summary_frame
                        else:
                            baseline_sf = {}
                        
                        if (isinstance(result['results'], dict) and 
                            'summary_frame' in result['results']):
                            scenario_sf = result['results']['summary_frame']
                        elif hasattr(result['results'], 'summary_frame'):
                            scenario_sf = result['results'].summary_frame
                        else:
                            scenario_sf = {}
                        
                        if baseline_sf and scenario_sf:
                            col1, col2 = st.columns(2)
                            
                            # Validate that changes occurred as expected
                            for group in ['ed_patients', 'non_ed_patients']:
                                if (group in baseline_sf and group in scenario_sf and
                                    baseline_sf[group].get('median_patients', 0) > 0):
                                    
                                    baseline_census = baseline_sf[group]['median_patients']
                                    scenario_census = scenario_sf[group]['median_patients']
                                    actual_change = ((scenario_census - baseline_census) / 
                                                   baseline_census * 100)
                                    
                                    if group == target_group:
                                        with col1:
                                            # Target group should show significant change
                                            if abs(actual_change) > 5:
                                                st.success(
                                                    f"Target group ({group.replace('_', ' ')}) "
                                                    f"changed significantly: {actual_change:+.1f}%"
                                                )
                                            else:
                                                st.warning(
                                                    f"Target group change smaller than expected: "
                                                    f"{actual_change:+.1f}%"
                                                )
                                    else:
                                        with col2:
                                            # Non-target group should remain stable
                                            if abs(actual_change) < 10:
                                                st.success(
                                                    f"Constant group ({group.replace('_', ' ')}) "
                                                    f"remained stable: {actual_change:+.1f}%"
                                                )
                                            else:
                                                st.warning(
                                                    f"Constant group changed unexpectedly: "
                                                    f"{actual_change:+.1f}%"
                                                )
            else:
                st.error(f"Scenario failed: {result.get('error', 'Invalid results structure')}")
    

    
    if st.session_state.results:
        st.sidebar.header("Export Results")
        
        if st.sidebar.button("Export All Results to CSV"):
            try:
                # Create comprehensive export with all scenario results
                export_df = create_export_dataframe(st.session_state.results)
                
                if not export_df.empty:
                    csv = export_df.to_csv(index=False)
                    
                    st.sidebar.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"ao_simulation_results_{int(time.time())}.csv",
                        mime="text/csv"
                    )
                    
                    st.sidebar.success("Export ready!")
                else:
                    st.sidebar.warning("No results to export")
            except Exception as e:
                st.sidebar.error(f"Export failed: {str(e)}")
    

    #Documentation
    
    st.sidebar.header("Documentation")
    st.sidebar.markdown("""

    
    **Key Findings:**
    - ED patients require 1.48x resources vs non-ED
    - Service at 81-100% capacity (24.3/26-27 patients)
    - 10% volume increases manageable, 20-30% exceed capacity
    - Separating pathways reduces capacity by 39%
                        
    Parameters used for these results: 
    - Warmup period: 365 days
    - Run length: 1825 (5 years)
    - Number of replications (n_reps) = 30
    
    **Recommendation:** Maintain integrated service structure while 
    preparing for ED referral increases.
    """)


if __name__ == "__main__":
    main()
