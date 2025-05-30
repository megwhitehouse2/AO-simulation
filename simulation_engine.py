#!/usr/bin/env python3
"""
Assertive Outreach Service Simulation - Engine Components
Contains patient generators, auditor, and simulation runner

Purpose of the simulation is to simulate patient flow through this service, 
which provides support to individuals undergoing core CAMHS treatments
and needing additional intensity within the community.
"""

import numpy as np
import pandas as pd
import simpy
from typing import Dict, List, Optional

from simulation_core import (
    Patient, Scenario, DEFAULT_RANDOM_SEED, routine_probability, urgent_probability,
    create_distributions, calculate_differential_parameters,
    DEFAULT_RESULTS_COLLECTION_PERIOD, DEFAULT_N_REPS
)


class PatientGenerator:
    def __init__(self, env, scenario, auditor, ed_probability_override=None):
        self.env = env
        self.scenario = scenario
        self.auditor = auditor
        self.patient_id = 0
        self.total_patients = 0
        # KEY CHANGE: Allow overriding the ED probability
        self.ed_probability = ed_probability_override if ed_probability_override is not None else 0.559
        self.env.process(self.generate_patients())

    def generate_patients(self):
        if self.scenario.separate_patient_groups:
            # Check only once
            if not hasattr(self, '_debug_printed'):
                self._debug_printed = True
                test_iat = self.scenario.get_distributions('iat', 'all_patients')
                print(f"Separate groups using all_patients IAT with scale: {test_iat.scale}")
                
        if self.scenario.separate_patient_groups:
            while True:
                # Use combined IAT for the single CAMHS referral stream
                iat_distribution = self.scenario.get_distributions(
                    dist_type='iat',
                    patient_group='all_patients'  # calibrated for 50.5/year (mean of 2022: 50, 2023: 51)
                )
                iat_value = iat_distribution.rvs()
                
                # Then classify the patient (Bernoulli implementation)
                if self.scenario.rng.random() < self.ed_probability:
                    patient_group = 'ed_patients'
                else:
                    patient_group = 'non_ed_patients'
                
                self.patient_id += 1
                Patient(self.patient_id, self.env, self.auditor, 
                        self.scenario, patient_group=patient_group)
                
                yield self.env.timeout(iat_value)
        else:
            # All Patients does not split by group
            while True:
                iat_distribution = self.scenario.get_distributions(
                    dist_type='iat',
                    patient_group='all_patients'
                )
                iat_value = iat_distribution.rvs()
                self.patient_id += 1
                Patient(self.patient_id, self.env, self.auditor, 
                    self.scenario, patient_group='all_patients')
                yield self.env.timeout(iat_value)


class IndependentStreamsGenerator:
    """
    Class for Scenario Analysis. 
    Excludes one group whilst the other is generated- this uses seperate processes and random seeds for ED and Non-ED. 
    """
    def __init__(self, env, scenario, auditor):
        self.env = env
        self.scenario = scenario
        self.auditor = auditor
        self.patient_id = 0
        self.total_patients = 0
        
        # Create separate random seeds for each stream
        base_seed = scenario.random_seed
        self.ed_rng = np.random.RandomState(base_seed + 101)  # Different seed for ED
        self.non_ed_rng = np.random.RandomState(base_seed + 102)  # Different seed for non-ED
        
        # Start two independent processes
        self.env.process(self.generate_ed_patients())
        self.env.process(self.generate_non_ed_patients())
    
    def generate_ed_patients(self):
        """Generate ED patients with their own arrival process and RNG"""
        ed_id_counter = 0
        while True:
            # Get ED-specific interarrival time with ED-specific RNG
            iat_distribution = self.scenario.get_distributions(
                dist_type='iat',
                patient_group='ed_patients'
            )
            
            # Create a patient-specific random state for this sampling
            ed_sample_seed = self.ed_rng.randint(1, 10000)
            iat_value = iat_distribution.rvs(random_state=ed_sample_seed)
            
            # Wait for the interarrival time
            yield self.env.timeout(iat_value)
            
            # Create an ED patient with a unique ID
            ed_id_counter += 1
            self.patient_id += 1
            
            # Use ED-specific RNG for patient initialization
            Patient(self.patient_id, self.env, self.auditor, 
                    self.scenario, patient_group='ed_patients')
    
    def generate_non_ed_patients(self):
        """Generate non-ED patients with their own arrival process and RNG"""
        non_ed_id_counter = 0
        while True:
            # Get non-ED-specific interarrival time with non-ED-specific RNG
            iat_distribution = self.scenario.get_distributions(
                dist_type='iat',
                patient_group='non_ed_patients'
            )
            
            # Create a patient-specific random state for this sampling
            non_ed_sample_seed = self.non_ed_rng.randint(1, 10000)
            iat_value = iat_distribution.rvs(random_state=non_ed_sample_seed)
            
            # Wait for the interarrival time
            yield self.env.timeout(iat_value)
            
            # Create a non-ED patient with a unique ID
            non_ed_id_counter += 1
            self.patient_id += 1
            
            # Use non-ED-specific RNG for patient initialization
            Patient(self.patient_id, self.env, self.auditor, 
                    self.scenario, patient_group='non_ed_patients')


class ExclusivePatientGenerator(PatientGenerator):
    def __init__(self, env, scenario, auditor, exclusive_type):
        self.env = env
        self.scenario = scenario
        self.auditor = auditor
        self.patient_id = 0
        self.total_patients = 0
        self.ed_probability = 0.559
        self.exclusive_type = exclusive_type
        self.env.process(self.generate_patients())

    def generate_patients(self):
        if self.scenario.separate_patient_groups:
            while True:
                if self.exclusive_type == 'ed_only':
                    patient_group = 'ed_patients'
                    iat_distribution = self.scenario.get_distributions(
                        dist_type='iat',
                        patient_group='ed_patients'
                    )
                    iat_value = iat_distribution.rvs() / self.ed_probability  # Divide by 0.559
                else:
                    patient_group = 'non_ed_patients'
                    iat_distribution = self.scenario.get_distributions(
                        dist_type='iat',
                        patient_group='non_ed_patients'
                    )
                    iat_value = iat_distribution.rvs() / (1 - self.ed_probability)  # Divide by 0.441
                
                self.patient_id += 1
                Patient(self.patient_id, self.env, self.auditor, 
                    self.scenario, patient_group=patient_group)
                
                yield self.env.timeout(iat_value)
        else:
            # Original all patients code - unchanged
            while True:
                iat_distribution = self.scenario.get_distributions(
                    dist_type='iat',
                    patient_group='all_patients'
                )
                iat_value = iat_distribution.rvs()
                self.patient_id += 1
                Patient(self.patient_id, self.env, self.auditor, 
                    self.scenario, patient_group='all_patients')
                yield self.env.timeout(iat_value)


class FixedPatientGenerator:
    """
    PatientGenerator class to calculate updated ED proportion, based on the % increase needed in the differential scenario analysis.  
    Probability override = another probability is used instead of baseline. 
    for differential scenario analysis
    """
    def __init__(self, env, scenario, auditor, ed_probability_override=None):
        self.env = env
        self.scenario = scenario
        self.auditor = auditor
        self.patient_id = 0
        self.total_patients = 0
        
        # CRITICAL FIX: Ensure ED probability override is properly stored and used
        if ed_probability_override is not None:
            self.ed_probability = float(ed_probability_override)  # Explicit conversion
            print(f"ED probability override applied: {self.ed_probability:.6f}")
        else:
            self.ed_probability = 0.559
            print(f"Using baseline ED probability: {self.ed_probability:.6f}")
        
        #Create RNG for patient classification
        self.classification_rng = np.random.RandomState(scenario.random_seed + 999)
        
        # Debug counters
        self.ed_count = 0
        self.non_ed_count = 0
        
        self.env.process(self.generate_patients())

    def generate_patients(self):
        """
        FIXED patient generation with proper ED probability handling
        """
        if self.scenario.separate_patient_groups:
            # Debug check only once
            if not hasattr(self, '_debug_printed'):
                self._debug_printed = True
                test_iat = self.scenario.get_distributions('iat', 'all_patients')
                print(f"Separate groups using all_patients IAT with scale: {test_iat.scale}")
                print(f"Patient classification will use ED probability: {self.ed_probability:.6f}")
        
        if self.scenario.separate_patient_groups:
            while True:
                # Use modified arrival rate
                iat_distribution = self.scenario.get_distributions(
                    dist_type='iat',
                    patient_group='all_patients'
                )
                iat_value = iat_distribution.rvs()
                
                # Uses dedicated RNG for classification
                classification_random = self.classification_rng.random()
                
                # Explicit comparison with stored ED probability
                if classification_random < self.ed_probability:
                    patient_group = 'ed_patients'
                    self.ed_count += 1
                else:
                    patient_group = 'non_ed_patients'
                    self.non_ed_count += 1
                
                self.patient_id += 1
                self.total_patients += 1
                
                # Debug print statements
                #if self.patient_id <= 5:
                #    print(f"Patient {self.patient_id}: rand={classification_random:.6f}, "
                #          f"threshold={self.ed_probability:.6f}, group={patient_group}")
                
                # Periodic proportion check
                if self.patient_id in [50, 100, 200]:
                    actual_ed_prop = self.ed_count / self.total_patients
                #   print(f"After {self.patient_id} patients: "
                #          f"ED={actual_ed_prop:.3f} (target={self.ed_probability:.3f}), "
                #          f"difference={actual_ed_prop - self.ed_probability:+.3f}")
                
                # Create patient
                Patient(self.patient_id, self.env, self.auditor, 
                        self.scenario, patient_group=patient_group)
                
                yield self.env.timeout(iat_value)
        else:
            # Original all patients code - unchanged
            while True:
                iat_distribution = self.scenario.get_distributions(
                    dist_type='iat',
                    patient_group='all_patients'
                )
                iat_value = iat_distribution.rvs()
                self.patient_id += 1
                Patient(self.patient_id, self.env, self.auditor, 
                    self.scenario, patient_group='all_patients')
                yield self.env.timeout(iat_value)



class Auditor:
    """Collects and processes simulation metrics"""
    def __init__(self, env, run_length, fitted_patient_types, scenario, 
                first_obs=None, interval=None):
        self.env = env
        self.run_length = run_length
        self.scenario = scenario
        self.first_observation = first_obs
        self.interval = interval

        self.departure_log = [] # track departures with timestamps

        # Initialise referral counts
        self.referral_counts = {'urgent': 0, 'routine': 0}

        # Initialise patient tracking dictionaries
        self.patients_count = {
            'all_patients': [],
            'ed_patients': [],
            'non_ed_patients': []
        }
        self.current_patients = {
            'all_patients': 0,
            'ed_patients': 0,
            'non_ed_patients': 0
        }
        self.current_year_admissions = {
            'all_patients': 0,
            'ed_patients': 0,
            'non_ed_patients': 0
        }
        self.yearly_admissions = {
            'all_patients': [],
            'ed_patients': [],
            'non_ed_patients': []
        }
        self.total_admissions = {
            'all_patients': 0,
            'ed_patients': 0,
            'non_ed_patients': 0
        }
        self.year_start_time = 0.0
        self.los_data = pd.DataFrame({
        'referral_type': pd.Series(dtype='object'),  
        'los_days': pd.Series(dtype='float'),
        'patient_group': pd.Series(dtype='object')  
})

        if first_obs is not None:
            self.env.process(self.scheduled_observation())
        self.env.process(self.process_end_of_run())


    def scheduled_observation(self):
        yield self.env.timeout(self.first_observation)
        
        '''if self.scenario.separate_patient_groups:
            print(f"Starting observations at time {self.env.now}")
            print(f"Initial patient counts: ED={self.current_patients['ed_patients']}, Non-ED={self.current_patients['non_ed_patients']}")
        else:
            print(f"Starting observations at time {self.env.now}")
            print(f"Initial patient counts: All={self.current_patients['all_patients']}")
            print(f"Starting observations at time {self.env.now}")
            print(f"Initial patient counts: ED={self.current_patients['ed_patients']}, Non-ED={self.current_patients['non_ed_patients']}")
        '''
        
        while True:
            for group in self.patients_count:
                if self.scenario.separate_patient_groups or group == 'all_patients':
                    self.patients_count[group].append(self.current_patients[group])
            yield self.env.timeout(self.interval)

    def patient_entered(self, referral_type, los, patient_group):
        self.current_patients[patient_group] += 1
        if patient_group != 'all_patients':
            self.current_patients['all_patients'] += 1
        self.referral_counts[referral_type] += 1

        current_time = np.asarray(self.env.now).item()
        if (current_time - self.year_start_time) >= 365:
            print(f"Year complete at time {current_time}")
            for group in self.yearly_admissions:
                self.yearly_admissions[group].append(self.current_year_admissions[group])
                self.current_year_admissions[group] = 0
                self.year_start_time = float(np.floor(current_time / 365) * 365)

        #Seperate groups scenario increments yearly admissions counter as an 'all patients' total
        #This is for an overall patient count, not just the counts of ED and Non-ED
        self.current_year_admissions[patient_group] += 1
        if patient_group != 'all_patients':
            self.current_year_admissions['all_patients'] += 1

        #Seperate groups counter for yearly admissions (same logic as above)
        self.total_admissions[patient_group] += 1
        if patient_group != 'all_patients':
            self.total_admissions['all_patients'] += 1

        new_data = pd.DataFrame({
            'referral_type': [referral_type],
            'los_days': [los],
            'patient_group': [patient_group]
        })
    
        self.los_data = pd.concat([self.los_data, new_data], ignore_index=True)


    def patient_left(self, referral_type, patient_group):
        current_time = self.env.now
        
        # Add to departure log
        self.departure_log.append({
            'time': current_time,
            'group': patient_group,
            'referral_type': referral_type
        })
        
        #Patient counts are decremented depending on scenario and groups to be tested
        self.current_patients[patient_group] -= 1
        if patient_group != 'all_patients':
            self.current_patients['all_patients'] -= 1

    def process_end_of_run(self):
        yield self.env.timeout(self.run_length - 1)
        
        #Debug print
        #print(self.los_data['los_days'].describe(percentiles=[.10, .25, .5, .75, .90]))
    
        results = {}

        # Calculate referral proportions
        total_referrals = self.referral_counts['urgent'] + self.referral_counts['routine']
        urgent_proportion = self.referral_counts['urgent'] / total_referrals if total_referrals > 0 else 0
        routine_proportion = self.referral_counts['routine'] / total_referrals if total_referrals > 0 else 0

        # Ensure the empty data to be stored for each group corresponds to length of their metrics
        # E.g. 12 total metrics for 'All Patients', and 24 combined for 'Separate Groups'
        if self.scenario.separate_patient_groups:
            # Process ED and Non-ED groups individually
            for group in ['ed_patients', 'non_ed_patients']:
                if len(self.patients_count[group]) == 0:
                    results[group] = self._create_empty_results(group)
                else:
                    results[group] = self._process_group_results(group)
            
            # Create combined "all_patients" results from ED + Non-ED data
            results['all_patients'] = self._create_combined_results(results)
            
        else:
            # Original all patients logic
            group = 'all_patients'
            if len(self.patients_count[group]) == 0:
                results[group] = self._create_empty_results(group)
            else:
                results[group] = self._process_group_results(group)

        # Store results in summary frame
        self.summary_frame = results

        #Debugging statements- uncomment for future changes
        '''print(f"\nActual arrival rate check:")
        print(f"Total admissions: {self.total_admissions['all_patients']}")
        print(f"Post-warmup days: {self.run_length - 365}")
        print(f"Annual arrival rate: {self.total_admissions['all_patients'] / ((self.run_length - 365) / 365):.1f}")

        # Check the IAT scale being used
        if hasattr(self.scenario, 'fitted_patient_types'):
            iat_scale = self.scenario.fitted_patient_types.loc['all_patients_iat', 'Scale']
            print(f"IAT scale: {iat_scale:.2f}")
            print(f"Expected arrivals/year: {365 / iat_scale:.1f}")'''

        # Count patients still in system
        in_system_ed = self.current_patients['ed_patients']
        in_system_non_ed = self.current_patients['non_ed_patients']
        total_in_system = self.current_patients['all_patients']

       #print(f"\nPatients still in system at simulation end:")
        #print(f"  ED: {in_system_ed}")
        #print(f"  Non-ED: {in_system_non_ed}")
        #print(f"  Total: {total_in_system}")

        # Check their average time in system so far
        if len(self.los_data) > 0:
            completed_los = self.los_data['los_days'].mean()
        #    print(f"  Average LOS of completed patients: {completed_los:.1f} days")

    #where _ before a method indicates a class-specific, private method
    def _create_empty_results(self, group):
        """Create empty results structure for groups with no data."""
        return {
            'min_patients': 0,
            'max_patients': 0,
            'median_patients': 0,
            'sd_patients': 0,
            'total_admissions': self.total_admissions[group],
            'avg_yearly_admissions': np.mean(self.yearly_admissions[group]) if self.yearly_admissions[group] else 0,
            'p10_los': 0, 'p25_los': 0, 'median_los': 0, 'p75_los': 0, 
            'p90_los': 0, 'p99_los': 0, 'iqr_los': 0, 'sd_los': 0
        }

    def _process_group_results(self, group):
        """Process results for a single patient group."""
        # Append current year's admissions if applicable
        if self.current_year_admissions[group] > 0:
            self.yearly_admissions[group].append(self.current_year_admissions[group])

        # Filter patient counts to exclude low-occupancy periods (< 5 patients)
        filtered_counts = [count for count in self.patients_count[group] if count >= 5]
        min_patients_value = min(self.patients_count[group]) if self.patients_count[group] else 0

        results = {
            'min_patients': min_patients_value,
            'max_patients': max(self.patients_count[group]),  
            'median_patients': np.median(filtered_counts) if filtered_counts else 0,
            'sd_patients': np.std(filtered_counts) if filtered_counts else 0,
            'total_admissions': self.total_admissions[group],
            'avg_yearly_admissions': np.mean(self.yearly_admissions[group]) if self.yearly_admissions[group] else 0
        }

        # Add length-of-stay (LOS) statistics
        group_los_data = self.los_data[self.los_data['patient_group'] == group]
        if len(group_los_data) > 0:
            results.update({
                'p10_los': np.percentile(group_los_data['los_days'], 10),
                'p25_los': np.percentile(group_los_data['los_days'], 25),
                'median_los': np.percentile(group_los_data['los_days'], 50),
                'p75_los': np.percentile(group_los_data['los_days'], 75),
                'p90_los': np.percentile(group_los_data['los_days'], 90),
                'p99_los': np.percentile(group_los_data['los_days'], 99),
                'iqr_los': np.percentile(group_los_data['los_days'], 75) - np.percentile(group_los_data['los_days'], 25),
                'sd_los': np.std(group_los_data['los_days'])
            })
        else:
            # No LOS data available
            results.update({
                'p10_los': 0, 'p25_los': 0, 'median_los': 0, 'p75_los': 0,
                'p90_los': 0, 'p99_los': 0, 'iqr_los': 0, 'sd_los': 0
            })

        # Add yearly admissions data
        results['yearly_admissions'] = self.yearly_admissions[group]
        return results

    def _create_combined_results(self, individual_results):
        """Create combined all_patients results from ED and Non-ED data"""
        ed_results = individual_results.get('ed_patients', {})
        non_ed_results = individual_results.get('non_ed_patients', {})
        
        # Debug print 
        '''print(f"\nDEBUG _create_combined_results:")
        print(f"  ED median: {ed_results.get('median_patients', 0)}")
        print(f"  Non-ED median: {non_ed_results.get('median_patients', 0)}")
        print(f"  Sum: {ed_results.get('median_patients', 0) + non_ed_results.get('median_patients', 0)}")'''
        
        # For exclusive scenarios (1a, 1b), handle differently
        ed_median = ed_results.get('median_patients', 0)
        non_ed_median = non_ed_results.get('median_patients', 0)
        
        # Check if this is an exclusive scenario
        if ed_median == 0 and non_ed_median > 0:
            # Non-ED only scenario (1a)
            print("  → Detected Non-ED only scenario")
            return non_ed_results.copy()
        elif non_ed_median == 0 and ed_median > 0:
            # ED only scenario (1b)
            print("  → Detected ED only scenario")
            return ed_results.copy()
        
        # For mixed scenarios, combine the metrics
        combined_results = {
            # Census metrics - simple addition
            'min_patients': ed_results.get('min_patients', 0) + non_ed_results.get('min_patients', 0),
            'max_patients': ed_results.get('max_patients', 0) + non_ed_results.get('max_patients', 0),
            'median_patients': ed_results.get('median_patients', 0) + non_ed_results.get('median_patients', 0),
            
            # SD of combined groups (approximation using root sum of squares)
            'sd_patients': np.sqrt(ed_results.get('sd_patients', 0)**2 + non_ed_results.get('sd_patients', 0)**2),
            
            # Admission metrics - simple addition
            'total_admissions': ed_results.get('total_admissions', 0) + non_ed_results.get('total_admissions', 0),
            'avg_yearly_admissions': ed_results.get('avg_yearly_admissions', 0) + non_ed_results.get('avg_yearly_admissions', 0)
        }
        
        # For LOS metrics, use weighted average based on admissions
        ed_admissions = ed_results.get('avg_yearly_admissions', 0)
        non_ed_admissions = non_ed_results.get('avg_yearly_admissions', 0)
        total_admissions = ed_admissions + non_ed_admissions
        
        if total_admissions > 0:
            # Calculate weights
            ed_weight = ed_admissions / total_admissions
            non_ed_weight = non_ed_admissions / total_admissions
            
            # Weighted average for LOS percentiles
            los_metrics = ['p10_los', 'p25_los', 'median_los', 'p75_los', 'p90_los', 'p99_los', 'sd_los']
            
            for metric in los_metrics:
                ed_val = ed_results.get(metric, 0)
                non_ed_val = non_ed_results.get(metric, 0)
                combined_results[metric] = (ed_val * ed_weight + non_ed_val * non_ed_weight)
            
            # IQR is special - calculate from p75 and p25
            combined_results['iqr_los'] = combined_results.get('p75_los', 0) - combined_results.get('p25_los', 0)
        else:
            # No admissions, set LOS metrics to 0
            los_metrics = ['p10_los', 'p25_los', 'median_los', 'p75_los', 'p90_los', 'p99_los', 'iqr_los', 'sd_los']
            for metric in los_metrics:
                combined_results[metric] = 0
        
        # Handle yearly admissions list
        ed_yearly = ed_results.get('yearly_admissions', [])
        non_ed_yearly = non_ed_results.get('yearly_admissions', [])
        
        if ed_yearly and non_ed_yearly and len(ed_yearly) == len(non_ed_yearly):
            combined_results['yearly_admissions'] = [ed + non_ed for ed, non_ed in zip(ed_yearly, non_ed_yearly)]
        elif ed_yearly:
            combined_results['yearly_admissions'] = ed_yearly
        elif non_ed_yearly:
            combined_results['yearly_admissions'] = non_ed_yearly
        else:
            combined_results['yearly_admissions'] = []
        
        print(f"  Final combined median: {combined_results['median_patients']}")
        
        return combined_results

    def get_summary(self, separate_patient_groups=False):
        summary_output = []
        summary_output.append("Referral Proportions:")
        total_referrals = self.referral_counts['urgent'] + self.referral_counts['routine']
        if total_referrals > 0:
            summary_output.append(f"Urgent: {self.referral_counts['urgent'] / total_referrals:.2%}")
            summary_output.append(f"Routine: {self.referral_counts['routine'] / total_referrals:.2%}\n")
        else:
            summary_output.append("No referrals were recorded.\n")


        if separate_patient_groups:
            for group in ['ed_patients', 'non_ed_patients']:
                group_name = "Eating Disorder Patients Results:" if group == 'ed_patients' else "Non-Eating Disorder Patients Results:"
                group_metrics = self.summary_frame[group]

                metrics_for_df = {k: v for k, v in group_metrics.items() if not isinstance(v, list)}
                group_df = pd.DataFrame.from_dict(metrics_for_df, orient='index', columns=['estimate'])

                yearly_breakdown = "\nYearly Admissions:"
                for year, admissions in enumerate(group_metrics['yearly_admissions'], 1):
                    yearly_breakdown += f"\nYear {year}: {admissions} admissions"

                summary_output.append(f"{group_name}\n{group_df.to_string()}{yearly_breakdown}\n")

        else:
            group_name = "All Patients Scenario Results:"
            group_metrics = self.summary_frame['all_patients']

            metrics_for_df = {k: v for k, v in group_metrics.items() if not isinstance(v, list)}
            group_df = pd.DataFrame.from_dict(metrics_for_df, orient='index', columns=['estimate'])

            yearly_breakdown = "\nYearly Admissions:"
            for year, admissions in enumerate(group_metrics['yearly_admissions'], 1):
                yearly_breakdown += f"\nYear {year}: {admissions} admissions"

            summary_output.append(f"{group_name}\n{group_df.to_string()}{yearly_breakdown}\n")

        return "\n".join(summary_output)


## Simulation runner functions ##

# Single run to DataFrame aggregates metrics over one run, 
# where multiple_simulation_runs extends on this functionality to add multiple replications

def single_run_to_dataframe(scenario, rc_period=DEFAULT_RESULTS_COLLECTION_PERIOD, warm_up=0):
    """
    Perform a single run of the model and return the Auditor object to access results.

    Parameters:
    -----------
    scenario: Scenario object
        The scenario/parameters to run.

    rc_period: int
        The length of the simulation run that collects results.

    warm_up: int, optional (default=0)
        Warm-up period in the model. The model will not collect any results before the warm-up period is reached.

    Returns:
    --------
    Auditor object
        The Auditor object with the summary results.
    """
    # Run the model
    env = simpy.Environment()
    total_run_length = rc_period + warm_up + 5 # allow 5 days for routine referrals in system to be processed
    auditor = Auditor(env, 
                 run_length=total_run_length, 
                 fitted_patient_types=scenario.fitted_patient_types, 
                 scenario=scenario,
                 first_obs=warm_up,  
                 interval=30)
    
    # Choose the appropriate patient generator based on scenario
    if scenario.separate_patient_groups and hasattr(scenario, 'independent_streams') and scenario.independent_streams:
        # use independent streams for ED and Non ED
        patient_generator = IndependentStreamsGenerator(env, scenario, auditor)
    else:
        # use the generator for all groups
        patient_generator = PatientGenerator(env, scenario, auditor)
    env.run(until=total_run_length)
    
    return auditor


def multiple_simulation_runs(scenario, rc_period, warm_up=0, n_reps=DEFAULT_N_REPS, single_run_func=single_run_to_dataframe):
    all_results = []
    total_referral_counts = {'urgent': 0, 'routine': 0}  
    rep_metrics = []
    
    for rep in range(n_reps):
        #Use a different random seed for each replication
        scenario.random_seed = DEFAULT_RANDOM_SEED + rep
        np.random.seed(scenario.random_seed)
        
        auditor = single_run_func(
            scenario=scenario,
            rc_period=rc_period,
            warm_up=warm_up
        )
        
        total_referral_counts['urgent'] += auditor.referral_counts['urgent']
        total_referral_counts['routine'] += auditor.referral_counts['routine']
        all_results.append(auditor)
        
        # Initialise base metrics
        metrics = {'replication': rep + 1}
        
        if scenario.separate_patient_groups:
            # Add ED metrics if appropriate (e.g. separate groups scenario)
            if 'ed_patients' in auditor.summary_frame and len(auditor.los_data[auditor.los_data['patient_group'] == 'ed_patients']) > 0:
                metrics.update({
                    'ed_min_patients': auditor.summary_frame['ed_patients']['min_patients'],
                    'ed_median_patients': auditor.summary_frame['ed_patients']['median_patients'],
                    'ed_max_patients': auditor.summary_frame['ed_patients']['max_patients'],

                    # Percentile-based LOS metrics
                    'ed_p10_los': auditor.summary_frame['ed_patients'].get('p10_los', 0),
                    'ed_p25_los': auditor.summary_frame['ed_patients'].get('p25_los', 0),
                    'ed_median_los': auditor.summary_frame['ed_patients'].get('median_los', 0),
                    'ed_p75_los': auditor.summary_frame['ed_patients'].get('p75_los', 0),
                    'ed_p90_los': auditor.summary_frame['ed_patients'].get('p90_los', 0),
                    'ed_p99_los': auditor.summary_frame['ed_patients'].get('p99_los',0),
                    'ed_iqr_los': auditor.summary_frame['ed_patients'].get('iqr_los', 0),
                    'ed_sd_los': auditor.summary_frame['ed_patients'].get('sd_los', 0),
                    
                    'ed_yearly_admissions': auditor.summary_frame['ed_patients']['avg_yearly_admissions']
                })
            
            # Add non-ED metrics if they exist
            if 'non_ed_patients' in auditor.summary_frame and len(auditor.los_data[auditor.los_data['patient_group'] == 'non_ed_patients']) > 0:
                metrics.update({
                    # same here for non-ED patient count metrics
                    'non_ed_min_patients': auditor.summary_frame['non_ed_patients']['min_patients'],
                    'non_ed_median_patients': auditor.summary_frame['non_ed_patients']['median_patients'],
                    'non_ed_max_patients': auditor.summary_frame['non_ed_patients']['max_patients'],

                    # New LOS metrics
                    'non_ed_p10_los': auditor.summary_frame['non_ed_patients'].get('p10_los', 0),
                    'non_ed_p25_los': auditor.summary_frame['non_ed_patients'].get('p25_los', 0),
                    'non_ed_median_los': auditor.summary_frame['non_ed_patients'].get('median_los', 0),
                    'non_ed_p75_los': auditor.summary_frame['non_ed_patients'].get('p75_los', 0),
                    'non_ed_p90_los': auditor.summary_frame['non_ed_patients'].get('p90_los', 0),
                    'non_ed_p99_los': auditor.summary_frame['non_ed_patients'].get('p99_los',0),
                    'non_ed_iqr_los': auditor.summary_frame['non_ed_patients'].get('iqr_los', 0),
                    'non_ed_sd_los': auditor.summary_frame['non_ed_patients'].get('sd_los', 0),
                    
                    'non_ed_yearly_admissions': auditor.summary_frame['non_ed_patients']['avg_yearly_admissions']
                })
                
        else:
            metrics.update({
                'min_patients': auditor.summary_frame['all_patients']['min_patients'],
                'median_patients': auditor.summary_frame['all_patients']['median_patients'],
                'max_patients': auditor.summary_frame['all_patients']['max_patients'],

                # New LOS metrics for all patients
                'p10_los': auditor.summary_frame['all_patients'].get('p10_los', 0),
                'p25_los': auditor.summary_frame['all_patients'].get('p25_los', 0),
                'median_los': auditor.summary_frame['all_patients'].get('median_los', 0),
                'p75_los': auditor.summary_frame['all_patients'].get('p75_los', 0),
                'p90_los': auditor.summary_frame['all_patients'].get('p90_los', 0),
                'p99_los': auditor.summary_frame['all_patients'].get('p99_los', 0),
                'iqr_los': auditor.summary_frame['all_patients'].get('iqr_los', 0),
                'sd_los': auditor.summary_frame['all_patients'].get('sd_los', 0),
                
                'yearly_admissions': auditor.summary_frame['all_patients']['avg_yearly_admissions']
            })
        
        rep_metrics.append(metrics)

    # Calculate summary statistics
    metrics_df = pd.DataFrame(rep_metrics)
    
    # Print formatted summary statistics
    print(f"\nSummary Statistics from {n_reps} Simulation Replications (N={n_reps})")
    
    if scenario.separate_patient_groups:
        # Only print ED metrics if they exist in the DataFrame
        if any(col.startswith('ed_') for col in metrics_df.columns):
            print("\nED Patients")
            print("Metric                    Mean (SD)        Range [Min, Max]")
            print("----------------------------------------------------------------")
            ed_metrics = {
                'Min Patients': ['ed_min_patients', 2],
                'Median Patients': ['ed_median_patients', 2],
                'Max Patients': ['ed_max_patients', 2],
            }
             # Add new LOS metrics
            ed_los_metrics = {
                'P10 LOS': ['ed_p10_los', 2],
                'P25 LOS': ['ed_p25_los', 2],
                'Median LOS (P50)': ['ed_median_los', 2],
                'P75 LOS': ['ed_p75_los', 2],
                'P90 LOS': ['ed_p90_los', 2],
                'P99 LOS': ['ed_p99_los', 2],
                'IQR LOS': ['ed_iqr_los', 2],
                'SD LOS': ['ed_sd_los', 2],
                'Yearly Admissions': ['ed_yearly_admissions', 2]
            }
            
            # combine and print metrics
            all_ed_metrics = {**ed_metrics, **ed_los_metrics}

            for metric_name, (col_name, decimals) in all_ed_metrics.items():
                if col_name in metrics_df.columns:
                    mean = metrics_df[col_name].mean()
                    std = metrics_df[col_name].std()
                    min_val = metrics_df[col_name].min()
                    max_val = metrics_df[col_name].max()
                    print(f"{metric_name:<20} {mean:>8.{decimals}f} ({std:.{decimals}f})    [{min_val:.{decimals}f}, {max_val:.{decimals}f}]")
        
        # Only print non-ED metrics if they exist in the DataFrame
        if any(col.startswith('non_ed_') for col in metrics_df.columns):
            print("\nNon-ED Patients")
            print("Metric                    Mean (SD)        Range [Min, Max]")
            print("----------------------------------------------------------------")
            non_ed_metrics = {
                'Min Patients': ['non_ed_min_patients', 2],
                'Median Patients': ['non_ed_median_patients', 2],
                'Max Patients': ['non_ed_max_patients', 2],
            }
             # Add new LOS metrics
            non_ed_los_metrics = {
                'P10 LOS': ['non_ed_p10_los', 2],
                'P25 LOS': ['non_ed_p25_los', 2],
                'Median LOS (P50)': ['non_ed_median_los', 2],
                'P75 LOS': ['non_ed_p75_los', 2],
                'P90 LOS': ['non_ed_p90_los', 2],
                'P99_LOS': ['non_ed_p99_los', 2],
                'IQR LOS': ['non_ed_iqr_los', 2],
                'SD LOS': ['non_ed_sd_los', 2],
                'Yearly Admissions': ['non_ed_yearly_admissions', 2]
            }

            all_non_ed_metrics = {**non_ed_metrics, **non_ed_los_metrics}

            for metric_name, (col_name, decimals) in all_non_ed_metrics.items():
                if col_name in metrics_df.columns:
                    mean = metrics_df[col_name].mean()
                    std = metrics_df[col_name].std()
                    min_val = metrics_df[col_name].min()
                    max_val = metrics_df[col_name].max()
                    print(f"{metric_name:<20} {mean:>8.{decimals}f} ({std:.{decimals}f})    [{min_val:.{decimals}f}, {max_val:.{decimals}f}]")
    
    else:
        print("\nAll Patients")
        print("Metric                    Mean (SD)        Range [Min, Max]")
        print("----------------------------------------------------------------")
        all_patients_metrics = {
            'Min Patients': ['min_patients', 2],
            'Median Patients': ['median_patients', 2],
            'Max Patients': ['max_patients', 2],
        }
        
        # Add new LOS metrics
        all_patients_los_metrics = {
            'P10 LOS': ['p10_los', 2],
            'P25 LOS': ['p25_los', 2],
            'Median LOS (P50)': ['median_los', 2],
            'P75 LOS': ['p75_los', 2],
            'P90 LOS': ['p90_los', 2],
            'P99 LOS': ['p99_los', 2],
            'IQR LOS': ['iqr_los', 2],
            'SD LOS': ['sd_los', 2],
            'Yearly Admissions': ['yearly_admissions', 2]
        }
        
        all_metrics = {**all_patients_metrics, **all_patients_los_metrics}
        for metric_name, (col_name, decimals) in all_metrics.items():
            mean = metrics_df[col_name].mean()
            std = metrics_df[col_name].std()
            min_val = metrics_df[col_name].min()
            max_val = metrics_df[col_name].max()
            print(f"{metric_name:<20} {mean:>8.{decimals}f} ({std:.{decimals}f})    [{min_val:.{decimals}f}, {max_val:.{decimals}f}]")

    final_auditor = all_results[-1]
        
    # Safely aggregate metrics across all results
    for group in final_auditor.summary_frame:
        for metric in final_auditor.summary_frame[group]:
            if metric != 'yearly_admissions':
                # Safely collect values that exist
                values = []
                for result in all_results:
                    if (group in result.summary_frame and 
                        metric in result.summary_frame[group] and
                        result.summary_frame[group][metric] is not None):
                        values.append(result.summary_frame[group][metric])
                    
                    # Only calculate mean if we have values
                if values:
                    final_auditor.summary_frame[group][metric] = np.mean(values)
                else:
                    # Set to 0 or keep original value if no values available
                    final_auditor.summary_frame[group][metric] = 0.0
        
    total_referrals = total_referral_counts['urgent'] + total_referral_counts['routine']
    if total_referrals > 0:
        final_auditor.referral_proportions = {
            'urgent': total_referral_counts['urgent'] / total_referrals * 100,
            'routine': total_referral_counts['routine'] / total_referrals * 100
        }
    else:
        final_auditor.referral_proportions = {'urgent': 0, 'routine': 0}
        
    return final_auditor