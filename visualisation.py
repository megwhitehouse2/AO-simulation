#!/usr/bin/env python3
"""
Mental Health Service Simulation - Visualisation Components
Contains plotting and charting functions for Streamlit interface
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional

# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def create_results_summary_table(results: Dict) -> pd.DataFrame:
    """Create a summary table of key metrics"""
    summary_data = []
    
    if isinstance(results, dict) and 'summary_frame' in results:
        sf = results['summary_frame']
    elif hasattr(results, 'summary_frame'):
        sf = results.summary_frame
    else:
        return pd.DataFrame()
    
    for group in sf.keys():
        if sf[group].get('median_patients', 0) > 0:
            summary_data.append({
                'Patient Group': group.replace('_', ' ').title(),
                'Median Census': f"{sf[group].get('median_patients', 0):.1f}",
                'Annual Admissions': f"{sf[group].get('avg_yearly_admissions', 0):.1f}",
                'Median LOS (days)': f"{sf[group].get('median_los', 0):.1f}",
                'LOS IQR (days)': f"{sf[group].get('iqr_los', 0):.1f}",
                'Max Census': f"{sf[group].get('max_patients', 0):.0f}"
            })
    
    return pd.DataFrame(summary_data)


def extract_census_data(results: Dict) -> tuple:
    """Extract census data from results for plotting"""
    if isinstance(results, dict) and 'summary_frame' in results:
        sf = results['summary_frame']
    elif hasattr(results, 'summary_frame'):
        sf = results.summary_frame
    else:
        return [], []
    
    groups = []
    census_values = []
    
    for group in sf.keys():
        if sf[group].get('median_patients', 0) > 0:
            groups.append(group.replace('_', ' ').title())
            census_values.append(sf[group].get('median_patients', 0))
    
    return groups, census_values


def extract_los_data(results: Dict) -> Dict:
    """Extract LOS percentile data from results for plotting"""
    if isinstance(results, dict) and 'summary_frame' in results:
        sf = results['summary_frame']
    elif hasattr(results, 'summary_frame'):
        sf = results.summary_frame
    else:
        return {}
    
    los_data = {}
    
    for group in sf.keys():
        if sf[group].get('median_patients', 0) > 0:
            percentiles = [10, 25, 50, 75, 90, 99]
            values = [
                sf[group].get('p10_los', 0),
                sf[group].get('p25_los', 0),
                sf[group].get('median_los', 0),
                sf[group].get('p75_los', 0),
                sf[group].get('p90_los', 0),
                sf[group].get('p99_los', 0)
            ]
            
            los_data[group.replace('_', ' ').title()] = {
                'percentiles': percentiles,
                'values': values
            }
    
    return los_data


# =============================================================================
# CHART CREATION FUNCTIONS
# =============================================================================

def create_census_comparison_chart(baseline_results: Dict, 
                                 scenario_results: Dict, 
                                 scenario_name: str) -> go.Figure:
    """Create comparison chart for census metrics"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Baseline', f'{scenario_name}'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    baseline_groups, baseline_census = extract_census_data(baseline_results)
    scenario_groups, scenario_census = extract_census_data(scenario_results)
    
    # Add baseline bars
    if baseline_groups:
        fig.add_trace(
            go.Bar(x=baseline_groups, y=baseline_census, name='Baseline', 
                  marker_color='lightblue'),
            row=1, col=1
        )
    
    # Add scenario bars
    if scenario_groups:
        fig.add_trace(
            go.Bar(x=scenario_groups, y=scenario_census, name=scenario_name,
                  marker_color='orange'),
            row=1, col=2
        )
    
    fig.update_layout(
        title_text="Census Comparison",
        showlegend=False,
        height=400
    )
    
    fig.update_yaxes(title_text="Median Census", row=1, col=1)
    fig.update_yaxes(title_text="Median Census", row=1, col=2)
    
    return fig


def create_los_distribution_chart(results: Dict) -> go.Figure:
    """Create length of stay distribution chart"""
    
    los_data = extract_los_data(results)
    
    if not los_data:
        return go.Figure()
    
    fig = go.Figure()
    colors = ['blue', 'red', 'green']
    
    for i, (group, data) in enumerate(los_data.items()):
        fig.add_trace(go.Scatter(
            x=data['percentiles'],
            y=data['values'],
            mode='lines+markers',
            name=group,
            line=dict(color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        title="Length of Stay Distribution (Percentiles)",
        xaxis_title="Percentile",
        yaxis_title="Length of Stay (days)",
        height=400
    )
    
    return fig


def create_change_analysis_chart(baseline_results: Dict, 
                                scenario_results: Dict,
                                expected_change: float,
                                scenario_name: str) -> go.Figure:
    """Create chart showing expected vs actual changes"""
    
    if isinstance(baseline_results, dict) and 'summary_frame' in baseline_results:
        baseline_sf = baseline_results['summary_frame']
    elif hasattr(baseline_results, 'summary_frame'):
        baseline_sf = baseline_results.summary_frame
    else:
        return go.Figure()
    
    if isinstance(scenario_results, dict) and 'summary_frame' in scenario_results:
        scenario_sf = scenario_results['summary_frame']
    elif hasattr(scenario_results, 'summary_frame'):
        scenario_sf = scenario_results.summary_frame
    else:
        return go.Figure()
    
    groups = []
    expected_changes = []
    actual_changes = []
    
    for group in baseline_sf.keys():
        if (baseline_sf[group].get('median_patients', 0) > 0 and 
            group in scenario_sf):
            
            baseline_census = baseline_sf[group]['median_patients']
            scenario_census = scenario_sf[group]['median_patients']
            actual_change = (scenario_census - baseline_census) / baseline_census * 100
            
            groups.append(group.replace('_', ' ').title())
            expected_changes.append(expected_change * 100)
            actual_changes.append(actual_change)
    
    if not groups:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Expected Change',
        x=groups,
        y=expected_changes,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Actual Change',
        x=groups,
        y=actual_changes,
        marker_color='orange'
    ))
    
    fig.update_layout(
        title=f"Expected vs Actual Changes - {scenario_name}",
        xaxis_title="Patient Group",
        yaxis_title="Change (%)",
        barmode='group',
        height=400
    )
    
    return fig


def create_differential_validation_chart(baseline_results: Dict,
                                        scenario_results: Dict,
                                        target_group: str,
                                        volume_change: float) -> go.Figure:
    """Create chart validating differential effects"""
    
    if isinstance(baseline_results, dict) and 'summary_frame' in baseline_results:
        baseline_sf = baseline_results['summary_frame']
    elif hasattr(baseline_results, 'summary_frame'):
        baseline_sf = baseline_results.summary_frame
    else:
        return go.Figure()
    
    if isinstance(scenario_results, dict) and 'summary_frame' in scenario_results:
        scenario_sf = scenario_results['summary_frame']
    elif hasattr(scenario_results, 'summary_frame'):
        scenario_sf = scenario_results.summary_frame
    else:
        return go.Figure()
    
    groups = []
    changes = []
    expected_status = []
    colors = []
    
    for group in ['ed_patients', 'non_ed_patients']:
        if (group in baseline_sf and group in scenario_sf and
            baseline_sf[group].get('median_patients', 0) > 0):
            
            baseline_census = baseline_sf[group]['median_patients']
            scenario_census = scenario_sf[group]['median_patients']
            actual_change = (scenario_census - baseline_census) / baseline_census * 100
            
            groups.append(group.replace('_', ' ').title())
            changes.append(actual_change)
            
            if group == target_group:
                expected_status.append(f"Target: {volume_change:+.0%}")
                colors.append('orange')
            else:
                expected_status.append("Should be constant")
                colors.append('lightblue')
    
    if not groups:
        return go.Figure()
    
    fig = go.Figure(data=[
        go.Bar(x=groups, y=changes, 
               marker_color=colors,
               text=[f"{change:+.1f}%" for change in changes],
               textposition='auto')
    ])
    
    fig.update_layout(
        title="Differential Effect Validation",
        xaxis_title="Patient Group", 
        yaxis_title="Census Change (%)",
        height=400
    )
    
    # Add horizontal line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    return fig


def create_metrics_comparison_table(baseline_results: Dict,
                                   scenario_results: Dict,
                                   scenario_name: str) -> pd.DataFrame:
    """Create detailed comparison table"""
    
    def extract_summary_frame(results):
        if isinstance(results, dict) and 'summary_frame' in results:
            return results['summary_frame']
        elif hasattr(results, 'summary_frame'):
            return results.summary_frame
        else:
            return {}
    
    baseline_sf = extract_summary_frame(baseline_results)
    scenario_sf = extract_summary_frame(scenario_results)
    
    if not baseline_sf or not scenario_sf:
        return pd.DataFrame()
    
    comparison_data = []
    
    for group in baseline_sf.keys():
        if (baseline_sf[group].get('median_patients', 0) > 0 and 
            group in scenario_sf):
            
            baseline_census = baseline_sf[group]['median_patients']
            scenario_census = scenario_sf[group]['median_patients']
            census_change = (scenario_census - baseline_census) / baseline_census * 100
            
            baseline_admits = baseline_sf[group]['avg_yearly_admissions']
            scenario_admits = scenario_sf[group]['avg_yearly_admissions']
            admits_change = (scenario_admits - baseline_admits) / baseline_admits * 100
            
            baseline_los = baseline_sf[group]['median_los']
            scenario_los = scenario_sf[group]['median_los']
            los_change = (scenario_los - baseline_los) / baseline_los * 100 if baseline_los > 0 else 0
            
            comparison_data.append({
                'Patient Group': group.replace('_', ' ').title(),
                'Baseline Census': f"{baseline_census:.1f}",
                f'{scenario_name} Census': f"{scenario_census:.1f}",
                'Census Change (%)': f"{census_change:+.1f}%",
                'Baseline Admissions': f"{baseline_admits:.1f}",
                f'{scenario_name} Admissions': f"{scenario_admits:.1f}",
                'Admissions Change (%)': f"{admits_change:+.1f}%",
                'Baseline LOS': f"{baseline_los:.1f}",
                f'{scenario_name} LOS': f"{scenario_los:.1f}",
                'LOS Change (%)': f"{los_change:+.1f}%"
            })
    
    return pd.DataFrame(comparison_data)


def create_export_dataframe(results_dict: Dict) -> pd.DataFrame:
    """Create comprehensive DataFrame for CSV export"""
    
    export_data = []
    
    for scenario_key, scenario_data in results_dict.items():
        # Handle different result structures
        if 'results' in scenario_data:
            if isinstance(scenario_data['results'], dict) and 'summary_frame' in scenario_data['results']:
                sf = scenario_data['results']['summary_frame']
            elif hasattr(scenario_data['results'], 'summary_frame'):
                sf = scenario_data['results'].summary_frame
            else:
                continue
        elif isinstance(scenario_data, dict) and 'summary_frame' in scenario_data:
            sf = scenario_data['summary_frame']
        elif hasattr(scenario_data, 'summary_frame'):
            sf = scenario_data.summary_frame
        else:
            continue
        
        for group in sf.keys():
            if sf[group].get('median_patients', 0) > 0:
                export_data.append({
                    'Scenario': scenario_key,
                    'Patient_Group': group,
                    'Median_Census': sf[group].get('median_patients', 0),
                    'Min_Census': sf[group].get('min_patients', 0),
                    'Max_Census': sf[group].get('max_patients', 0),
                    'Census_SD': sf[group].get('sd_patients', 0),
                    'Annual_Admissions': sf[group].get('avg_yearly_admissions', 0),
                    'Total_Admissions': sf[group].get('total_admissions', 0),
                    'P10_LOS': sf[group].get('p10_los', 0),
                    'P25_LOS': sf[group].get('p25_los', 0),
                    'Median_LOS': sf[group].get('median_los', 0),
                    'P75_LOS': sf[group].get('p75_los', 0),
                    'P90_LOS': sf[group].get('p90_los', 0),
                    'P99_LOS': sf[group].get('p99_los', 0),
                    'IQR_LOS': sf[group].get('iqr_los', 0),
                    'SD_LOS': sf[group].get('sd_los', 0)
                })
    
    return pd.DataFrame(export_data)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_percentage_change(baseline: float, scenario: float) -> str:
    """Format percentage change with proper handling of zero values"""
    if baseline == 0:
        return "N/A"
    change = (scenario - baseline) / baseline * 100
    return f"{change:+.1f}%"


def get_scenario_description(scenario_key: str) -> str:
    """Get human-readable description for scenario key"""
    scenario_descriptions = {
        'baseline_all': 'Baseline: All Patients Model',
        'baseline_separate': 'Baseline: Separate Groups Model',
        '1a_non_ed_only': 'Scenario 1a: Non-ED Patients Only',
        '1b_ed_only': 'Scenario 1b: ED Patients Only',
    }
    
    # Handle dynamic scenario names
    if '2a_uniform' in scenario_key:
        change = scenario_key.split('_')[-1]
        return f'Scenario 2a: Uniform Volume Change {change}'
    elif '2b_ed' in scenario_key:
        change = scenario_key.split('_')[-1]
        return f'Scenario 2b: ED Volume Change {change}'
    elif '2c_non_ed' in scenario_key:
        change = scenario_key.split('_')[-1]
        return f'Scenario 2c: Non-ED Volume Change {change}'
    
    return scenario_descriptions.get(scenario_key, scenario_key)


def validate_results_structure(results: Dict) -> bool:
    """Validate that results have the expected structure"""
    try:
        if isinstance(results, dict) and 'summary_frame' in results:
            sf = results['summary_frame']
        elif hasattr(results, 'summary_frame'):
            sf = results.summary_frame
        else:
            return False
        
        # Check if summary_frame has expected structure
        if not isinstance(sf, dict):
            return False
        
        # Check for at least one patient group with expected metrics
        for group in sf.keys():
            if isinstance(sf[group], dict) and 'median_patients' in sf[group]:
                return True
        
        return False
        
    except Exception:
        return False