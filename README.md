# Mental Health Service Simulation

A discrete event simulation model which utilises the 'SimPy' package to assess patient flow through Assertive Outreach (AO) Devon. Scenario analysis explores likely future demand, suggesting utility for this model as a decision support tool. 

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/megwhitehouse2/AO-simulation.git
cd AO-simulation

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

### Access
- **Local**: http://localhost:8501
- **Production**: https://ao-simulation.streamlit.app

## Overview

This simulation models patient flow through Assessment and Outcome (AO) services that wrap around core CAMHS treatments. The model uses calibrated statistical distributions based on historical data (2021-2023) to support:

- **Service capacity planning** under varying demand scenarios
- **Resource allocation optimization** between patient groups
- **Policy analysis** for service inclusion/exclusion criteria
- **Impact assessment** of demographic and referral pattern changes

## Patient Flow: Scenario 1- Separate Groups

Step 1: CAMHS Referrals as IAT
   │
   └→ Step 2: Patient Classification
      ├→ ED Patient (55.9%)
      └→ Non-ED Patient (44.1%)
         │
         └→ Step 3: Referral Type Determination
            ├→ Urgent (39.6%) → 1-day delay
            └→ Routine (60.4%) → 5-day delay
               │
               └→ Step 4: Service Entry
                  │
                  └→ Step 5: Sample from respective LOS
                     ├→ ED patients
                     └→ Non-ED patients: 
                        │
                        └→ Step 6: Discharge


## Patient Flow: Scenario 2- All Patients

Step 1: CAMHS Referral
   │
   └→ Step 2: All Patients (Single Group)
      │ No ED vs Non-ED classification
      │ Uses unified statistical distributions
      │
      └→ Step 3: Referral Type Determination
         ├→ Urgent (39.6%) → 1-day delay
         └→ Routine (60.4%) → 5-day delay
            │
            └→ Step 4: Service Entry
               │
               └→ Step 5: Length of Stay
                  │ All patients: 
                  │ (Single distribution parameters)
                  │
                  └→ Step 6: Discharge


## Key Features

- **Interactive Interface**: Streamlit-based parameter configuration
- **Scenario Analysis**: Baseline vs. scenario comparisons
- **Statistical Rigor**: 30+ replications with reproducible random seeds
- **Export Functionality**: CSV output for further analysis
- **Real-time Results**: Fast discrete event simulation engine

## Scenarios

### 1. Service Inclusion Analysis
- **1a. Non-ED Only**: Service dynamics without ED patients
- **1b. ED Only**: Service functioning with ED patients exclusively

### 2. Volume Change Analysis  
- **2a. Uniform Changes**: ±10% to ±30% across all referrals
- **2b. ED-Focused**: Changes in ED referrals, Non-ED constant
- **2c. Non-ED-Focused**: Changes in Non-ED referrals, ED constant

## Usage

1. **Configure Parameters**: Set warm-up period, run length, and replications
2. **Select Scenario**: Choose from predefined scenario types
3. **Set Parameters**: Configure scenario-specific settings
4. **Run Analysis**: Execute simulation and review results
5. **Export Data**: Download results as CSV for further analysis

## Model Validation

- **Historical Accuracy**: Validated against 2021-2023 service data
- **Statistical Validation**: Chi-square and KS tests for distribution fitting
- **Performance Metrics**: 5-10% accuracy thresholds met on most primary metrics
- **Clinical Validation**: Reviewed with AO Lead Psychiatrist

## Key Metrics

| Category | Measurements |
|----------|-------------|
| **Occupancy** | Min, Max, Median, Standard Deviation |
| **Admissions** | Annual rate, Total admissions |
| **Length of Stay** | 10th, 25th, 50th, 75th, 90th, 99th percentiles |
| **Patient Mix** | ED vs Non-ED proportions |

## Technical Highlights

- **SimPy-based** discrete event simulation
- **Mixed distributions** for realistic Length of Stay modeling
- **Hierarchical seeding** for reproducible results
- **Memory-efficient** event processing
- **Calibrated parameters** from maximum likelihood estimation

## File Structure

```
AO-simulation/
├── streamlit_app.py          # Main Streamlit interface
├── simulation_core.py        # Core simulation classes
├── simulation_engine.py      # Patient generators and execution
├── requirements.txt          # Dependencies
├── README.md                # This file
└── docs/                    # Technical documentation
    ├── technical_details.md
    
```


## Function Documentation

All original docstrings and comments have been preserved from the research implementation, including:

- **Distribution parameters**: Detailed calibration notes and empirical justifications
- **Method explanations**: Original algorithm descriptions and statistical rationale
- **Private methods**: Internal functions marked with leading underscore (`_method_name`) for class encapsulation
- **Clinical context**: References to AO service operations and community CAMHS integration


## Contact

**Author**: Megan Whitehouse 
**Institution**: University of Exeter  
**Email**: mw813@exeter.ac.uk


## Acknowledgments

-  The Devon Assertive outreach team for providing the historical service data, as well as clinical context for the project.

---

**Note**: This simulation models a specific outreach service in Devon that wraps around community CAMHS treatments. The model is a simplified representation of current operating conditions, and lacks much of the clinically meaningful data. The outputs may not be representative of patients generally seen in either CAMHS, or AO services, as the model is based on a small sample of complex patients. This service is distinct from most traditional DES models due to an unconstrained capacity, where patients in the service do not use up 'pooled' resources. A key assumption is LOS as a proxy for clinical requirement, when this relationship is unlikely to be linear. Therefore, outputs should only be used as a guide for decision making, with more emphasis being on individual patient needs and local service constraints.
