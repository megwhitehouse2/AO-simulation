# Assertive Outreach Service Simulation Model

## Technical Summary

This discrete event simulation (DES) models patient flow through an Assertive Outreach (AO) service that provides intensive community support wrapping around core CAMHS treatments. The model was developed in collaboration with the AO Lead Psychiatrist to inform service planning and resource allocation decisions.

### Key Choices in Implementing the service include:

- **Execution of two Scenarios**: 
  - "All Patients" - analyses the entire caseload as a single distribution
  - "Separate Groups" - models ED and non-ED patients with distinct characteristics
  
- **Efficient Event Processing**: Uses next-event time advancement for computational efficiency, jumping between patient events rather than fixed time intervals

- **Data-Driven Parameters**: Based on analysis of 119 historical patients (2021-2023) with validated statistical distributions

Overview
A discrete event simulation (DES) model for analysing patient flow through an Assertive Outreach (AO) service providing intensive community support for CAMHS patients. The model uses advanced statistical techniques to capture complex patient dynamics and inform service planning decisions.

Key Innovation: Single Arrival Stream Architecture
The model implements a single CAMHS referral stream that subsequently classifies patients into ED/non-ED categories using a Bernoulli distribution (p=0.559 for ED). This design reflects the clinical reality where all patients arrive through the same referral pathway, with ED status determined post-referral. This approach differs from traditional multi-stream DES models and ensures realistic patient mix proportions.


# Quick Start
Installation
bash# Clone repository
git clone https://github.com/yourusername/ao-service-simulation.git
cd ao-service-simulation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Running the Simulation
bash# Command line
python run_simulation.py --scenario baseline --replications 30

# Streamlit interface
streamlit run app.py
📊 Technical Architecture
Core Components
1. Event Scheduling Engine (simulation_engine.py)

SimPy Environment: Manages discrete event scheduling with next-event time advancement
Efficiency: O(log n) event insertion/removal using heap-based priority queue
Time scales: Handles 21-869 day patient journeys without fixed time slicing

2. Statistical Distributions (simulation_core.py)
Inter-Arrival Times (IAT):
python# Truncated Erlang Distribution
Shape: {all: 0.93, ed: 1.11, non_ed: 2.21}
Location: {all: 0.86, ed: 0.0, non_ed: 0.0}
Scale: {all: 6.5, ed: 8.0, non_ed: 7.5}
Bounds: [0.01, 40.0] days
Length of Stay (LOS):
python# Mixed Distribution Model
if LOS <= threshold:
    # Lognormal component
    LOS ~ Lognormal(μ, σ)
else:
    # Generalized Pareto Distribution
    LOS ~ threshold + GPD(shape, scale)

# Parameters by group:
ED: threshold=200, μ=4.64, σ=0.46, shape=0.213, scale=125.3
Non-ED: threshold=220, μ=4.5, σ=0.58, shape=0.095, scale=130.0
3. Patient Flow Logic
mermaidgraph LR
    A[CAMHS Referral] --> B{Classify Patient}
    B -->|55.9%| C[ED Patient]
    B -->|44.1%| D[Non-ED Patient]
    C --> E{Referral Type}
    D --> E
    E -->|39.6%| F[Urgent: 1-day delay]
    E -->|60.4%| G[Routine: 5-day delay]
    F --> H[Enter Service]
    G --> H
    H --> I[Stay ~LOS days]
    I --> J[Discharge]
Random Number Generation Architecture



python# Hierarchical seeding structure
base_seed = 42
├── scenario_seed = base_seed + replication_number
    ├── patient_generator_seed = scenario_seed + 100
    ├── classification_rng_seed = scenario_seed + 999
    └── distribution_seeds
        ├── ed_los_seed = patient_id * 100 + scenario_seed + 101
        ├── non_ed_los_seed = patient_id * 100 + scenario_seed + 102
        └── iat_seed = rng.randint(1, 10000)
This ensures:

Reproducibility: Same seed → same results
Independence: Changes in one component don't affect others
Scenario isolation: Different scenarios have non-overlapping random streams

Performance Optimizations

Memory Management

Pre-allocated numpy arrays for large result sets
Efficient pandas DataFrame concatenation using list accumulation
Filtered occupancy tracking (excludes periods <5 patients)


Computational Efficiency

Next-event scheduling: O(1) time advancement vs O(n) for time-slicing
Vectorized distribution sampling for batch operations
Lazy evaluation of expensive percentile calculations


Statistical Validity

Warm-up period: 365 days (ensures steady-state)
Observation interval: 30 days (balances accuracy vs memory)
Replication management: 10-30 runs with independent streams



📈 Model Outputs
Primary Metrics
Metric CategoryMeasurementsCalculation MethodOccupancyMin, Max, Median, SDPoint-in-time observations every 30 daysAdmissionsAnnual rate, TotalCounted at entry, aggregated yearlyLength of Stayp10, p25, p50, p75, p90, p99Empirical percentiles from completed staysPatient MixED %, Non-ED %Proportion of admissions by type
Scenario Configurations
python# Baseline
scenario = Scenario(
    fitted_patient_types=create_distributions(),
    separate_patient_groups=False,
    random_seed=42
)

# Differential ED increase (+30%)
total_multiplier = 1.0997  # Calculated via mathematical model
new_ed_probability = 0.6115  # Adjusted from baseline 0.559
scenario = create_differential_scenario(
    target_group='ed_patients',
    volume_change=0.30,
    iat_multiplier=1/total_multiplier
)
🔧 Advanced Configuration
Custom Distribution Parameters
python# Override default distributions
custom_params = DISTRIBUTION_PARAMS.copy()
custom_params['los']['ed_patients']['lognormal']['mu'] = 5.0
scenario = Scenario(
    fitted_patient_types=create_distributions(custom_params),
    separate_patient_groups=True
)
Exclusive Service Scenarios
python# ED-only service (Scenario 1b)
generator = ExclusivePatientGenerator(
    env, scenario, auditor, 
    exclusive_type='ed_only'
)
# IAT automatically adjusted by 1/0.559 to maintain arrival rate
Independent Streams (Validation Only)
python# For methodological comparison
generator = IndependentStreamsGenerator(env, scenario, auditor)
# Creates separate RNG streams for ED/non-ED arrivals
📝 Validation & Testing
Unit Tests
bashpytest tests/ -v --cov=simulation_core --cov=simulation_engine
Statistical Validation

Chi-square tests for distribution fitting
Kolmogorov-Smirnov tests for empirical vs theoretical CDFs
Welch's t-test for scenario comparisons

Historical Validation
MetricHistoricalSimulatedDifferenceAnnual Arrivals50.550.3 ± 2.1-0.4%ED Proportion55.9%55.7% ± 1.2%-0.2%Median LOS (ED)141.5142.3 ± 3.4+0.6%
🚀 Deployment
Streamlit Configuration
python# .streamlit/config.toml
[theme]
primaryColor = "#0E4C92"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"

[server]
maxUploadSize = 200
enableCORS = false
Docker Deployment
dockerfileFROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
📚 Mathematical Foundations
Little's Law Application
L = λW
where:
L = Average number in system (occupancy)
λ = Arrival rate
W = Average time in system (LOS)

Example: 50.5 arrivals/year × 133 days / 365 = 18.4 expected occupancy
Differential Scenario Calculations
For increasing ED by X% while keeping non-ED constant:
new_total_rate = baseline_rate × (1 + X × p_ed)
new_ed_probability = (1 + X) × p_ed / (1 + X × p_ed)

## Core Components

### 1. **Entities (Patients)**
Patients flow through the system with key attributes:
- ED status (55.9% ED, 44.1% non-ED based on historical data)
- Referral type (urgent: 1-day delay, routine: 5-day delay)
- Group-specific length of stay patterns

### 2. **Environment**
SimPy-based event scheduler that:
- Manages patient arrivals, admissions, and discharges
- Advances simulation time only when events occur
- Handles time scales from 3 weeks to 3+ years

### 3. **Service Model**
Unlike traditional DES models, the AO service has:
- No explicit capacity constraints (anecdotal maximum ~30 patients)
- Independent arrivals and departures
- No queueing mechanism

### 4. **Statistical Distributions**
- **Inter-arrival times**: Erlang distributions with a single arrival stream: CAMHS outpatient referrals
- **Length of stay**: Mixed distribution
  - Lognormal for stays ≤200 days
  - Generalized Pareto Distribution for extended stays >200 days

## Key Assumptions

### Process
- Patient arrivals occur independently of system state
- Service capacity determined by CAMHS referral rate
- Uniform resource requirements across patient types

### Statistical
- ED classification: 55.9% (Bernoulli distribution)
- Urgent referrals: 39.6% of arrivals
- No seasonal patterns modeled (insufficient data)

### Simplifications
- No care state transitions or readmissions
- Constant resource utilization per patient
- Limited to ED status and referral type attributes

## Technical Implementation

### Software Requirements
- Python 3.11+
- Core libraries: NumPy, Pandas, Matplotlib
- SimPy for discrete event simulation
- SciPy for statistical distributions

### Random Number Generation
Hierarchical seeding ensures reproducibility:
- Base seed (default: 42) generates derived seeds
- Independent random streams for ED/non-ED in separate group scenarios
- Each distribution maintains its own RNG for statistical independence

### Model Validation
- Validated against 2022-2023 historical data
- Median occupancy: 24.3 patients (baseline)
- Annual arrivals: ~50.5 patients/year
- Service operates at 81-100% of anecdotal capacity

## Usage

The model supports various scenario analyses:
1. **Baseline**: Current service operation
2. **Volume changes**: ±10-30% in specific patient groups
3. **Exclusive services**: ED-only or non-ED-only configurations
4. **Differential impacts**: Targeted changes in one group while holding the other constant
