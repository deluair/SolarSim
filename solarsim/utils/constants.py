"""
Physical constants and system parameters for SolarSim.
"""

import numpy as np

# Physical Constants
STANDARD_TEST_CONDITIONS_IRRADIANCE = 1000  # W/m²
STANDARD_TEST_CONDITIONS_TEMPERATURE = 25   # °C
STEFAN_BOLTZMANN_CONSTANT = 5.67e-8        # W/(m²·K⁴)
SPECIFIC_HEAT_AIR = 1005                   # J/(kg·K)
AIR_DENSITY = 1.225                        # kg/m³

# Solar Constants
SOLAR_CONSTANT = 1366                      # W/m²
EARTH_SUN_DISTANCE_AU = 1.0               # Astronomical units

# PV Technology Parameters
PV_TECHNOLOGIES = {
    'monocrystalline': {
        'efficiency_range': (0.15, 0.24),
        'temperature_coefficient': -0.004,  # %/°C
        'degradation_rate': 0.005,          # %/year
        'cost_per_watt': 0.45              # $/W (2025)
    },
    'n_type_topcon': {
        'efficiency_range': (0.22, 0.23),
        'temperature_coefficient': -0.0035,
        'degradation_rate': 0.0055,
        'cost_per_watt': 0.55
    },
    'hjt': {
        'efficiency_range': (0.21, 0.23),
        'temperature_coefficient': -0.0025,
        'degradation_rate': 0.0025,
        'cost_per_watt': 0.60
    }
}

# Battery Technology Parameters
BATTERY_TECHNOLOGIES = {
    'lifepo4': {
        'energy_density': 160,              # Wh/kg
        'cycle_life': 6500,                 # cycles
        'round_trip_efficiency': 0.95,
        'calendar_life': 15,                # years
        'dod_max': 0.95,                   # depth of discharge
        'cost_per_kwh': 200,               # $/kWh (2025)
        'self_discharge_rate': 0.003       # %/day
    },
    'li_ion_nmc': {
        'energy_density': 250,
        'cycle_life': 4000,
        'round_trip_efficiency': 0.92,
        'calendar_life': 12,
        'dod_max': 0.90,
        'cost_per_kwh': 175,
        'self_discharge_rate': 0.005
    },
    'lead_acid': {
        'energy_density': 35,
        'cycle_life': 800,
        'round_trip_efficiency': 0.80,
        'calendar_life': 8,
        'dod_max': 0.50,
        'cost_per_kwh': 100,
        'self_discharge_rate': 0.01
    }
}

# Inverter Parameters
INVERTER_EFFICIENCY_CURVE = {
    'load_fraction': np.array([0.1, 0.2, 0.3, 0.5, 0.75, 1.0]),
    'efficiency': np.array([0.90, 0.94, 0.96, 0.97, 0.965, 0.96])
}

# Economic Parameters (2025 baseline)
ECONOMIC_PARAMETERS = {
    'federal_tax_credit': 0.30,            # 30% through 2032
    'discount_rate': 0.06,                 # 6% default
    'inflation_rate': 0.025,               # 2.5% annual
    'electricity_escalation': 0.03,        # 3% annual
    'system_lifetime': 25,                 # years
    'o_m_cost_annual': 0.01,              # 1% of system cost annually
}

# Load Profile Parameters
HOUSEHOLD_LOAD_PROFILES = {
    'small': {
        'daily_energy': (2, 3),            # kWh/day range
        'peak_power': (0.8, 1.5),          # kW range
        'base_load': 0.2                   # kW
    },
    'medium': {
        'daily_energy': (8, 12),
        'peak_power': (2.5, 4.5),
        'base_load': 0.5
    },
    'large': {
        'daily_energy': (15, 25),
        'peak_power': (5.0, 8.0),
        'base_load': 1.0
    }
}

# Weather Parameters
CLIMATE_ZONES = {
    'desert': {
        'annual_ghi': (2000, 2500),        # kWh/m²/year
        'temp_range': (-5, 45),            # °C
        'humidity_range': (10, 40),        # %
        'wind_speed_avg': 3.5              # m/s
    },
    'temperate': {
        'annual_ghi': (1200, 1800),
        'temp_range': (-10, 35),
        'humidity_range': (40, 80),
        'wind_speed_avg': 4.0
    },
    'tropical': {
        'annual_ghi': (1500, 2000),
        'temp_range': (15, 35),
        'humidity_range': (60, 95),
        'wind_speed_avg': 2.5
    },
    'cold': {
        'annual_ghi': (800, 1400),
        'temp_range': (-25, 25),
        'humidity_range': (30, 70),
        'wind_speed_avg': 5.0
    }
}

# System Sizing Guidelines
SIZING_RULES = {
    'battery_to_load_ratio': (1.5, 3.0),   # Days of autonomy
    'pv_to_load_ratio': (1.2, 2.0),        # Oversizing factor
    'inverter_to_load_ratio': (1.25, 1.5), # Capacity factor
}

# Time Constants
SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365.25
MINUTES_PER_DAY = 1440

# Data Resolution
TIME_RESOLUTION_MINUTES = 15    # Default simulation time step
WEATHER_RESOLUTION_HOURS = 1    # Weather data resolution
LOAD_RESOLUTION_MINUTES = 15    # Load data resolution 