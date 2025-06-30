"""
SolarSim: Comprehensive Off-Grid Solar Simulation Framework

A Python-based simulation framework for modeling complete off-grid residential 
solar photovoltaic systems integrated with Battery Energy Storage Systems (BESS).

This framework encompasses the full spectrum of technical, economic, and 
operational considerations affecting system performance, financial viability, 
and energy reliability across diverse scenarios.

Main Components:
- Solar PV arrays with multiple technologies
- Battery systems with aging models  
- Power inverters with realistic efficiency curves
- Load management with appliance-level modeling
- Weather data generation
- Economic analysis and optimization

Example Usage:
    >>> from solarsim import SystemConfig, OffGridSolarSimulation
    >>> from solarsim.utils.config import create_default_config
    >>> 
    >>> # Create a default system configuration
    >>> config = create_default_config()
    >>> 
    >>> # Initialize and run simulation
    >>> sim = OffGridSolarSimulation(config)
    >>> results = sim.run_simulation(duration_years=1)
    >>> 
    >>> # Display results
    >>> print(sim.get_summary_report())
"""

__version__ = "1.0.0"
__author__ = "SolarSim Development Team"
__email__ = "info@solarsim.org"

# Core simulation engine
from .simulation import OffGridSolarSimulation

# Configuration management
from .utils.config import (
    SystemConfig,
    LocationConfig, 
    PVConfig,
    BatteryConfig,
    InverterConfig,
    LoadConfig,
    EconomicConfig,
    SimulationConfig,
    create_default_config,
    create_small_system_config,
    create_large_system_config
)

# Component models
from .components.solar import SolarPVArray
from .components.battery import BatterySystem
from .components.inverter import Inverter
from .components.loads import LoadManager

# Data generators
from .data.weather import WeatherGenerator
from .data.loads import LoadProfileGenerator
from .data.economics import EconomicDataGenerator

# Utility functions
from .utils.constants import (
    PV_TECHNOLOGIES,
    BATTERY_TECHNOLOGIES,
    HOUSEHOLD_LOAD_PROFILES,
    CLIMATE_ZONES
)
from .utils.helpers import (
    solar_position,
    air_mass,
    clear_sky_irradiance,
    create_time_index,
    validate_coordinates,
    validate_positive,
    validate_range
)

# Define public API
__all__ = [
    # Main simulation class
    'OffGridSolarSimulation',
    
    # Configuration classes
    'SystemConfig',
    'LocationConfig',
    'PVConfig', 
    'BatteryConfig',
    'InverterConfig',
    'LoadConfig',
    'EconomicConfig',
    'SimulationConfig',
    
    # Configuration factory functions
    'create_default_config',
    'create_small_system_config', 
    'create_large_system_config',
    
    # Component classes
    'SolarPVArray',
    'BatterySystem',
    'Inverter',
    'LoadManager',
    
    # Data generator classes
    'WeatherGenerator',
    'LoadProfileGenerator',
    'EconomicDataGenerator',
    
    # Constants and utilities
    'PV_TECHNOLOGIES',
    'BATTERY_TECHNOLOGIES', 
    'HOUSEHOLD_LOAD_PROFILES',
    'CLIMATE_ZONES',
    'solar_position',
    'air_mass',
    'irradiance_on_tilted_surface',
    'create_time_index',
    'validate_coordinates',
    'validate_positive',
    'validate_range'
]


def get_version():
    """Return the version string."""
    return __version__


def quick_simulation(
    pv_capacity_kw: float = 5.0,
    battery_capacity_kwh: float = 10.0,
    daily_load_kwh: float = 15.0,
    latitude: float = 35.0,
    longitude: float = -80.0,
    duration_months: int = 3
):
    """
    Run a quick simulation with basic parameters.
    
    Args:
        pv_capacity_kw: Solar array capacity in kW
        battery_capacity_kwh: Battery capacity in kWh  
        daily_load_kwh: Daily energy consumption in kWh
        latitude: Location latitude
        longitude: Location longitude
        duration_months: Simulation duration in months
        
    Returns:
        OffGridSolarSimulation instance with results
    """
    from .utils.config import (
        SystemConfig, LocationConfig, PVConfig, 
        BatteryConfig, InverterConfig, LoadConfig
    )
    
    # Create simple configuration
    location = LocationConfig(latitude=latitude, longitude=longitude)
    pv = PVConfig(capacity_kw=pv_capacity_kw)
    battery = BatteryConfig(capacity_kwh=battery_capacity_kwh)
    inverter = InverterConfig(capacity_kw=pv_capacity_kw)
    load = LoadConfig(daily_energy_kwh=daily_load_kwh)
    
    config = SystemConfig(
        location=location,
        pv=pv,
        battery=battery,
        inverter=inverter,
        load=load
    )
    
    # Run simulation
    sim = OffGridSolarSimulation(config)
    sim.run_simulation(duration_years=duration_months/12)
    
    return sim


def print_system_summary(config: SystemConfig):
    """Print a formatted summary of system configuration."""
    print("SolarSim System Configuration")
    print("=" * 40)
    print(config.summary())


def list_available_technologies():
    """List all available component technologies."""
    print("Available Technologies in SolarSim")
    print("=" * 40)
    
    print("\nPV Technologies:")
    for tech, params in PV_TECHNOLOGIES.items():
        eff_range = params['efficiency_range']
        print(f"  {tech}: {eff_range[0]*100:.1f}-{eff_range[1]*100:.1f}% efficiency")
    
    print("\nBattery Technologies:")
    for tech, params in BATTERY_TECHNOLOGIES.items():
        cycles = params['cycle_life']
        efficiency = params['round_trip_efficiency']
        print(f"  {tech}: {cycles:,} cycles, {efficiency*100:.0f}% efficiency")
    
    print("\nHousehold Types:")
    for household, params in HOUSEHOLD_LOAD_PROFILES.items():
        energy_range = params['daily_energy']
        print(f"  {household}: {energy_range[0]}-{energy_range[1]} kWh/day")
    
    print("\nClimate Zones:")
    for zone in CLIMATE_ZONES.keys():
        print(f"  {zone}")


# Package metadata
__doc_info__ = {
    'title': 'SolarSim',
    'description': 'Comprehensive Off-Grid Solar Simulation Framework',
    'version': __version__,
    'components': len(__all__),
    'technologies': {
        'pv': len(PV_TECHNOLOGIES),
        'battery': len(BATTERY_TECHNOLOGIES), 
        'household': len(HOUSEHOLD_LOAD_PROFILES),
        'climate': len(CLIMATE_ZONES)
    }
} 