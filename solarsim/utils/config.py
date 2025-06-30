"""
Configuration management for SolarSim.

Contains the SystemConfig class for managing system parameters and settings.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import yaml
import json
from pathlib import Path

from .constants import (
    PV_TECHNOLOGIES, BATTERY_TECHNOLOGIES, ECONOMIC_PARAMETERS,
    HOUSEHOLD_LOAD_PROFILES, CLIMATE_ZONES
)
from .helpers import validate_coordinates, validate_positive, validate_range


@dataclass
class LocationConfig:
    """Configuration for geographic location."""
    latitude: float
    longitude: float
    altitude: float = 0.0  # meters above sea level
    timezone: str = 'UTC'
    climate_zone: str = 'temperate'
    
    def __post_init__(self):
        """Validate location parameters."""
        if not validate_coordinates(self.latitude, self.longitude):
            raise ValueError(f"Invalid coordinates: lat={self.latitude}, lon={self.longitude}")
        
        if self.climate_zone not in CLIMATE_ZONES:
            raise ValueError(f"Unknown climate zone: {self.climate_zone}")


@dataclass 
class PVConfig:
    """Configuration for photovoltaic array."""
    technology: str = 'monocrystalline'
    capacity_kw: float = 5.0
    panel_power_w: float = 400
    efficiency: Optional[float] = None  # If None, use technology default
    temperature_coefficient: Optional[float] = None
    degradation_rate: Optional[float] = None
    tilt_angle: float = 30.0  # degrees
    azimuth_angle: float = 180.0  # degrees (south-facing)
    mounting_type: str = 'roof'  # 'roof', 'ground', 'pole'
    tracking: bool = False
    dc_ac_ratio: float = 1.2
    
    def __post_init__(self):
        """Validate and set default PV parameters."""
        if self.technology not in PV_TECHNOLOGIES:
            raise ValueError(f"Unknown PV technology: {self.technology}")
        
        validate_positive(self.capacity_kw, "PV capacity")
        validate_positive(self.panel_power_w, "Panel power")
        validate_range(self.tilt_angle, 0, 90, "Tilt angle")
        validate_range(self.azimuth_angle, 0, 360, "Azimuth angle")
        validate_range(self.dc_ac_ratio, 1.0, 2.0, "DC/AC ratio")
        
        # Set defaults from technology if not specified
        tech_params = PV_TECHNOLOGIES[self.technology]
        if self.efficiency is None:
            # Use middle of efficiency range
            eff_range = tech_params['efficiency_range']
            self.efficiency = (eff_range[0] + eff_range[1]) / 2
        
        if self.temperature_coefficient is None:
            self.temperature_coefficient = tech_params['temperature_coefficient']
            
        if self.degradation_rate is None:
            self.degradation_rate = tech_params['degradation_rate']
    
    @property
    def num_panels(self) -> int:
        """Calculate number of panels."""
        return int(self.capacity_kw * 1000 / self.panel_power_w)
    
    @property
    def array_area_m2(self) -> float:
        """Calculate array area in m²."""
        # Assume ~2 m² per panel (rough estimate)
        return self.num_panels * 2.0


@dataclass
class BatteryConfig:
    """Configuration for battery energy storage system."""
    technology: str = 'lifepo4'
    capacity_kwh: float = 10.0
    voltage_system: float = 48.0  # V
    round_trip_efficiency: Optional[float] = None
    cycle_life: Optional[int] = None
    calendar_life: Optional[int] = None
    dod_max: Optional[float] = None
    self_discharge_rate: Optional[float] = None
    thermal_management: bool = True
    
    def __post_init__(self):
        """Validate and set default battery parameters."""
        if self.technology not in BATTERY_TECHNOLOGIES:
            raise ValueError(f"Unknown battery technology: {self.technology}")
        
        validate_positive(self.capacity_kwh, "Battery capacity")
        validate_positive(self.voltage_system, "System voltage")
        
        # Set defaults from technology if not specified
        tech_params = BATTERY_TECHNOLOGIES[self.technology]
        if self.round_trip_efficiency is None:
            self.round_trip_efficiency = tech_params['round_trip_efficiency']
        
        if self.cycle_life is None:
            self.cycle_life = tech_params['cycle_life']
            
        if self.calendar_life is None:
            self.calendar_life = tech_params['calendar_life']
            
        if self.dod_max is None:
            self.dod_max = tech_params['dod_max']
            
        if self.self_discharge_rate is None:
            self.self_discharge_rate = tech_params['self_discharge_rate']
    
    @property
    def usable_capacity_kwh(self) -> float:
        """Calculate usable battery capacity."""
        return self.capacity_kwh * (self.dod_max or 0.9)


@dataclass
class InverterConfig:
    """Configuration for power inverter."""
    capacity_kw: float = 5.0
    efficiency_rated: float = 0.96
    voltage_input: float = 48.0  # V DC
    voltage_output: float = 240.0  # V AC
    frequency: float = 60.0  # Hz
    mppt_channels: int = 2
    
    def __post_init__(self):
        """Validate inverter parameters."""
        validate_positive(self.capacity_kw, "Inverter capacity")
        validate_range(self.efficiency_rated, 0.8, 0.99, "Inverter efficiency")
        validate_positive(self.voltage_input, "Input voltage")
        validate_positive(self.voltage_output, "Output voltage")
        validate_range(self.frequency, 50, 60, "Frequency")


@dataclass
class LoadConfig:
    """Configuration for electrical loads."""
    household_type: str = 'medium'
    daily_energy_kwh: Optional[float] = None
    peak_power_kw: Optional[float] = None
    base_load_kw: Optional[float] = None
    seasonal_variation: float = 0.2  # ±20% seasonal variation
    weekend_factor: float = 1.1  # 10% higher on weekends
    growth_rate: float = 0.02  # 2% annual growth
    critical_load_fraction: float = 0.3  # 30% critical loads
    
    def __post_init__(self):
        """Validate and set default load parameters."""
        if self.household_type not in HOUSEHOLD_LOAD_PROFILES:
            raise ValueError(f"Unknown household type: {self.household_type}")
        
        validate_range(self.seasonal_variation, 0, 1, "Seasonal variation")
        validate_range(self.weekend_factor, 0.5, 2.0, "Weekend factor")
        validate_range(self.growth_rate, -0.1, 0.1, "Growth rate")
        validate_range(self.critical_load_fraction, 0, 1, "Critical load fraction")
        
        # Set defaults from household type if not specified
        profile_params = HOUSEHOLD_LOAD_PROFILES[self.household_type]
        if self.daily_energy_kwh is None:
            # Use middle of energy range
            energy_range = profile_params['daily_energy']
            self.daily_energy_kwh = (energy_range[0] + energy_range[1]) / 2
        
        if self.peak_power_kw is None:
            # Use middle of power range
            power_range = profile_params['peak_power']
            self.peak_power_kw = (power_range[0] + power_range[1]) / 2
            
        if self.base_load_kw is None:
            self.base_load_kw = profile_params['base_load']


@dataclass
class EconomicConfig:
    """Configuration for economic analysis."""
    discount_rate: float = 0.06
    inflation_rate: float = 0.025
    electricity_escalation: float = 0.03
    system_lifetime: int = 25
    federal_tax_credit: float = 0.30
    state_incentives: float = 0.0
    net_metering: bool = False
    export_rate_multiplier: float = 1.0  # Relative to retail rate
    financing_type: str = 'cash'  # 'cash', 'loan', 'ppa'
    loan_rate: float = 0.05
    loan_term: int = 15
    
    def __post_init__(self):
        """Validate economic parameters."""
        validate_range(self.discount_rate, 0, 0.2, "Discount rate")
        validate_range(self.inflation_rate, -0.05, 0.1, "Inflation rate")
        validate_range(self.electricity_escalation, -0.1, 0.1, "Electricity escalation")
        validate_range(self.system_lifetime, 10, 50, "System lifetime")
        validate_range(self.federal_tax_credit, 0, 1, "Federal tax credit")
        validate_range(self.state_incentives, 0, 1, "State incentives")
        validate_range(self.export_rate_multiplier, 0, 2, "Export rate multiplier")
        
        if self.financing_type not in ['cash', 'loan', 'ppa']:
            raise ValueError(f"Unknown financing type: {self.financing_type}")


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    start_date: str = '2024-01-01'
    years: int = 1
    time_step_minutes: int = 15
    weather_uncertainty: bool = True
    load_uncertainty: bool = True
    component_failures: bool = False
    monte_carlo_runs: int = 1
    optimization_enabled: bool = False
    
    def __post_init__(self):
        """Validate simulation parameters."""
        validate_range(self.years, 1, 50, "Simulation years")
        validate_range(self.time_step_minutes, 1, 60, "Time step")
        validate_range(self.monte_carlo_runs, 1, 10000, "Monte Carlo runs")


@dataclass
class SystemConfig:
    """Complete system configuration for SolarSim."""
    location: LocationConfig = field(default_factory=lambda: LocationConfig(35.2271, -80.8431))
    pv: PVConfig = field(default_factory=PVConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    inverter: InverterConfig = field(default_factory=InverterConfig)
    load: LoadConfig = field(default_factory=LoadConfig)
    economic: EconomicConfig = field(default_factory=EconomicConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    
    def __post_init__(self):
        """Validate overall system configuration."""
        # Check compatibility between components
        if self.inverter.capacity_kw < self.pv.capacity_kw / self.pv.dc_ac_ratio:
            raise ValueError("Inverter capacity too small for PV array")
        
        if self.battery.voltage_system != self.inverter.voltage_input:
            raise ValueError("Battery and inverter voltage mismatch")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
        """Create SystemConfig from dictionary."""
        return cls(
            location=LocationConfig(**config_dict.get('location', {})),
            pv=PVConfig(**config_dict.get('pv', {})),
            battery=BatteryConfig(**config_dict.get('battery', {})),
            inverter=InverterConfig(**config_dict.get('inverter', {})),
            load=LoadConfig(**config_dict.get('load', {})),
            economic=EconomicConfig(**config_dict.get('economic', {})),
            simulation=SimulationConfig(**config_dict.get('simulation', {}))
        )
    
    @classmethod
    def from_yaml(cls, file_path: str) -> 'SystemConfig':
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, file_path: str) -> 'SystemConfig':
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert SystemConfig to dictionary."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dict__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            return obj
        
        return dataclass_to_dict(self)
    
    def to_yaml(self, file_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_json(self, file_path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate(self) -> bool:
        """Comprehensive validation of configuration."""
        try:
            # All validation is done in __post_init__ methods
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def summary(self) -> str:
        """Generate a summary of the system configuration."""
        return f"""
SolarSim System Configuration Summary
===================================

Location: {self.location.latitude:.3f}°N, {self.location.longitude:.3f}°W
Climate Zone: {self.location.climate_zone}
Altitude: {self.location.altitude}m

PV System:
- Technology: {self.pv.technology}
- Capacity: {self.pv.capacity_kw:.1f} kW
- Panels: {self.pv.num_panels} × {self.pv.panel_power_w}W
 - Efficiency: {(self.pv.efficiency or 0.2)*100:.1f}%
- Tilt/Azimuth: {self.pv.tilt_angle}°/{self.pv.azimuth_angle}°

Battery Storage:
- Technology: {self.battery.technology}
- Capacity: {self.battery.capacity_kwh:.1f} kWh
- Usable: {self.battery.usable_capacity_kwh:.1f} kWh
 - Efficiency: {(self.battery.round_trip_efficiency or 0.9)*100:.1f}%

Inverter:
- Capacity: {self.inverter.capacity_kw:.1f} kW
- Efficiency: {self.inverter.efficiency_rated*100:.1f}%

Load Profile:
- Household Type: {self.load.household_type}
- Daily Energy: {self.load.daily_energy_kwh:.1f} kWh/day
- Peak Power: {self.load.peak_power_kw:.1f} kW

Economic:
- System Lifetime: {self.economic.system_lifetime} years
- Discount Rate: {self.economic.discount_rate*100:.1f}%
- Federal Tax Credit: {self.economic.federal_tax_credit*100:.0f}%

Simulation:
- Duration: {self.simulation.years} year(s)
- Time Step: {self.simulation.time_step_minutes} minutes
- Monte Carlo Runs: {self.simulation.monte_carlo_runs}
"""


def create_default_config() -> SystemConfig:
    """Create a default system configuration."""
    return SystemConfig()


def create_small_system_config() -> SystemConfig:
    """Create configuration for a small residential system."""
    return SystemConfig(
        pv=PVConfig(capacity_kw=3.0),
        battery=BatteryConfig(capacity_kwh=6.0),
        inverter=InverterConfig(capacity_kw=3.0),
        load=LoadConfig(household_type='small')
    )


def create_large_system_config() -> SystemConfig:
    """Create configuration for a large residential system."""
    return SystemConfig(
        pv=PVConfig(capacity_kw=15.0, technology='hjt'),
        battery=BatteryConfig(capacity_kwh=30.0),
        inverter=InverterConfig(capacity_kw=12.0),
        load=LoadConfig(household_type='large')
    ) 