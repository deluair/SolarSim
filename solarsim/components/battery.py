"""
Battery Energy Storage System (BESS) component for SolarSim.

Contains the BatterySystem class for modeling battery performance,
including various battery technologies, aging, and thermal effects.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import math

from ..utils.constants import BATTERY_TECHNOLOGIES
from ..utils.helpers import battery_thermal_model


class BatterySystem:
    """
    Battery Energy Storage System model with realistic performance characteristics.
    
    Supports multiple battery technologies with aging models, thermal effects,
    and depth-of-discharge limitations.
    """
    
    def __init__(self, config):
        """
        Initialize battery system.
        
        Args:
            config: BatteryConfig object with battery parameters
        """
        self.config = config
        self.technology = config.technology
        self.nominal_capacity_kwh = config.capacity_kwh
        self.voltage_system = config.voltage_system
        self.round_trip_efficiency = config.round_trip_efficiency
        self.cycle_life = config.cycle_life
        self.calendar_life = config.calendar_life
        self.dod_max = config.dod_max
        self.self_discharge_rate = config.self_discharge_rate
        self.thermal_management = config.thermal_management
        
        # State variables
        self.current_capacity_kwh = self.nominal_capacity_kwh
        self.current_soc = 0.5  # Start at 50% SOC
        self.temperature = 25.0  # °C
        self.cycle_count = 0
        self.equivalent_full_cycles = 0.0
        self.total_energy_charged = 0.0  # kWh
        self.total_energy_discharged = 0.0  # kWh
        
        # Aging tracking
        self.installation_date = None
        self.capacity_fade_factor = 1.0
        self.last_soc = self.current_soc
        
    def get_usable_capacity(self) -> float:
        """
        Get current usable battery capacity considering degradation.
        
        Returns:
            Usable capacity in kWh
        """
        return self.current_capacity_kwh * self.dod_max
    
    def get_energy_stored(self) -> float:
        """
        Get current energy stored in battery.
        
        Returns:
            Energy stored in kWh
        """
        return self.current_capacity_kwh * self.current_soc
    
    def get_available_charge_capacity(self) -> float:
        """
        Get available charging capacity.
        
        Returns:
            Available capacity for charging in kWh
        """
        max_energy = self.current_capacity_kwh * self.dod_max
        current_energy = self.get_energy_stored()
        return max(0, max_energy - current_energy)
    
    def get_available_discharge_capacity(self) -> float:
        """
        Get available discharging capacity.
        
        Returns:
            Available capacity for discharging in kWh
        """
        min_soc = 1.0 - self.dod_max
        min_energy = self.current_capacity_kwh * min_soc
        current_energy = self.get_energy_stored()
        return max(0, current_energy - min_energy)
    
    def calculate_charge_power(self, 
                             desired_power_kw: float,
                             ambient_temp: float = 25.0,
                             time_step_hours: float = 0.25) -> Dict[str, float]:
        """
        Calculate actual charging power considering constraints.
        
        Args:
            desired_power_kw: Desired charging power in kW
            ambient_temp: Ambient temperature in °C
            time_step_hours: Time step duration in hours
            
        Returns:
            Dictionary with charging results
        """
        if desired_power_kw <= 0:
            return {
                'actual_power_kw': 0,
                'energy_charged_kwh': 0,
                'new_soc': self.current_soc,
                'efficiency_loss': 0,
                'temperature_limit': False,
                'capacity_limit': False
            }
        
        # Calculate battery temperature
        self.temperature = battery_thermal_model(
            ambient_temp, desired_power_kw, 0, self.nominal_capacity_kwh
        )
        
        # Temperature-based power derating
        temp_factor = self._get_temperature_factor(self.temperature)
        max_charge_power = desired_power_kw * temp_factor
        
        # SOC-based charging curve (lower power near full charge)
        soc_factor = self._get_charge_soc_factor(self.current_soc)
        max_charge_power *= soc_factor
        
        # C-rate limitation (maximum charging rate)
        max_c_rate = self._get_max_charge_c_rate()
        max_power_c_rate = self.current_capacity_kwh * max_c_rate
        max_charge_power = min(max_charge_power, max_power_c_rate)
        
        # Available capacity limitation
        available_capacity = self.get_available_charge_capacity()
        max_energy = available_capacity / math.sqrt(self.round_trip_efficiency)  # Account for losses
        max_power_capacity = max_energy / time_step_hours
        max_charge_power = min(max_charge_power, max_power_capacity)
        
        # Final charging power
        actual_power_kw = min(desired_power_kw, max_charge_power)
        
        # Energy calculations
        energy_input_kwh = actual_power_kw * time_step_hours
        energy_stored_kwh = energy_input_kwh * math.sqrt(self.round_trip_efficiency)
        efficiency_loss = energy_input_kwh - energy_stored_kwh
        
        # Update SOC
        new_soc = self.current_soc + energy_stored_kwh / self.current_capacity_kwh
        new_soc = min(new_soc, self.dod_max)  # Enforce maximum SOC
        
        # Check constraints
        temperature_limit = temp_factor < 1.0
        capacity_limit = actual_power_kw < desired_power_kw
        
        return {
            'actual_power_kw': actual_power_kw,
            'energy_charged_kwh': energy_stored_kwh,
            'new_soc': new_soc,
            'efficiency_loss': efficiency_loss,
            'temperature_limit': temperature_limit,
            'capacity_limit': capacity_limit,
            'battery_temperature': self.temperature
        }
    
    def calculate_discharge_power(self,
                                desired_power_kw: float,
                                ambient_temp: float = 25.0,
                                time_step_hours: float = 0.25) -> Dict[str, float]:
        """
        Calculate actual discharging power considering constraints.
        
        Args:
            desired_power_kw: Desired discharging power in kW
            ambient_temp: Ambient temperature in °C
            time_step_hours: Time step duration in hours
            
        Returns:
            Dictionary with discharging results
        """
        if desired_power_kw <= 0:
            return {
                'actual_power_kw': 0,
                'energy_discharged_kwh': 0,
                'new_soc': self.current_soc,
                'efficiency_loss': 0,
                'temperature_limit': False,
                'capacity_limit': False
            }
        
        # Calculate battery temperature
        self.temperature = battery_thermal_model(
            ambient_temp, 0, desired_power_kw, self.nominal_capacity_kwh
        )
        
        # Temperature-based power derating
        temp_factor = self._get_temperature_factor(self.temperature)
        max_discharge_power = desired_power_kw * temp_factor
        
        # SOC-based discharging curve (lower power at low SOC)
        soc_factor = self._get_discharge_soc_factor(self.current_soc)
        max_discharge_power *= soc_factor
        
        # C-rate limitation
        max_c_rate = self._get_max_discharge_c_rate()
        max_power_c_rate = self.current_capacity_kwh * max_c_rate
        max_discharge_power = min(max_discharge_power, max_power_c_rate)
        
        # Available energy limitation
        available_energy = self.get_available_discharge_capacity()
        max_power_capacity = available_energy / time_step_hours
        max_discharge_power = min(max_discharge_power, max_power_capacity)
        
        # Final discharging power
        actual_power_kw = min(desired_power_kw, max_discharge_power)
        
        # Energy calculations
        energy_from_battery_kwh = actual_power_kw * time_step_hours / math.sqrt(self.round_trip_efficiency)
        energy_delivered_kwh = actual_power_kw * time_step_hours
        efficiency_loss = energy_from_battery_kwh - energy_delivered_kwh
        
        # Update SOC
        new_soc = self.current_soc - energy_from_battery_kwh / self.current_capacity_kwh
        min_soc = 1.0 - self.dod_max
        new_soc = max(new_soc, min_soc)  # Enforce minimum SOC
        
        # Check constraints
        temperature_limit = temp_factor < 1.0
        capacity_limit = actual_power_kw < desired_power_kw
        
        return {
            'actual_power_kw': actual_power_kw,
            'energy_discharged_kwh': energy_delivered_kwh,
            'new_soc': new_soc,
            'efficiency_loss': efficiency_loss,
            'temperature_limit': temperature_limit,
            'capacity_limit': capacity_limit,
            'battery_temperature': self.temperature
        }
    
    def apply_self_discharge(self, time_step_hours: float) -> float:
        """
        Apply self-discharge to battery.
        
        Args:
            time_step_hours: Time step duration in hours
            
        Returns:
            Energy lost to self-discharge in kWh
        """
        daily_rate = self.self_discharge_rate
        hourly_rate = daily_rate / 24
        
        energy_lost = self.get_energy_stored() * hourly_rate * time_step_hours
        
        # Update SOC
        new_soc = self.current_soc - energy_lost / self.current_capacity_kwh
        self.current_soc = max(new_soc, 0)
        
        return energy_lost
    
    def update_aging(self, time_step_hours: float, timestamp: Optional[pd.Timestamp] = None) -> None:
        """
        Update battery aging based on cycling and calendar aging.
        
        Args:
            time_step_hours: Time step duration in hours
            timestamp: Current timestamp for calendar aging
        """
        # Calculate cycle aging
        soc_change = abs(self.current_soc - self.last_soc)
        if soc_change > 0.01:  # Only count significant SOC changes
            cycle_fraction = soc_change / 2  # Full cycle = 100% SOC change
            self.equivalent_full_cycles += cycle_fraction
        
        # Cycle aging factor (simplified)
        cycle_aging_factor = 1 - (self.equivalent_full_cycles / self.cycle_life) * 0.2
        cycle_aging_factor = max(cycle_aging_factor, 0.8)  # Minimum 80% capacity
        
        # Calendar aging
        calendar_aging_factor = 1.0
        if timestamp and self.installation_date:
            years_operating = (timestamp - self.installation_date).days / 365.25
            calendar_aging_factor = 1 - (years_operating / self.calendar_life) * 0.2
            calendar_aging_factor = max(calendar_aging_factor, 0.8)
        
        # Combined aging factor (most restrictive)
        self.capacity_fade_factor = min(cycle_aging_factor, calendar_aging_factor)
        self.current_capacity_kwh = self.nominal_capacity_kwh * self.capacity_fade_factor
        
        # Update last SOC for next cycle calculation
        self.last_soc = self.current_soc
    
    def _get_temperature_factor(self, temperature: float) -> float:
        """Get temperature derating factor."""
        if self.technology == 'lifepo4':
            # LiFePO4 performs well at high temperatures
            if temperature < 0:
                return 0.7  # Reduced performance in freezing
            elif temperature > 45:
                return 0.9  # Slight reduction at high temps
            else:
                return 1.0
        elif self.technology == 'li_ion_nmc':
            # Li-ion NMC more temperature sensitive
            if temperature < 0:
                return 0.5
            elif temperature > 40:
                return 0.8
            else:
                return 1.0
        else:  # lead_acid
            # Lead-acid very temperature sensitive
            if temperature < 5:
                return 0.6
            elif temperature > 35:
                return 0.9
            else:
                return 1.0
    
    def _get_charge_soc_factor(self, soc: float) -> float:
        """Get SOC-based charging derating factor."""
        if soc > 0.9:
            return 0.3  # Trickle charge near full
        elif soc > 0.8:
            return 0.6  # Reduced power in upper range
        else:
            return 1.0  # Full power in main range
    
    def _get_discharge_soc_factor(self, soc: float) -> float:
        """Get SOC-based discharging derating factor."""
        min_soc = 1.0 - self.dod_max
        if soc < min_soc + 0.1:
            return 0.5  # Reduced power near minimum
        elif soc < min_soc + 0.2:
            return 0.8  # Slightly reduced power
        else:
            return 1.0  # Full power in main range
    
    def _get_max_charge_c_rate(self) -> float:
        """Get maximum charging C-rate based on technology."""
        c_rates = {
            'lifepo4': 1.0,      # 1C charging
            'li_ion_nmc': 0.8,   # 0.8C charging
            'lead_acid': 0.2     # 0.2C charging (slow)
        }
        return c_rates.get(self.technology, 0.5)
    
    def _get_max_discharge_c_rate(self) -> float:
        """Get maximum discharging C-rate based on technology."""
        c_rates = {
            'lifepo4': 2.0,      # 2C discharging
            'li_ion_nmc': 1.5,   # 1.5C discharging
            'lead_acid': 0.5     # 0.5C discharging
        }
        return c_rates.get(self.technology, 1.0)
    
    def update_soc(self, new_soc: float) -> None:
        """
        Update state of charge.
        
        Args:
            new_soc: New state of charge (0-1)
        """
        min_soc = 1.0 - self.dod_max
        self.current_soc = max(min_soc, min(new_soc, self.dod_max))
    
    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get summary of battery performance metrics.
        
        Returns:
            Dictionary with performance summary
        """
        return {
            'nominal_capacity_kwh': self.nominal_capacity_kwh,
            'current_capacity_kwh': self.current_capacity_kwh,
            'usable_capacity_kwh': self.get_usable_capacity(),
            'current_soc': self.current_soc,
            'energy_stored_kwh': self.get_energy_stored(),
            'capacity_fade_factor': self.capacity_fade_factor,
            'equivalent_full_cycles': self.equivalent_full_cycles,
            'total_energy_charged_kwh': self.total_energy_charged,
            'total_energy_discharged_kwh': self.total_energy_discharged,
            'round_trip_efficiency': self.round_trip_efficiency,
            'technology': self.technology,
            'temperature_c': self.temperature
        }
    
    def set_installation_date(self, date: str) -> None:
        """
        Set battery installation date for aging calculations.
        
        Args:
            date: Installation date string (e.g., '2024-01-01')
        """
        self.installation_date = pd.to_datetime(date)
    
    def __str__(self) -> str:
        """String representation of the battery system."""
        return (
            f"BatterySystem({self.current_capacity_kwh:.1f}kWh {self.technology}, "
            f"SOC={self.current_soc*100:.1f}%, "
            f"efficiency={self.round_trip_efficiency*100:.1f}%)"
        ) 