"""
Electrical loads component for SolarSim.

Contains the LoadManager class for modeling residential electrical loads
with realistic consumption patterns and load prioritization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import math
from datetime import datetime

from ..utils.constants import HOUSEHOLD_LOAD_PROFILES, MINUTES_PER_DAY
from ..utils.helpers import create_time_index


class LoadManager:
    """
    Electrical load manager with realistic consumption patterns.
    
    Models residential loads with appliance-level detail, seasonal variations,
    and smart load management capabilities.
    """
    
    def __init__(self, config):
        """
        Initialize load manager.
        
        Args:
            config: LoadConfig object with load parameters
        """
        self.config = config
        self.household_type = config.household_type
        self.daily_energy_kwh = config.daily_energy_kwh
        self.peak_power_kw = config.peak_power_kw
        self.base_load_kw = config.base_load_kw
        self.seasonal_variation = config.seasonal_variation
        self.weekend_factor = config.weekend_factor
        self.growth_rate = config.growth_rate
        self.critical_load_fraction = config.critical_load_fraction
        
        # Load statistics
        self.total_energy_consumed = 0.0  # kWh
        self.peak_demand_recorded = 0.0   # kW
        self.load_factor = 0.0           # Average/Peak
        
        # Load categories
        self.appliances = self._initialize_appliances()
        self.current_load_kw = 0.0
        self.curtailed_load_kw = 0.0
        
    def _initialize_appliances(self) -> Dict[str, Dict]:
        """Initialize appliance models based on household type."""
        
        # Base appliance mix for different household types
        appliance_configs = {
            'small': {
                'lighting': {'power_kw': 0.15, 'duty_cycle': 0.3, 'critical': True},
                'refrigerator': {'power_kw': 0.2, 'duty_cycle': 0.4, 'critical': True},
                'electronics': {'power_kw': 0.1, 'duty_cycle': 0.6, 'critical': False},
                'water_heater': {'power_kw': 0.5, 'duty_cycle': 0.15, 'critical': False},
                'hvac': {'power_kw': 0.8, 'duty_cycle': 0.2, 'critical': False},
                'other': {'power_kw': 0.2, 'duty_cycle': 0.1, 'critical': False}
            },
            'medium': {
                'lighting': {'power_kw': 0.4, 'duty_cycle': 0.35, 'critical': True},
                'refrigerator': {'power_kw': 0.25, 'duty_cycle': 0.4, 'critical': True},
                'electronics': {'power_kw': 0.3, 'duty_cycle': 0.7, 'critical': False},
                'water_heater': {'power_kw': 1.2, 'duty_cycle': 0.2, 'critical': False},
                'hvac': {'power_kw': 2.5, 'duty_cycle': 0.3, 'critical': False},
                'washer_dryer': {'power_kw': 1.0, 'duty_cycle': 0.1, 'critical': False},
                'cooking': {'power_kw': 1.5, 'duty_cycle': 0.08, 'critical': False},
                'other': {'power_kw': 0.5, 'duty_cycle': 0.15, 'critical': False}
            },
            'large': {
                'lighting': {'power_kw': 0.8, 'duty_cycle': 0.4, 'critical': True},
                'refrigerator': {'power_kw': 0.3, 'duty_cycle': 0.4, 'critical': True},
                'electronics': {'power_kw': 0.6, 'duty_cycle': 0.8, 'critical': False},
                'water_heater': {'power_kw': 2.0, 'duty_cycle': 0.25, 'critical': False},
                'hvac': {'power_kw': 5.0, 'duty_cycle': 0.4, 'critical': False},
                'washer_dryer': {'power_kw': 1.8, 'duty_cycle': 0.15, 'critical': False},
                'cooking': {'power_kw': 2.5, 'duty_cycle': 0.1, 'critical': False},
                'ev_charger': {'power_kw': 7.2, 'duty_cycle': 0.12, 'critical': False},
                'pool_pump': {'power_kw': 1.2, 'duty_cycle': 0.2, 'critical': False},
                'other': {'power_kw': 1.0, 'duty_cycle': 0.2, 'critical': False}
            }
        }
        
        return appliance_configs.get(self.household_type, appliance_configs['medium'])
    
    def calculate_instantaneous_load(self,
                                   timestamp: pd.Timestamp,
                                   ambient_temp: float = 25.0,
                                   available_power_kw: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate instantaneous electrical load.
        
        Args:
            timestamp: Current timestamp
            ambient_temp: Ambient temperature affecting HVAC loads
            available_power_kw: Available power for load management
            
        Returns:
            Dictionary with load breakdown
        """
        # Time-based factors
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
        day_of_year = timestamp.dayofyear
        
        # Seasonal factor
        seasonal_factor = 1 + self.seasonal_variation * math.sin(
            2 * math.pi * (day_of_year - 80) / 365  # Peak in summer
        )
        
        # Weekend factor
        weekend_factor = self.weekend_factor if day_of_week >= 5 else 1.0
        
        # Daily load profile
        daily_profile = self._get_daily_profile(hour_of_day)
        
        # Calculate appliance loads
        appliance_loads = {}
        total_load = 0.0
        critical_load = 0.0
        
        for appliance, config in self.appliances.items():
            # Base load from appliance duty cycle and daily profile
            base_power = config['power_kw'] * config['duty_cycle'] * daily_profile
            
            # Apply factors
            appliance_power = base_power * seasonal_factor * weekend_factor
            
            # Temperature-dependent loads (HVAC)
            if appliance == 'hvac':
                temp_factor = self._get_hvac_temperature_factor(ambient_temp)
                appliance_power *= temp_factor
            
            # Water heater temperature dependency
            elif appliance == 'water_heater':
                temp_factor = self._get_water_heater_factor(ambient_temp, hour_of_day)
                appliance_power *= temp_factor
            
            # Add some randomness (±10%)
            noise_factor = 1 + np.random.normal(0, 0.1)
            noise_factor = max(0.5, min(noise_factor, 1.5))  # Bound the noise
            appliance_power *= noise_factor
            
            appliance_loads[appliance] = max(0, appliance_power)
            total_load += appliance_loads[appliance]
            
            if config['critical']:
                critical_load += appliance_loads[appliance]
        
        # Add base load (always-on devices)
        base_load_power = self.base_load_kw * seasonal_factor
        appliance_loads['base_load'] = base_load_power
        total_load += base_load_power
        critical_load += base_load_power  # Base load is critical
        
        # Apply load management if power is limited
        if available_power_kw is not None and total_load > available_power_kw:
            managed_loads = self._apply_load_management(
                appliance_loads, available_power_kw, critical_load
            )
            total_load = sum(managed_loads.values())
            self.curtailed_load_kw = sum(appliance_loads.values()) - total_load
            appliance_loads = managed_loads
        else:
            self.curtailed_load_kw = 0.0
        
        self.current_load_kw = total_load
        
        return {
            'total_load_kw': total_load,
            'critical_load_kw': critical_load,
            'curtailed_load_kw': self.curtailed_load_kw,
            'appliance_loads': appliance_loads,
            'seasonal_factor': seasonal_factor,
            'daily_profile_factor': daily_profile
        }
    
    def _get_daily_profile(self, hour: int) -> float:
        """
        Get daily load profile factor based on hour of day.
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            Profile factor relative to average
        """
        # Typical residential load profile
        # Lower at night, peaks in morning and evening
        profile = np.array([
            0.6, 0.5, 0.5, 0.5, 0.6, 0.8,  # 0-5: Night/Early morning
            1.2, 1.5, 1.3, 1.0, 0.9, 0.9,  # 6-11: Morning peak
            1.0, 1.0, 1.0, 1.1, 1.3, 1.6,  # 12-17: Afternoon
            1.8, 1.9, 1.7, 1.4, 1.0, 0.8   # 18-23: Evening peak
        ])
        
        return profile[hour]
    
    def _get_hvac_temperature_factor(self, ambient_temp: float) -> float:
        """
        Calculate HVAC load factor based on temperature.
        
        Args:
            ambient_temp: Ambient temperature in °C
            
        Returns:
            Load factor for HVAC
        """
        comfort_temp = 22.0  # °C
        temp_diff = abs(ambient_temp - comfort_temp)
        
        if temp_diff < 3:
            return 0.3  # Minimal HVAC usage
        elif temp_diff < 8:
            return 0.5 + (temp_diff - 3) * 0.3  # Linear increase
        else:
            return 2.0  # Maximum HVAC usage
    
    def _get_water_heater_factor(self, ambient_temp: float, hour: int) -> float:
        """
        Calculate water heater load factor.
        
        Args:
            ambient_temp: Ambient temperature in °C
            hour: Hour of day
            
        Returns:
            Load factor for water heater
        """
        # Higher usage in morning and evening
        time_factor = 1.5 if hour in [6, 7, 8, 18, 19, 20] else 0.8
        
        # More heating needed in cold weather
        temp_factor = 1.5 - (ambient_temp - 10) * 0.02
        temp_factor = max(0.5, min(temp_factor, 2.0))
        
        return time_factor * temp_factor
    
    def _apply_load_management(self,
                             appliance_loads: Dict[str, float],
                             available_power: float,
                             critical_load: float) -> Dict[str, float]:
        """
        Apply load management to prioritize critical loads.
        
        Args:
            appliance_loads: Dictionary of appliance loads
            available_power: Available power limit
            critical_load: Total critical load
            
        Returns:
            Managed appliance loads
        """
        managed_loads = appliance_loads.copy()
        
        # If critical load exceeds available power, proportionally reduce all loads
        if critical_load > available_power:
            reduction_factor = available_power / critical_load
            for appliance in managed_loads:
                managed_loads[appliance] *= reduction_factor
            return managed_loads
        
        # Otherwise, maintain critical loads and manage non-critical loads
        remaining_power = available_power - critical_load
        non_critical_loads = {}
        
        for appliance, config in self.appliances.items():
            if not config['critical'] and appliance in managed_loads:
                non_critical_loads[appliance] = managed_loads[appliance]
        
        # Add non-critical base loads if any
        total_non_critical = sum(non_critical_loads.values())
        
        if total_non_critical > remaining_power:
            # Proportionally reduce non-critical loads
            if total_non_critical > 0:
                reduction_factor = remaining_power / total_non_critical
                for appliance in non_critical_loads:
                    managed_loads[appliance] *= reduction_factor
        
        return managed_loads
    
    def generate_load_time_series(self,
                                weather_data: pd.DataFrame,
                                start_date: str = '2024-01-01',
                                time_step_minutes: int = 15) -> pd.DataFrame:
        """
        Generate time series of electrical loads.
        
        Args:
            weather_data: DataFrame with temperature data
            start_date: Start date for load profile
            time_step_minutes: Time step in minutes
            
        Returns:
            DataFrame with load time series
        """
        # Create time index matching weather data
        time_index = weather_data.index
        
        results = []
        
        for timestamp in time_index:
            # Get temperature if available
            temp = weather_data.loc[timestamp, 'temp_air'] if 'temp_air' in weather_data.columns else 25.0
            
            # Calculate load
            load_result = self.calculate_instantaneous_load(timestamp, temp)
            
            # Update statistics
            time_step_hours = time_step_minutes / 60.0
            self.total_energy_consumed += load_result['total_load_kw'] * time_step_hours
            self.peak_demand_recorded = max(self.peak_demand_recorded, load_result['total_load_kw'])
            
            # Store results
            result_dict = {
                'timestamp': timestamp,
                'total_load_kw': load_result['total_load_kw'],
                'critical_load_kw': load_result['critical_load_kw'],
                'curtailed_load_kw': load_result['curtailed_load_kw'],
                'seasonal_factor': load_result['seasonal_factor'],
                'daily_profile_factor': load_result['daily_profile_factor']
            }
            
            # Add individual appliance loads
            for appliance, power in load_result['appliance_loads'].items():
                result_dict[f'{appliance}_kw'] = power
            
            results.append(result_dict)
        
        df = pd.DataFrame(results).set_index('timestamp')
        
        # Calculate load factor
        if self.peak_demand_recorded > 0:
            avg_load = self.total_energy_consumed / len(df) * (60 / time_step_minutes)
            self.load_factor = avg_load / self.peak_demand_recorded
        
        return df
    
    def apply_demand_response(self,
                            load_kw: float,
                            price_signal: float,
                            time_of_use_period: str = 'off_peak') -> Dict[str, float]:
        """
        Apply demand response based on price signals.
        
        Args:
            load_kw: Current load demand
            price_signal: Electricity price signal
            time_of_use_period: 'peak', 'off_peak', or 'shoulder'
            
        Returns:
            Dictionary with demand response results
        """
        # Base elasticity by appliance type
        elasticity = {
            'hvac': 0.3,         # Highly responsive
            'water_heater': 0.2,  # Moderately responsive
            'ev_charger': 0.4,    # Very responsive
            'pool_pump': 0.3,     # Moderately responsive
            'washer_dryer': 0.15, # Slightly responsive
            'other': 0.1          # Low responsiveness
        }
        
        # Price response factor
        high_price_threshold = 0.15  # $/kWh
        if price_signal > high_price_threshold:
            price_factor = 1 - (price_signal - high_price_threshold) * 2  # Reduce load
        else:
            price_factor = 1.0
        
        # Time-of-use response
        tou_factors = {
            'peak': 0.8,      # Reduce load during peak
            'shoulder': 0.95,  # Slight reduction
            'off_peak': 1.1    # Increase load during off-peak
        }
        tou_factor = tou_factors.get(time_of_use_period, 1.0)
        
        # Apply demand response
        total_factor = price_factor * tou_factor
        responsive_load = load_kw * total_factor
        load_reduction = load_kw - responsive_load
        
        return {
            'original_load_kw': load_kw,
            'responsive_load_kw': max(0, responsive_load),
            'load_reduction_kw': max(0, load_reduction),
            'price_factor': price_factor,
            'tou_factor': tou_factor
        }
    
    def get_load_statistics(self) -> Dict[str, float]:
        """
        Get summary statistics for load consumption.
        
        Returns:
            Dictionary with load statistics
        """
        return {
            'household_type': self.household_type,
            'daily_energy_target_kwh': self.daily_energy_kwh,
            'total_energy_consumed_kwh': self.total_energy_consumed,
            'peak_demand_kw': self.peak_demand_recorded,
            'load_factor': self.load_factor,
            'critical_load_fraction': self.critical_load_fraction,
            'current_load_kw': self.current_load_kw,
            'curtailed_load_kw': self.curtailed_load_kw
        }
    
    def simulate_load_growth(self, years: int) -> None:
        """
        Simulate load growth over time.
        
        Args:
            years: Number of years for growth simulation
        """
        growth_factor = (1 + self.growth_rate) ** years
        self.daily_energy_kwh *= growth_factor
        self.peak_power_kw *= growth_factor
        
        # Update appliance power ratings
        for appliance in self.appliances:
            self.appliances[appliance]['power_kw'] *= growth_factor
    
    def __str__(self) -> str:
        """String representation of the load manager."""
        return (
            f"LoadManager({self.household_type}, "
            f"daily_energy={self.daily_energy_kwh:.1f}kWh, "
            f"peak_power={self.peak_power_kw:.1f}kW)"
        ) 