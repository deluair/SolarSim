"""
Load profile generation module for SolarSim.

Contains functions and classes for generating realistic residential
electrical load profiles with appliance-level detail.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import math

from ..utils.constants import HOUSEHOLD_LOAD_PROFILES
from ..utils.helpers import create_time_index


class LoadProfileGenerator:
    """
    Generator for realistic residential load profiles.
    
    Creates time series load data with:
    - Seasonal variations
    - Daily patterns
    - Weekend/weekday differences
    - Individual appliance profiles
    """
    
    def __init__(self, household_type: str = 'medium'):
        """
        Initialize load profile generator.
        
        Args:
            household_type: Type of household ('small', 'medium', 'large')
        """
        if household_type not in HOUSEHOLD_LOAD_PROFILES:
            raise ValueError(f"Unknown household type: {household_type}")
        
        self.household_type = household_type
        self.profile_params = HOUSEHOLD_LOAD_PROFILES[household_type]
        self.random_seed = None
        
    def generate_annual_load_profile(self,
                                   year: int = 2024,
                                   time_step_minutes: int = 15) -> pd.DataFrame:
        """
        Generate annual load profile with specified time resolution.
        
        Args:
            year: Year for the profile
            time_step_minutes: Time step in minutes
            
        Returns:
            DataFrame with load time series
        """
        # Create time index
        start_date = f"{year}-01-01"
        periods = int(365.25 * 24 * 60 / time_step_minutes)
        time_index = create_time_index(start_date, periods, f"{time_step_minutes}T")
        
        # Initialize random generator
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        load_data = []
        
        for timestamp in time_index:
            # Calculate load for this timestamp
            load_result = self._calculate_load_at_time(timestamp)
            
            load_data.append({
                'timestamp': timestamp,
                'total_load_kw': load_result['total_load'],
                'base_load_kw': load_result['base_load'],
                'lighting_kw': load_result['lighting'],
                'hvac_kw': load_result['hvac'],
                'water_heater_kw': load_result['water_heater'],
                'appliances_kw': load_result['appliances'],
                'seasonal_factor': load_result['seasonal_factor'],
                'daily_factor': load_result['daily_factor'],
                'weekend_factor': load_result['weekend_factor']
            })
        
        return pd.DataFrame(load_data).set_index('timestamp')
    
    def _calculate_load_at_time(self, timestamp: pd.Timestamp) -> Dict[str, float]:
        """Calculate load breakdown at specific timestamp."""
        
        # Time-based factors
        day_of_year = timestamp.dayofyear
        hour_of_day = timestamp.hour
        minute_of_hour = timestamp.minute
        day_of_week = timestamp.weekday()  # 0=Monday
        
        # Seasonal factor (higher in summer for cooling, winter for heating)
        seasonal_factor = 1.0 + 0.3 * (
            math.sin(2 * math.pi * (day_of_year - 80) / 365) +  # Summer cooling
            0.5 * math.cos(2 * math.pi * (day_of_year - 350) / 365)  # Winter heating
        )
        
        # Weekend factor
        weekend_factor = 1.15 if day_of_week >= 5 else 1.0
        
        # Daily load pattern
        daily_factor = self._get_daily_load_pattern(hour_of_day, minute_of_hour)
        
        # Base load (always-on devices)
        base_load = self.profile_params['base_load'] * seasonal_factor
        
        # Lighting load
        lighting_load = self._calculate_lighting_load(
            hour_of_day, day_of_year, seasonal_factor
        )
        
        # HVAC load (most variable)
        hvac_load = self._calculate_hvac_load(
            hour_of_day, day_of_year, seasonal_factor, daily_factor
        )
        
        # Water heater load
        water_heater_load = self._calculate_water_heater_load(
            hour_of_day, weekend_factor
        )
        
        # Other appliances
        appliances_load = self._calculate_appliances_load(
            hour_of_day, weekend_factor, daily_factor
        )
        
        # Total load
        total_load = (
            base_load + lighting_load + hvac_load + 
            water_heater_load + appliances_load
        )
        
        # Apply randomness (±15%)
        noise_factor = 1 + np.random.normal(0, 0.15)
        noise_factor = max(0.5, min(noise_factor, 1.5))  # Bound the noise
        total_load *= noise_factor
        
        return {
            'total_load': max(0, total_load),
            'base_load': base_load,
            'lighting': lighting_load,
            'hvac': hvac_load,
            'water_heater': water_heater_load,
            'appliances': appliances_load,
            'seasonal_factor': seasonal_factor,
            'daily_factor': daily_factor,
            'weekend_factor': weekend_factor
        }
    
    def _get_daily_load_pattern(self, hour: int, minute: int) -> float:
        """Get normalized daily load pattern."""
        # Convert to decimal hour
        decimal_hour = hour + minute / 60.0
        
        # Base daily pattern (higher during morning and evening)
        if 0 <= decimal_hour < 6:
            # Night: Low load
            factor = 0.6 + 0.1 * math.sin(math.pi * decimal_hour / 6)
        elif 6 <= decimal_hour < 9:
            # Morning: Rising
            factor = 0.7 + 0.6 * (decimal_hour - 6) / 3
        elif 9 <= decimal_hour < 17:
            # Day: Moderate
            factor = 0.8 + 0.2 * math.sin(math.pi * (decimal_hour - 9) / 8)
        elif 17 <= decimal_hour < 22:
            # Evening: Peak
            factor = 1.3 + 0.4 * math.sin(math.pi * (decimal_hour - 17) / 5)
        else:
            # Late evening: Declining
            factor = 1.0 - 0.4 * (decimal_hour - 22) / 2
        
        return factor
    
    def _calculate_lighting_load(self, hour: int, day_of_year: int, 
                               seasonal_factor: float) -> float:
        """Calculate lighting load based on time and season."""
        # Base lighting power
        base_power = 0.3 if self.household_type == 'small' else (
            0.5 if self.household_type == 'medium' else 0.8
        )
        
        # Daylight hours vary by season
        daylight_factor = 0.5 + 0.3 * math.cos(2 * math.pi * (day_of_year - 172) / 365)
        
        # Time of day factor
        if 6 <= hour <= 8:
            time_factor = 0.7  # Morning
        elif 9 <= hour <= 16:
            time_factor = 0.2 * (1 - daylight_factor)  # Day (less in summer)
        elif 17 <= hour <= 22:
            time_factor = 1.0  # Evening peak
        elif 23 <= hour or hour <= 5:
            time_factor = 0.3  # Night
        else:
            time_factor = 0.4
        
        return base_power * time_factor * seasonal_factor
    
    def _calculate_hvac_load(self, hour: int, day_of_year: int, 
                           seasonal_factor: float, daily_factor: float) -> float:
        """Calculate HVAC load with seasonal and daily patterns."""
        # Base HVAC power by household size
        base_power = {
            'small': 1.5,
            'medium': 3.0,
            'large': 5.0
        }[self.household_type]
        
        # Seasonal HVAC usage (cooling in summer, heating in winter)
        summer_peak = day_of_year in range(150, 240)  # Jun-Aug
        winter_peak = day_of_year in range(350, 365) or day_of_year in range(1, 60)  # Dec-Feb
        
        if summer_peak:
            seasonal_hvac_factor = 1.5 + 0.5 * math.sin(2 * math.pi * (day_of_year - 150) / 90)
        elif winter_peak:
            day_adj = day_of_year if day_of_year > 300 else day_of_year + 365
            seasonal_hvac_factor = 1.3 + 0.4 * math.cos(2 * math.pi * (day_adj - 350) / 75)
        else:
            seasonal_hvac_factor = 0.3  # Spring/Fall minimal usage
        
        # Daily pattern (peak during hot afternoon or cold morning/evening)
        if summer_peak:
            # Cooling peak in afternoon
            if 12 <= hour <= 18:
                daily_hvac_factor = 1.5 + 0.5 * math.sin(math.pi * (hour - 12) / 6)
            else:
                daily_hvac_factor = 0.6
        elif winter_peak:
            # Heating peaks morning and evening
            if 6 <= hour <= 9 or 17 <= hour <= 22:
                daily_hvac_factor = 1.2
            else:
                daily_hvac_factor = 0.8
        else:
            daily_hvac_factor = 0.5
        
        return base_power * seasonal_hvac_factor * daily_hvac_factor
    
    def _calculate_water_heater_load(self, hour: int, weekend_factor: float) -> float:
        """Calculate water heater load with usage patterns."""
        # Base water heater power
        base_power = {
            'small': 1.0,
            'medium': 2.0,
            'large': 3.0
        }[self.household_type]
        
        # Usage pattern (peaks during morning and evening showers)
        if hour in [6, 7, 8]:  # Morning shower
            time_factor = 0.8
        elif hour in [18, 19, 20]:  # Evening shower/dishes
            time_factor = 1.0
        elif hour in [21, 22]:  # Evening dishes/cleanup
            time_factor = 0.6
        else:
            time_factor = 0.2  # Standby/maintenance
        
        return base_power * time_factor * weekend_factor
    
    def _calculate_appliances_load(self, hour: int, weekend_factor: float, 
                                 daily_factor: float) -> float:
        """Calculate miscellaneous appliances load."""
        # Base appliance power
        base_power = {
            'small': 0.5,
            'medium': 1.2,
            'large': 2.0
        }[self.household_type]
        
        # Usage patterns for different appliances
        # Cooking (morning and evening)
        cooking_factor = 0.5 if hour in [7, 8, 18, 19, 20] else 0.1
        
        # Laundry (typically afternoon/evening, more on weekends)
        laundry_factor = 0.3 if hour in [14, 15, 16, 17] else 0.05
        laundry_factor *= weekend_factor
        
        # Entertainment/electronics (evening peak)
        electronics_factor = 1.0 if 18 <= hour <= 23 else 0.6
        
        # Total appliances load
        total_factor = (cooking_factor + laundry_factor + electronics_factor) / 3
        
        return base_power * total_factor * daily_factor
    
    def set_random_seed(self, seed: int) -> None:
        """Set random seed for reproducible load generation."""
        self.random_seed = seed
    
    def get_load_statistics(self, load_profile: pd.DataFrame) -> Dict[str, float]:
        """Calculate statistics for generated load profile."""
        # Calculate time step in hours
        time_diff = load_profile.index[1] - load_profile.index[0]
        time_step_hours = time_diff.total_seconds() / 3600
        
        total_energy = load_profile['total_load_kw'].sum() * time_step_hours
        
        peak_load = load_profile['total_load_kw'].max()
        avg_load = load_profile['total_load_kw'].mean()
        load_factor = avg_load / peak_load if peak_load > 0 else 0
        
        # Daily energy
        daily_data = load_profile.resample('D')['total_load_kw'].sum() * time_step_hours
        avg_daily_energy = daily_data.mean()
        
        return {
            'total_energy_kwh': total_energy,
            'peak_load_kw': peak_load,
            'average_load_kw': avg_load,
            'load_factor': load_factor,
            'average_daily_energy_kwh': avg_daily_energy,
            'household_type': self.household_type
        }


def generate_load_profile(household_type: str = 'medium',
                         years: int = 1,
                         time_step_minutes: int = 15,
                         random_seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate residential load profile.
    
    Args:
        household_type: Type of household ('small', 'medium', 'large')
        years: Number of years to generate
        time_step_minutes: Time resolution in minutes
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with load time series
    """
    generator = LoadProfileGenerator(household_type)
    
    if random_seed is not None:
        generator.set_random_seed(random_seed)
    
    # Generate load profiles for each year
    load_frames = []
    
    for year in range(2024, 2024 + years):
        annual_load = generator.generate_annual_load_profile(
            year=year,
            time_step_minutes=time_step_minutes
        )
        load_frames.append(annual_load)
    
    # Combine all years
    if len(load_frames) == 1:
        return load_frames[0]
    else:
        return pd.concat(load_frames, ignore_index=False)
