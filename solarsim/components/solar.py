"""
Solar photovoltaic array component for SolarSim.

Contains the SolarPVArray class for modeling solar panel performance,
including temperature effects, degradation, and various PV technologies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import math

from ..utils.constants import (
    STANDARD_TEST_CONDITIONS_IRRADIANCE,
    STANDARD_TEST_CONDITIONS_TEMPERATURE,
    PV_TECHNOLOGIES
)
from ..utils.helpers import pv_cell_temperature, interpolate_efficiency


class SolarPVArray:
    """
    Solar photovoltaic array model with realistic performance characteristics.
    
    Supports multiple PV technologies with temperature effects, degradation,
    and various mounting configurations.
    """
    
    def __init__(self, config):
        """
        Initialize solar PV array.
        
        Args:
            config: PVConfig object with array parameters
        """
        self.config = config
        self.technology = config.technology
        self.capacity_kw = config.capacity_kw
        self.efficiency = config.efficiency
        self.temperature_coefficient = config.temperature_coefficient
        self.degradation_rate = config.degradation_rate
        self.tilt_angle = config.tilt_angle
        self.azimuth_angle = config.azimuth_angle
        self.mounting_type = config.mounting_type
        self.tracking = config.tracking
        self.dc_ac_ratio = config.dc_ac_ratio
        
        # Performance tracking
        self.total_energy_generated = 0.0  # kWh
        self.operating_hours = 0.0
        self.degraded_efficiency = self.efficiency
        self.installation_date = None
        
    def calculate_power_output(self, 
                             irradiance: float,
                             ambient_temp: float,
                             wind_speed: float = 2.0,
                             timestamp: Optional[pd.Timestamp] = None) -> Dict[str, float]:
        """
        Calculate instantaneous power output of the PV array.
        
        Args:
            irradiance: Solar irradiance in W/m²
            ambient_temp: Ambient temperature in °C
            wind_speed: Wind speed in m/s
            timestamp: Optional timestamp for degradation calculation
            
        Returns:
            Dictionary with power output and performance metrics
        """
        # Calculate cell temperature
        cell_temp = pv_cell_temperature(
            ambient_temp, irradiance, wind_speed, self.mounting_type
        )
        
        # Temperature effect on efficiency
        temp_effect = 1 + self.temperature_coefficient * (
            cell_temp - STANDARD_TEST_CONDITIONS_TEMPERATURE
        )
        
        # Current efficiency with temperature effect
        current_efficiency = self.degraded_efficiency * temp_effect
        
        # Irradiance effect (linear relationship)
        irradiance_factor = irradiance / STANDARD_TEST_CONDITIONS_IRRADIANCE
        
        # DC power output
        dc_power_kw = (
            self.capacity_kw * 
            current_efficiency / self.efficiency *  # Relative to nameplate
            irradiance_factor
        )
        
        # Apply realistic power curve (non-linear at low irradiance)
        if irradiance < 200:  # W/m²
            # Reduced performance at very low irradiance
            low_irradiance_factor = (irradiance / 200) ** 1.2
            dc_power_kw *= low_irradiance_factor
        
        # Ensure power doesn't exceed nameplate capacity
        dc_power_kw = min(dc_power_kw, self.capacity_kw)
        
        # Apply degradation if timestamp provided
        if timestamp and self.installation_date:
            years_operating = (timestamp - self.installation_date).days / 365.25
            degradation_factor = (1 - self.degradation_rate) ** years_operating
            dc_power_kw *= degradation_factor
        
        # Calculate performance ratio
        performance_ratio = 0.0
        if irradiance > 0:
            theoretical_power = self.capacity_kw * irradiance_factor
            performance_ratio = dc_power_kw / theoretical_power if theoretical_power > 0 else 0
        
        return {
            'dc_power_kw': max(0, dc_power_kw),
            'cell_temperature_c': cell_temp,
            'current_efficiency': current_efficiency,
            'performance_ratio': performance_ratio,
            'irradiance_factor': irradiance_factor,
            'temperature_factor': temp_effect
        }
    
    def calculate_angle_of_incidence(self, 
                                   solar_elevation: float,
                                   solar_azimuth: float) -> float:
        """
        Calculate angle of incidence of solar radiation on tilted surface.
        
        Args:
            solar_elevation: Solar elevation angle in degrees
            solar_azimuth: Solar azimuth angle in degrees
            
        Returns:
            Angle of incidence in degrees
        """
        # Convert to radians
        elevation_rad = math.radians(solar_elevation)
        azimuth_rad = math.radians(solar_azimuth)
        tilt_rad = math.radians(self.tilt_angle)
        panel_azimuth_rad = math.radians(self.azimuth_angle)
        
        # Calculate angle of incidence using spherical trigonometry
        cos_incidence = (
            math.sin(elevation_rad) * math.cos(tilt_rad) +
            math.cos(elevation_rad) * math.sin(tilt_rad) *
            math.cos(azimuth_rad - panel_azimuth_rad)
        )
        
        # Ensure cos_incidence is within valid range
        cos_incidence = max(-1, min(1, cos_incidence))
        
        incidence_angle = math.degrees(math.acos(cos_incidence))
        
        return incidence_angle
    
    def calculate_irradiance_on_tilted_surface(self,
                                             ghi: float,
                                             dni: float, 
                                             dhi: float,
                                             solar_elevation: float,
                                             solar_azimuth: float) -> Dict[str, float]:
        """
        Calculate irradiance components on tilted PV surface.
        
        Args:
            ghi: Global horizontal irradiance in W/m²
            dni: Direct normal irradiance in W/m²
            dhi: Diffuse horizontal irradiance in W/m²
            solar_elevation: Solar elevation angle in degrees
            solar_azimuth: Solar azimuth angle in degrees
            
        Returns:
            Dictionary with irradiance components on tilted surface
        """
        if solar_elevation <= 0:
            return {'direct': 0, 'diffuse': 0, 'reflected': 0, 'total': 0}
        
        # Angle of incidence
        incidence_angle = self.calculate_angle_of_incidence(
            solar_elevation, solar_azimuth
        )
        
        # Direct irradiance on tilted surface
        if incidence_angle <= 90:
            direct_tilted = dni * math.cos(math.radians(incidence_angle))
        else:
            direct_tilted = 0  # Sun behind the panel
        
        # Diffuse irradiance (isotropic sky model)
        tilt_rad = math.radians(self.tilt_angle)
        diffuse_tilted = dhi * (1 + math.cos(tilt_rad)) / 2
        
        # Ground-reflected irradiance
        ground_reflectance = 0.2  # Typical value
        reflected_tilted = ghi * ground_reflectance * (1 - math.cos(tilt_rad)) / 2
        
        # Total irradiance on tilted surface
        total_tilted = direct_tilted + diffuse_tilted + reflected_tilted
        
        return {
            'direct': max(0, direct_tilted),
            'diffuse': max(0, diffuse_tilted),
            'reflected': max(0, reflected_tilted),
            'total': max(0, total_tilted),
            'incidence_angle': incidence_angle
        }
    
    def update_degradation(self, years_operating: float) -> None:
        """
        Update panel efficiency due to degradation.
        
        Args:
            years_operating: Years since installation
        """
        degradation_factor = (1 - self.degradation_rate) ** years_operating
        self.degraded_efficiency = self.efficiency * degradation_factor
    
    def calculate_soiling_loss(self, 
                             days_since_cleaning: int,
                             climate_zone: str = 'temperate') -> float:
        """
        Calculate power loss due to soiling.
        
        Args:
            days_since_cleaning: Days since last panel cleaning
            climate_zone: Climate zone affecting soiling rate
            
        Returns:
            Soiling loss factor (0-1, where 1 is no loss)
        """
        # Soiling rates by climate zone (%/month)
        soiling_rates = {
            'desert': 2.0,      # High dust
            'tropical': 1.5,    # Rain helps cleaning
            'temperate': 1.0,   # Moderate soiling
            'cold': 0.5        # Low soiling in winter
        }
        
        monthly_rate = soiling_rates.get(climate_zone, 1.0) / 100  # Convert to fraction
        daily_rate = monthly_rate / 30  # Daily soiling rate
        
        # Maximum soiling loss (typically 5-10%)
        max_soiling_loss = 0.08
        
        # Calculate current soiling loss
        soiling_loss = min(
            daily_rate * days_since_cleaning,
            max_soiling_loss
        )
        
        return 1 - soiling_loss
    
    def generate_power_time_series(self,
                                 weather_data: pd.DataFrame,
                                 start_date: str = '2024-01-01') -> pd.DataFrame:
        """
        Generate time series of power output from weather data.
        
        Args:
            weather_data: DataFrame with columns ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed']
            start_date: Installation date for degradation calculation
            
        Returns:
            DataFrame with power output time series
        """
        self.installation_date = pd.to_datetime(start_date)
        
        results = []
        
        for idx, row in weather_data.iterrows():
            # Skip if required data is missing
            if pd.isna(row[['ghi', 'temp_air']]).any():
                results.append({
                    'timestamp': idx,
                    'dc_power_kw': 0,
                    'irradiance_poa': 0,
                    'cell_temperature': row.get('temp_air', 25),
                    'performance_ratio': 0
                })
                continue
            
            # Use plane-of-array irradiance if available, otherwise use GHI
            if 'irradiance_poa' in row and not pd.isna(row['irradiance_poa']):
                poa_irradiance = row['irradiance_poa']
            else:
                poa_irradiance = row['ghi']  # Simplified - assumes optimal tilt
            
            # Calculate power output
            power_result = self.calculate_power_output(
                irradiance=poa_irradiance,
                ambient_temp=row['temp_air'],
                wind_speed=row.get('wind_speed', 2.0),
                timestamp=idx
            )
            
            results.append({
                'timestamp': idx,
                'dc_power_kw': power_result['dc_power_kw'],
                'irradiance_poa': poa_irradiance,
                'cell_temperature': power_result['cell_temperature_c'],
                'performance_ratio': power_result['performance_ratio'],
                'current_efficiency': power_result['current_efficiency']
            })
        
        return pd.DataFrame(results).set_index('timestamp')
    
    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get summary of array performance metrics.
        
        Returns:
            Dictionary with performance summary
        """
        return {
            'nameplate_capacity_kw': self.capacity_kw,
            'current_efficiency': self.degraded_efficiency,
            'degradation_factor': self.degraded_efficiency / self.efficiency,
            'total_energy_generated_kwh': self.total_energy_generated,
            'operating_hours': self.operating_hours,
            'capacity_factor': (
                self.total_energy_generated / (self.capacity_kw * self.operating_hours)
                if self.operating_hours > 0 else 0
            ),
            'technology': self.technology,
            'mounting_type': self.mounting_type
        }
    
    def __str__(self) -> str:
        """String representation of the solar array."""
        return (
            f"SolarPVArray({self.capacity_kw:.1f}kW {self.technology}, "
            f"efficiency={self.efficiency*100:.1f}%, "
            f"tilt={self.tilt_angle}°)"
        ) 