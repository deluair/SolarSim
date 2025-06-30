"""
Weather data generation module for SolarSim.

Contains functions and classes for generating synthetic meteorological data
including solar irradiance, temperature, wind speed, and humidity.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import math
from datetime import datetime, timedelta

from ..utils.constants import CLIMATE_ZONES
from ..utils.helpers import (
    solar_position, air_mass, clear_sky_irradiance, 
    create_time_index, validate_coordinates
)


class WeatherGenerator:
    """
    Synthetic weather data generator for solar simulation.
    
    Generates realistic weather patterns including:
    - Solar irradiance (GHI, DNI, DHI)
    - Air temperature
    - Wind speed and direction
    - Humidity
    - Cloud cover variations
    """
    
    def __init__(self, location: Dict[str, float], climate_zone: str = 'temperate'):
        """
        Initialize weather generator.
        
        Args:
            location: Dictionary with 'latitude', 'longitude', 'altitude'
            climate_zone: Climate zone type
        """
        self.latitude = location['latitude']
        self.longitude = location['longitude']
        self.altitude = location.get('altitude', 0)
        self.climate_zone = climate_zone
        
        # Validate inputs
        if not validate_coordinates(self.latitude, self.longitude):
            raise ValueError(f"Invalid coordinates: {self.latitude}, {self.longitude}")
        
        if climate_zone not in CLIMATE_ZONES:
            raise ValueError(f"Unknown climate zone: {climate_zone}")
        
        # Get climate parameters
        self.climate_params = CLIMATE_ZONES[climate_zone]
        
        # Initialize random seed for reproducibility
        self.random_seed = None
        
    def generate_annual_weather(self,
                              year: int = 2024,
                              time_step_hours: float = 1.0,
                              weather_uncertainty: bool = True) -> pd.DataFrame:
        """
        Generate a full year of synthetic weather data.
        
        Args:
            year: Year for the data
            time_step_hours: Time step in hours
            weather_uncertainty: Include weather variability
            
        Returns:
            DataFrame with weather time series
        """
        # Create time index
        start_date = f"{year}-01-01"
        periods = int(365.25 * 24 / time_step_hours)
        time_index = create_time_index(start_date, periods, f"{int(time_step_hours*60)}T")
        
        # Initialize random generator
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        weather_data = []
        
        for timestamp in time_index:
            # Calculate solar position
            solar_elev, solar_azim = solar_position(
                self.latitude, self.longitude, timestamp
            )
            
            # Calculate clear sky irradiance
            am = air_mass(solar_elev)
            clear_sky = clear_sky_irradiance(solar_elev, am, self.altitude)
            
            # Generate cloud cover and atmospheric conditions
            cloud_cover = self._generate_cloud_cover(timestamp, weather_uncertainty)
            atmospheric_transmission = self._calculate_atmospheric_transmission(
                cloud_cover, timestamp
            )
            
            # Apply cloud effects to irradiance
            ghi = clear_sky['ghi'] * atmospheric_transmission['ghi_factor']
            dni = clear_sky['dni'] * atmospheric_transmission['dni_factor'] 
            dhi = clear_sky['dhi'] * atmospheric_transmission['dhi_factor']
            
            # Generate temperature
            temp_air = self._generate_temperature(timestamp, weather_uncertainty)
            
            # Generate wind
            wind_speed, wind_direction = self._generate_wind(timestamp, weather_uncertainty)
            
            # Generate humidity
            humidity = self._generate_humidity(temp_air, timestamp, weather_uncertainty)
            
            # Store data
            weather_data.append({
                'timestamp': timestamp,
                'ghi': max(0, ghi),
                'dni': max(0, dni),
                'dhi': max(0, dhi),
                'temp_air': temp_air,
                'wind_speed': wind_speed,
                'wind_direction': wind_direction,
                'humidity': humidity,
                'cloud_cover': cloud_cover,
                'solar_elevation': solar_elev,
                'solar_azimuth': solar_azim,
                'air_mass': am
            })
        
        return pd.DataFrame(weather_data).set_index('timestamp')
    
    def _generate_cloud_cover(self, timestamp: pd.Timestamp, uncertainty: bool) -> float:
        """Generate cloud cover fraction (0-1)."""
        day_of_year = timestamp.dayofyear
        hour_of_day = timestamp.hour
        
        # Seasonal variation
        seasonal_base = 0.4 + 0.2 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
        
        # Daily variation (more clouds in afternoon)
        daily_factor = 1 + 0.3 * math.sin(math.pi * (hour_of_day - 6) / 12)
        
        # Climate zone adjustment
        climate_factors = {
            'desert': 0.3,     # Less clouds
            'tropical': 1.5,   # More clouds
            'temperate': 1.0,  # Average
            'cold': 1.2       # Slightly more clouds
        }
        climate_factor = climate_factors.get(self.climate_zone, 1.0)
        
        cloud_cover = seasonal_base * daily_factor * climate_factor
        
        # Add uncertainty/noise
        if uncertainty:
            noise = np.random.normal(0, 0.2)
            cloud_cover += noise
        
        return max(0, min(cloud_cover, 1.0))
    
    def _calculate_atmospheric_transmission(self, cloud_cover: float, 
                                          timestamp: pd.Timestamp) -> Dict[str, float]:
        """Calculate atmospheric transmission factors."""
        # Base transmission (clear sky)
        base_transmission = 0.8
        
        # Cloud effects
        cloud_factor = 1 - 0.75 * cloud_cover  # Clouds reduce transmission
        
        # Seasonal atmospheric effects (pollution, humidity)
        day_of_year = timestamp.dayofyear
        seasonal_factor = 0.95 + 0.05 * math.cos(2 * math.pi * day_of_year / 365)
        
        # Overall transmission
        transmission = base_transmission * cloud_factor * seasonal_factor
        
        # Different effects on irradiance components
        ghi_factor = transmission
        dni_factor = transmission ** 1.5  # Direct beam more affected by clouds
        dhi_factor = min(1.0, transmission + 0.3 * cloud_cover)  # Clouds increase diffuse
        
        return {
            'ghi_factor': ghi_factor,
            'dni_factor': dni_factor,
            'dhi_factor': dhi_factor
        }
    
    def _generate_temperature(self, timestamp: pd.Timestamp, uncertainty: bool) -> float:
        """Generate air temperature in °C."""
        day_of_year = timestamp.dayofyear
        hour_of_day = timestamp.hour
        
        # Climate zone temperature ranges
        temp_range = self.climate_params['temp_range']
        temp_min, temp_max = temp_range
        temp_avg = (temp_min + temp_max) / 2
        temp_amplitude = (temp_max - temp_min) / 2
        
        # Annual temperature cycle
        annual_temp = temp_avg + temp_amplitude * math.cos(
            2 * math.pi * (day_of_year - 200) / 365  # Coldest around day 200 (winter)
        )
        
        # Daily temperature cycle
        daily_amplitude = 8.0  # ±8°C daily variation
        daily_temp = annual_temp + daily_amplitude * math.cos(
            2 * math.pi * (hour_of_day - 14) / 24  # Peak at 2 PM
        )
        
        # Add uncertainty
        if uncertainty:
            noise = np.random.normal(0, 2.0)  # ±2°C random variation
            daily_temp += noise
        
        return daily_temp
    
    def _generate_wind(self, timestamp: pd.Timestamp, uncertainty: bool) -> Tuple[float, float]:
        """Generate wind speed (m/s) and direction (degrees)."""
        # Base wind speed from climate
        base_wind = self.climate_params['wind_speed_avg']
        
        # Seasonal variation (higher in winter)
        day_of_year = timestamp.dayofyear
        seasonal_factor = 1 + 0.3 * math.cos(2 * math.pi * (day_of_year - 30) / 365)
        
        # Daily variation (higher during day)
        hour_of_day = timestamp.hour
        daily_factor = 1 + 0.2 * math.sin(math.pi * (hour_of_day - 6) / 12)
        
        wind_speed = base_wind * seasonal_factor * daily_factor
        
        # Add uncertainty
        if uncertainty:
            wind_noise = np.random.exponential(0.5)  # Wind speed variability
            wind_speed += wind_noise
        
        # Wind direction (simplified - could be more complex)
        if uncertainty:
            wind_direction = np.random.uniform(0, 360)
        else:
            wind_direction = 225  # SW winds (typical)
        
        return max(0, wind_speed), wind_direction
    
    def _generate_humidity(self, temperature: float, timestamp: pd.Timestamp, 
                          uncertainty: bool) -> float:
        """Generate relative humidity (%)."""
        # Base humidity from climate
        humidity_range = self.climate_params['humidity_range']
        humidity_min, humidity_max = humidity_range
        base_humidity = (humidity_min + humidity_max) / 2
        
        # Temperature correlation (inverse relationship)
        temp_factor = max(0.5, 1 - (temperature - 20) * 0.02)
        
        # Time of day variation (higher at night)
        hour_of_day = timestamp.hour
        daily_factor = 1 + 0.2 * math.cos(2 * math.pi * (hour_of_day - 6) / 24)
        
        humidity = base_humidity * temp_factor * daily_factor
        
        # Add uncertainty
        if uncertainty:
            noise = np.random.normal(0, 5.0)  # ±5% variation
            humidity += noise
        
        return max(10, min(humidity, 95))  # Bound humidity
    
    def set_random_seed(self, seed: int) -> None:
        """Set random seed for reproducible weather generation."""
        self.random_seed = seed


def generate_weather_data(location: Dict[str, float], 
                         years: int = 1,
                         climate_zone: str = 'temperate',
                         time_step_hours: float = 1.0,
                         random_seed: Optional[int] = None) -> pd.DataFrame:
    """
    Generate synthetic weather data for solar simulation.
    
    Args:
        location: Dictionary with latitude, longitude, altitude
        years: Number of years to generate
        climate_zone: Climate zone type
        time_step_hours: Time step in hours
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with weather time series
    """
    generator = WeatherGenerator(location, climate_zone)
    
    if random_seed is not None:
        generator.set_random_seed(random_seed)
    
    # Generate weather for each year
    weather_frames = []
    
    for year in range(2024, 2024 + years):
        annual_weather = generator.generate_annual_weather(
            year=year,
            time_step_hours=time_step_hours,
            weather_uncertainty=True
        )
        weather_frames.append(annual_weather)
    
    # Combine all years
    if len(weather_frames) == 1:
        return weather_frames[0]
    else:
        return pd.concat(weather_frames, ignore_index=False)
