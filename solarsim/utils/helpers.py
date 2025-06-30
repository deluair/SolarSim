"""
Helper functions for SolarSim calculations and utilities.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
from typing import Union, Tuple, Optional, Dict, Any

def solar_position(latitude: float, longitude: float, timestamp: pd.Timestamp) -> Tuple[float, float]:
    """
    Calculate solar position (elevation and azimuth) for given location and time.
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees  
        timestamp: Timestamp for calculation
        
    Returns:
        Tuple of (solar_elevation, solar_azimuth) in degrees
    """
    # Day of year
    day_of_year = timestamp.dayofyear
    
    # Solar declination angle
    declination = 23.45 * math.sin(math.radians(360 * (284 + day_of_year) / 365.25))
    
    # Hour angle
    solar_time = timestamp.hour + timestamp.minute / 60.0
    hour_angle = 15 * (solar_time - 12)
    
    # Convert to radians
    lat_rad = math.radians(latitude)
    dec_rad = math.radians(declination)
    hour_rad = math.radians(hour_angle)
    
    # Solar elevation
    sin_elevation = (
        math.sin(lat_rad) * math.sin(dec_rad) + 
        math.cos(lat_rad) * math.cos(dec_rad) * math.cos(hour_rad)
    )
    # Clamp to valid domain for asin
    sin_elevation = max(-1.0, min(1.0, sin_elevation))
    elevation = math.asin(sin_elevation)
    
    # Solar azimuth
    cos_elevation = math.cos(elevation)
    if abs(cos_elevation) < 1e-8:  # Near vertical sun
        azimuth_deg = 180.0  # Default to south
    else:
        cos_azimuth = (
            (math.sin(dec_rad) * math.cos(lat_rad) - 
             math.cos(dec_rad) * math.sin(lat_rad) * math.cos(hour_rad)) / 
            cos_elevation
        )
        # Clamp to valid domain for acos
        cos_azimuth = max(-1.0, min(1.0, cos_azimuth))
        azimuth = math.acos(cos_azimuth)
        azimuth_deg = math.degrees(azimuth)
    
    # Convert elevation to degrees
    elevation_deg = math.degrees(elevation)
    
    # Adjust azimuth for afternoon (only if it was calculated above)
    if abs(cos_elevation) >= 1e-8 and solar_time > 12:
        azimuth_deg = 360 - azimuth_deg
        
    return max(0, elevation_deg), azimuth_deg

def air_mass(solar_elevation: float) -> float:
    """
    Calculate air mass from solar elevation angle.
    
    Args:
        solar_elevation: Solar elevation angle in degrees
        
    Returns:
        Air mass value
    """
    if solar_elevation <= 0:
        return 10.0  # High air mass for sun below horizon
    
    zenith = 90 - solar_elevation
    zenith_rad = math.radians(zenith)
    
    # Kasten and Young formula
    air_mass = 1 / (math.cos(zenith_rad) + 0.50572 * (96.07995 - zenith)**(-1.6364))
    
    return min(air_mass, 10.0)  # Cap at reasonable maximum

def clear_sky_irradiance(solar_elevation: float, air_mass: float, 
                        altitude: float = 0) -> Dict[str, float]:
    """
    Calculate clear sky irradiance components using simple clear sky model.
    
    Args:
        solar_elevation: Solar elevation angle in degrees
        air_mass: Air mass value
        altitude: Altitude above sea level in meters
        
    Returns:
        Dictionary with GHI, DNI, DHI components in W/m²
    """
    if solar_elevation <= 0:
        return {'ghi': 0, 'dni': 0, 'dhi': 0}
    
    # Atmospheric transmittance (simplified)
    tau_b = 0.56 * (np.exp(-0.65 * air_mass) + np.exp(-0.095 * air_mass))
    tau_d = 0.271 - 0.294 * tau_b
    
    # Solar constant with distance correction (simplified)
    extraterrestrial = 1366  # W/m²
    
    # Altitude correction
    altitude_factor = 1 + altitude / 10000  # Rough correction
    
    # Direct normal irradiance
    dni = extraterrestrial * tau_b * altitude_factor
    
    # Diffuse horizontal irradiance
    dhi = extraterrestrial * tau_d * math.sin(math.radians(solar_elevation)) * altitude_factor
    
    # Global horizontal irradiance
    ghi = dni * math.sin(math.radians(solar_elevation)) + dhi
    
    return {
        'ghi': max(0, ghi),
        'dni': max(0, dni), 
        'dhi': max(0, dhi)
    }

def pv_cell_temperature(ambient_temp: float, irradiance: float, wind_speed: float,
                       mounting: str = 'roof') -> float:
    """
    Calculate PV cell temperature using NOCT model.
    
    Args:
        ambient_temp: Ambient temperature in °C
        irradiance: Solar irradiance in W/m²
        wind_speed: Wind speed in m/s
        mounting: Mounting type ('roof', 'ground', 'pole')
        
    Returns:
        Cell temperature in °C
    """
    # NOCT values by mounting type
    noct_values = {
        'roof': 47,      # °C
        'ground': 45,
        'pole': 43
    }
    
    noct = noct_values.get(mounting, 47)
    
    # Wind factor (simplified)
    wind_factor = 1 + 0.05 * (2 - wind_speed)  # Reference is 2 m/s
    wind_factor = max(0.5, min(wind_factor, 2.0))  # Reasonable bounds
    
    # Cell temperature calculation
    cell_temp = ambient_temp + (noct - 20) * (irradiance / 800) * wind_factor
    
    return cell_temp

def battery_thermal_model(ambient_temp: float, charge_rate: float, 
                         discharge_rate: float, capacity: float) -> float:
    """
    Simple battery thermal model.
    
    Args:
        ambient_temp: Ambient temperature in °C
        charge_rate: Charging rate in kW (positive)
        discharge_rate: Discharging rate in kW (positive)
        capacity: Battery capacity in kWh
        
    Returns:
        Battery temperature in °C
    """
    # Heat generation from inefficiency
    efficiency_loss = 0.05  # 5% heat loss
    heat_rate = (charge_rate + discharge_rate) * efficiency_loss  # kW
    
    # Thermal mass (simplified)
    thermal_mass = capacity * 0.5  # kWh -> thermal capacity proxy
    
    # Temperature rise from heat generation
    temp_rise = heat_rate / thermal_mass * 10  # Scaling factor
    
    # Battery temperature
    battery_temp = ambient_temp + temp_rise
    
    return battery_temp

def interpolate_efficiency(load_fraction: float, efficiency_curve: Dict) -> float:
    """
    Interpolate efficiency from load fraction using efficiency curve.
    
    Args:
        load_fraction: Load fraction (0-1)
        efficiency_curve: Dictionary with 'load_fraction' and 'efficiency' arrays
        
    Returns:
        Interpolated efficiency (0-1)
    """
    load_fractions = efficiency_curve['load_fraction']
    efficiencies = efficiency_curve['efficiency']
    
    # Bound the load fraction
    load_fraction = max(0, min(load_fraction, 1.0))
    
    # Interpolate
    efficiency = np.interp(load_fraction, load_fractions, efficiencies)
    
    return efficiency

def economic_escalation(base_value: float, years: int, escalation_rate: float) -> float:
    """
    Calculate escalated value over time.
    
    Args:
        base_value: Initial value
        years: Number of years
        escalation_rate: Annual escalation rate (e.g., 0.03 for 3%)
        
    Returns:
        Escalated value
    """
    return base_value * (1 + escalation_rate) ** years

def present_value(future_value: float, years: int, discount_rate: float) -> float:
    """
    Calculate present value of future cash flow.
    
    Args:
        future_value: Future value
        years: Number of years in future
        discount_rate: Annual discount rate (e.g., 0.06 for 6%)
        
    Returns:
        Present value
    """
    return future_value / (1 + discount_rate) ** years

def create_time_index(start_date: str, periods: int, freq: str = '15T') -> pd.DatetimeIndex:
    """
    Create a pandas DatetimeIndex for simulation.
    
    Args:
        start_date: Start date string (e.g., '2024-01-01')
        periods: Number of periods
        freq: Frequency string (e.g., '15T' for 15 minutes)
        
    Returns:
        DatetimeIndex
    """
    return pd.date_range(start=start_date, periods=periods, freq=freq)

def resample_data(data: pd.Series, target_freq: str, method: str = 'mean') -> pd.Series:
    """
    Resample time series data to different frequency.
    
    Args:
        data: Input time series with datetime index
        target_freq: Target frequency (e.g., 'H' for hourly)
        method: Resampling method ('mean', 'sum', 'max', 'min')
        
    Returns:
        Resampled time series
    """
    if method == 'mean':
        return data.resample(target_freq).mean()
    elif method == 'sum':
        return data.resample(target_freq).sum()
    elif method == 'max':
        return data.resample(target_freq).max()
    elif method == 'min':
        return data.resample(target_freq).min()
    else:
        raise ValueError(f"Unknown resampling method: {method}")

def validate_coordinates(latitude: float, longitude: float) -> bool:
    """
    Validate latitude and longitude coordinates.
    
    Args:
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        
    Returns:
        True if valid coordinates
    """
    return (-90 <= latitude <= 90) and (-180 <= longitude <= 180)

def validate_positive(value: float, name: str) -> None:
    """
    Validate that a value is positive.
    
    Args:
        value: Value to validate
        name: Name of the value for error messages
        
    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")

def validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
    """
    Validate that a value is within a specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error messages
        
    Raises:
        ValueError: If value is outside range
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")

def energy_to_power(energy_kwh: float, time_hours: float) -> float:
    """
    Convert energy to average power.
    
    Args:
        energy_kwh: Energy in kWh
        time_hours: Time period in hours
        
    Returns:
        Average power in kW
    """
    if time_hours <= 0:
        raise ValueError("Time must be positive")
    return energy_kwh / time_hours

def power_to_energy(power_kw: float, time_hours: float) -> float:
    """
    Convert power to energy.
    
    Args:
        power_kw: Power in kW
        time_hours: Time period in hours
        
    Returns:
        Energy in kWh
    """
    return power_kw * time_hours 