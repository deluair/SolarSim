"""
Power inverter component for SolarSim.

Contains the Inverter class for modeling DC-AC power conversion
with realistic efficiency curves and operating constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import math

from ..utils.constants import INVERTER_EFFICIENCY_CURVE
from ..utils.helpers import interpolate_efficiency


class Inverter:
    """
    Power inverter model with realistic efficiency characteristics.
    
    Models DC-AC conversion with variable efficiency curves,
    power limitations, and thermal effects.
    """
    
    def __init__(self, config):
        """
        Initialize inverter.
        
        Args:
            config: InverterConfig object with inverter parameters
        """
        self.config = config
        self.capacity_kw = config.capacity_kw
        self.efficiency_rated = config.efficiency_rated
        self.voltage_input = config.voltage_input
        self.voltage_output = config.voltage_output
        self.frequency = config.frequency
        self.mppt_channels = config.mppt_channels
        
        # State variables
        self.temperature = 25.0  # °C
        self.total_energy_converted = 0.0  # kWh
        self.operating_hours = 0.0
        self.fault_status = False
        
        # Efficiency curve (load fraction vs efficiency)
        self.efficiency_curve = INVERTER_EFFICIENCY_CURVE.copy()
        
    def calculate_ac_power(self,
                          dc_power_kw: float,
                          ambient_temp: float = 25.0,
                          time_step_hours: float = 0.25) -> Dict[str, float]:
        """
        Calculate AC power output from DC input.
        
        Args:
            dc_power_kw: DC power input in kW
            ambient_temp: Ambient temperature in °C
            time_step_hours: Time step duration in hours
            
        Returns:
            Dictionary with AC power and conversion metrics
        """
        if dc_power_kw <= 0 or self.fault_status:
            return {
                'ac_power_kw': 0,
                'efficiency': 0,
                'power_loss_kw': 0,
                'load_fraction': 0,
                'clipping': False,
                'temperature_derating': False,
                'inverter_temperature': ambient_temp
            }
        
        # Temperature effects on inverter
        self.temperature = self._calculate_inverter_temperature(
            ambient_temp, dc_power_kw
        )
        temp_derating = self._get_temperature_derating(self.temperature)
        
        # Available capacity considering temperature
        available_capacity = self.capacity_kw * temp_derating
        
        # Check for clipping (DC power > inverter capacity)
        clipping = dc_power_kw > available_capacity
        effective_dc_power = min(dc_power_kw, available_capacity)
        
        # Calculate load fraction for efficiency lookup
        load_fraction = effective_dc_power / self.capacity_kw
        
        # Get efficiency from curve
        efficiency = interpolate_efficiency(load_fraction, self.efficiency_curve)
        
        # Apply rated efficiency scaling
        efficiency *= (self.efficiency_rated / 0.96)  # Scale to rated efficiency
        
        # Calculate AC power output
        ac_power_kw = effective_dc_power * efficiency
        power_loss_kw = effective_dc_power - ac_power_kw
        
        # Update operating statistics
        if ac_power_kw > 0:
            self.operating_hours += time_step_hours
            self.total_energy_converted += ac_power_kw * time_step_hours
        
        return {
            'ac_power_kw': ac_power_kw,
            'efficiency': efficiency,
            'power_loss_kw': power_loss_kw,
            'load_fraction': load_fraction,
            'clipping': clipping,
            'temperature_derating': temp_derating < 1.0,
            'inverter_temperature': self.temperature
        }
    
    def calculate_standby_losses(self, time_step_hours: float = 0.25) -> float:
        """
        Calculate standby power losses when inverter is on but not converting.
        
        Args:
            time_step_hours: Time step duration in hours
            
        Returns:
            Standby energy loss in kWh
        """
        # Typical standby power: 10-30W for residential inverters
        standby_power_kw = 0.02  # 20W
        standby_energy_kwh = standby_power_kw * time_step_hours
        
        return standby_energy_kwh
    
    def _calculate_inverter_temperature(self, 
                                      ambient_temp: float,
                                      dc_power_kw: float) -> float:
        """
        Calculate inverter temperature based on ambient and power losses.
        
        Args:
            ambient_temp: Ambient temperature in °C
            dc_power_kw: DC power input in kW
            
        Returns:
            Inverter temperature in °C
        """
        # Estimate power losses (simplified)
        load_fraction = min(dc_power_kw / self.capacity_kw, 1.0)
        efficiency = interpolate_efficiency(load_fraction, self.efficiency_curve)
        power_loss_kw = dc_power_kw * (1 - efficiency)
        
        # Temperature rise due to losses (simplified thermal model)
        # Assume 10°C rise per kW of losses
        temp_rise_per_kw = 10.0
        temp_rise = power_loss_kw * temp_rise_per_kw
        
        # Consider thermal mass and cooling
        if dc_power_kw > 0.1 * self.capacity_kw:  # Active cooling kicks in
            cooling_factor = 0.7  # 30% reduction in temperature rise
            temp_rise *= cooling_factor
        
        inverter_temp = ambient_temp + temp_rise
        
        return inverter_temp
    
    def _get_temperature_derating(self, inverter_temp: float) -> float:
        """
        Calculate temperature derating factor.
        
        Args:
            inverter_temp: Inverter temperature in °C
            
        Returns:
            Derating factor (0-1)
        """
        # Most inverters start derating around 60°C
        derating_start_temp = 60.0
        max_operating_temp = 85.0
        
        if inverter_temp <= derating_start_temp:
            return 1.0
        elif inverter_temp >= max_operating_temp:
            return 0.0  # Shutdown
        else:
            # Linear derating between start and max temperature
            derating_range = max_operating_temp - derating_start_temp
            temp_above_start = inverter_temp - derating_start_temp
            derating_factor = 1.0 - (temp_above_start / derating_range)
            return max(0.0, derating_factor)
    
    def check_grid_conditions(self,
                            voltage: float = 240.0,
                            frequency: float = 60.0) -> Dict[str, bool]:
        """
        Check if grid conditions are within acceptable ranges.
        
        Args:
            voltage: Grid voltage in V
            frequency: Grid frequency in Hz
            
        Returns:
            Dictionary with grid condition status
        """
        # Standard grid limits for residential systems
        voltage_limits = (0.88 * self.voltage_output, 1.10 * self.voltage_output)
        frequency_limits = (59.3, 60.5) if self.frequency == 60 else (49.5, 50.2)
        
        voltage_ok = voltage_limits[0] <= voltage <= voltage_limits[1]
        frequency_ok = frequency_limits[0] <= frequency <= frequency_limits[1]
        
        grid_ok = voltage_ok and frequency_ok
        
        return {
            'grid_connected': grid_ok,
            'voltage_ok': voltage_ok,
            'frequency_ok': frequency_ok,
            'voltage_value': voltage,
            'frequency_value': frequency
        }
    
    def simulate_mppt_tracking(self,
                              dc_voltages: list,
                              dc_currents: list) -> Dict[str, float]:
        """
        Simulate MPPT (Maximum Power Point Tracking) operation.
        
        Args:
            dc_voltages: List of DC voltages for each MPPT channel
            dc_currents: List of DC currents for each MPPT channel
            
        Returns:
            Dictionary with MPPT performance metrics
        """
        if len(dc_voltages) != self.mppt_channels or len(dc_currents) != self.mppt_channels:
            raise ValueError(f"Expected {self.mppt_channels} voltage/current values")
        
        total_power = 0.0
        channel_powers = []
        
        for i in range(self.mppt_channels):
            # Calculate power for each channel
            channel_power = dc_voltages[i] * dc_currents[i] / 1000  # Convert to kW
            channel_powers.append(channel_power)
            total_power += channel_power
        
        # MPPT efficiency (typically 99%+)
        mppt_efficiency = 0.995
        optimized_power = total_power * mppt_efficiency
        
        # Check for mismatch losses between channels
        if len(channel_powers) > 1:
            avg_power = sum(channel_powers) / len(channel_powers)
            mismatch_loss = sum(abs(p - avg_power) for p in channel_powers) / total_power
        else:
            mismatch_loss = 0.0
        
        return {
            'total_dc_power_kw': optimized_power,
            'channel_powers_kw': channel_powers,
            'mppt_efficiency': mppt_efficiency,
            'mismatch_loss_fraction': mismatch_loss
        }
    
    def set_fault_status(self, fault: bool, fault_type: str = '') -> None:
        """
        Set inverter fault status.
        
        Args:
            fault: Fault status (True = faulted)
            fault_type: Type of fault for logging
        """
        self.fault_status = fault
        if fault:
            print(f"Inverter fault: {fault_type}")
    
    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get summary of inverter performance metrics.
        
        Returns:
            Dictionary with performance summary
        """
        avg_efficiency = 0.0
        if self.operating_hours > 0:
            # Estimate average efficiency from energy conversion
            avg_efficiency = self.total_energy_converted / (
                self.total_energy_converted / 0.95  # Assume 95% for estimation
            )
        
        return {
            'capacity_kw': self.capacity_kw,
            'efficiency_rated': self.efficiency_rated,
            'total_energy_converted_kwh': self.total_energy_converted,
            'operating_hours': self.operating_hours,
            'average_efficiency': avg_efficiency,
            'current_temperature': self.temperature,
            'fault_status': self.fault_status,
            'mppt_channels': self.mppt_channels
        }
    
    def reset_statistics(self) -> None:
        """Reset operating statistics."""
        self.total_energy_converted = 0.0
        self.operating_hours = 0.0
        self.fault_status = False
    
    def __str__(self) -> str:
        """String representation of the inverter."""
        return (
            f"Inverter({self.capacity_kw:.1f}kW, "
            f"efficiency={self.efficiency_rated*100:.1f}%, "
            f"MPPT channels={self.mppt_channels})"
        ) 