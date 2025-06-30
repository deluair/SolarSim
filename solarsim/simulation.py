"""
Main simulation engine for SolarSim.

This module provides the core OffGridSolarSimulation class that orchestrates
all system components and performs comprehensive energy system simulations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path

from .components.solar import SolarPVArray
from .components.battery import BatterySystem
from .components.inverter import Inverter
from .components.loads import LoadManager
from .data.weather import WeatherGenerator
from .data.loads import LoadProfileGenerator
from .data.economics import EconomicDataGenerator
from .utils.config import SystemConfig
from .utils.helpers import create_time_index, validate_range


class OffGridSolarSimulation:
    """
    Comprehensive off-grid solar simulation with BESS.
    
    This class orchestrates all system components and performs detailed
    energy balance simulations with economic analysis and reliability metrics.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the simulation.
        
        Args:
            config: Complete system configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.solar_array = SolarPVArray(config.pv)
        self.battery_system = BatterySystem(config.battery)
        self.inverter = Inverter(config.inverter)
        self.load_manager = LoadManager(config.load)
        
        # Initialize data generators
        self.weather_generator = WeatherGenerator(
            location={
                'latitude': config.location.latitude,
                'longitude': config.location.longitude,
                'altitude': config.location.altitude
            },
            climate_zone=config.location.climate_zone
        )
        self.economic_generator = EconomicDataGenerator()
        
        # Simulation results storage
        self.results = {}
        self.time_series_data = None
        self.performance_metrics = {}
        self.economic_metrics = {}
        
        # System state tracking
        self.system_age_years = 0.0
        self.total_energy_generated = 0.0  # kWh
        self.total_energy_consumed = 0.0   # kWh
        self.total_energy_stored = 0.0     # kWh
        self.unmet_load_hours = 0          # Count of hours with unmet load
        
        self.logger.info(f"Initialized OffGridSolarSimulation for simulation")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up simulation logging."""
        logger = logging.getLogger(f'SolarSim.{id(self)}')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def run_simulation(self,
                      start_date: str = '2024-01-01',
                      duration_years: float = 1.0,
                      time_step_minutes: int = 60,
                      include_degradation: bool = True,
                      monte_carlo_runs: int = 1) -> Dict[str, Any]:
        """
        Run comprehensive system simulation.
        
        Args:
            start_date: Simulation start date (YYYY-MM-DD)
            duration_years: Simulation duration in years
            time_step_minutes: Time step in minutes (15, 30, or 60)
            include_degradation: Whether to include component degradation
            monte_carlo_runs: Number of Monte Carlo simulation runs
            
        Returns:
            Complete simulation results dictionary
        """
        self.logger.info(f"Starting simulation: {duration_years} years, {time_step_minutes}-min steps")
        
        # Generate weather data
        weather_data = self.weather_generator.generate_annual_weather(
            year=int(start_date.split('-')[0]),
            time_step_hours=time_step_minutes / 60
        )
        
        # Run simulation (with Monte Carlo if specified)
        if monte_carlo_runs > 1:
            results = self._run_monte_carlo_simulation(
                weather_data, duration_years, include_degradation, monte_carlo_runs
            )
        else:
            results = self._run_single_simulation(
                weather_data, duration_years, include_degradation
            )
        
        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics(results)
        
        # Calculate economic metrics
        self.economic_metrics = self._calculate_economic_metrics(
            results, duration_years
        )
        
        # Store complete results
        self.results = {
            'time_series': results,
            'performance_metrics': self.performance_metrics,
            'economic_metrics': self.economic_metrics,
            'system_config': self.config.to_dict(),
            'simulation_parameters': {
                'start_date': start_date,
                'duration_years': duration_years,
                'time_step_minutes': time_step_minutes,
                'include_degradation': include_degradation,
                'monte_carlo_runs': monte_carlo_runs
            }
        }
        
        self.logger.info("Simulation completed successfully")
        return self.results
    
    def _run_single_simulation(self,
                             weather_data: pd.DataFrame,
                             duration_years: float,
                             include_degradation: bool) -> pd.DataFrame:
        """Run a single deterministic simulation."""
        
        timestamps = weather_data.index
        n_steps = len(timestamps)
        
        # Initialize result arrays
        results = pd.DataFrame(index=timestamps)
        
        # Copy weather data
        for col in weather_data.columns:
            results[col] = weather_data[col]
        
        # Initialize storage arrays
        solar_power = np.zeros(n_steps)
        battery_soc = np.zeros(n_steps)
        battery_power = np.zeros(n_steps)  # Positive = charging, negative = discharging
        load_power = np.zeros(n_steps)
        unmet_load = np.zeros(n_steps)
        system_efficiency = np.zeros(n_steps)
        
        # Initialize battery SOC
        current_soc = self.battery_system.current_soc
        
        for i, timestamp in enumerate(timestamps):
            # Calculate system age for degradation
            if include_degradation:
                years_elapsed = (timestamp - timestamps[0]).total_seconds() / (365.25 * 24 * 3600)
                self.system_age_years = years_elapsed
            
            # Get weather conditions
            irradiance = weather_data.loc[timestamp, 'ghi']
            ambient_temp = weather_data.loc[timestamp, 'temp_air']
            wind_speed = weather_data.loc[timestamp, 'wind_speed']
            
            # Calculate solar generation
            solar_output = self.solar_array.calculate_power_output(
                irradiance, ambient_temp, wind_speed
            )
            solar_power[i] = solar_output['dc_power_kw']
            
            # Convert DC to AC through inverter
            ac_solar_power = self.inverter.calculate_ac_power(solar_power[i])['ac_power_kw']
            
            # Calculate load demand
            load_demand = self.load_manager.calculate_instantaneous_load(
                timestamp, ambient_temp
            )
            load_power[i] = load_demand['total_load_kw']
            
            # Energy balance and battery management
            net_power = ac_solar_power - load_power[i]  # Positive = excess, negative = deficit
            
            # Get time step in hours for battery calculations
            time_step_hours = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600 if i > 0 else (timestamps[1] - timestamps[0]).total_seconds() / 3600

            if net_power > 0:
                # Excess power - charge battery
                charge_result = self.battery_system.calculate_charge_power(
                    net_power, ambient_temp, time_step_hours
                )
                battery_power[i] = charge_result['actual_power_kw']
                current_soc = charge_result['new_soc']
                unmet_load[i] = 0.0
                
            else:
                # Power deficit - discharge battery
                required_power = abs(net_power)
                discharge_result = self.battery_system.calculate_discharge_power(
                    required_power, ambient_temp, time_step_hours
                )
                battery_power[i] = -discharge_result['actual_power_kw']
                current_soc = discharge_result['new_soc']
                
                # Check if load is fully met
                power_from_battery = discharge_result['actual_power_kw']
                total_available = ac_solar_power + power_from_battery
                unmet_load[i] = max(0, load_power[i] - total_available)
            
            battery_soc[i] = current_soc
            
            # Calculate system efficiency
            if solar_power[i] > 0:
                energy_delivered = load_power[i] - unmet_load[i]
                system_efficiency[i] = min(1.0, energy_delivered / solar_power[i])
            else:
                system_efficiency[i] = 0.0
            
            # Apply degradation if enabled (yearly intervals)
            hours_per_year = 365.25 * 24
            time_step_hours_calc = duration_years * hours_per_year / n_steps
            if include_degradation and i > 0 and i % max(1, int(hours_per_year / time_step_hours_calc)) == 0:
                self.solar_array.update_degradation(1.0) # Degrade for 1 year
                self.battery_system.update_aging(time_step_hours=hours_per_year)  # 1 year
        
        # Store results
        results['solar_power_kw'] = solar_power
        results['load_power_kw'] = load_power
        results['battery_soc'] = battery_soc
        results['battery_power_kw'] = battery_power
        results['unmet_load_kw'] = unmet_load
        results['system_efficiency'] = system_efficiency
        
        # Calculate derived metrics
        results['net_solar_power_kw'] = solar_power - load_power
        
        self.time_series_data = results
        return results
    
    def _run_monte_carlo_simulation(self,
                                  weather_data: pd.DataFrame,
                                  duration_years: float,
                                  include_degradation: bool,
                                  n_runs: int) -> pd.DataFrame:
        """Run Monte Carlo simulation with uncertainty quantification."""
        
        self.logger.info(f"Running Monte Carlo simulation with {n_runs} iterations")
        
        all_results = []
        
        for run in range(n_runs):
            # Add uncertainty to system parameters
            self._apply_parameter_uncertainty()
            
            # Run single simulation
            result = self._run_single_simulation(weather_data, duration_years, include_degradation)
            result['run_id'] = run
            all_results.append(result)
            
            if (run + 1) % max(1, n_runs // 10) == 0:
                self.logger.info(f"Completed {run + 1}/{n_runs} Monte Carlo runs")
        
        # Combine results and calculate statistics
        combined_results = pd.concat(all_results)
        
        # Calculate percentiles for key metrics
        mc_results_grouped = combined_results.groupby(combined_results.index)
        
        mc_results = pd.DataFrame({
            'solar_power_kw_mean': mc_results_grouped['solar_power_kw'].mean(),
            'solar_power_kw_std': mc_results_grouped['solar_power_kw'].std(),
            'solar_power_kw_p10': mc_results_grouped['solar_power_kw'].quantile(0.1),
            'solar_power_kw_p90': mc_results_grouped['solar_power_kw'].quantile(0.9),
            'load_power_kw_mean': mc_results_grouped['load_power_kw'].mean(),
            'battery_soc_mean': mc_results_grouped['battery_soc'].mean(),
            'unmet_load_kw_mean': mc_results_grouped['unmet_load_kw'].mean(),
            'unmet_load_kw_max': mc_results_grouped['unmet_load_kw'].max(),
        })
        
        return mc_results
    
    def _apply_parameter_uncertainty(self):
        """Apply uncertainty to system parameters for Monte Carlo analysis."""
        # Add ±5% uncertainty to solar array efficiency
        efficiency_factor = np.random.normal(1.0, 0.05)
        efficiency_factor = max(0.8, min(efficiency_factor, 1.2))
        
        # Add ±3% uncertainty to battery capacity
        capacity_factor = np.random.normal(1.0, 0.03)
        capacity_factor = max(0.9, min(capacity_factor, 1.1))
        
        # Apply temporary modifications (would need to reset after each run)
        original_efficiency = self.solar_array.efficiency
        original_capacity = self.battery_system.nominal_capacity_kwh
        
        self.solar_array.efficiency = original_efficiency * efficiency_factor
        self.battery_system.nominal_capacity_kwh = original_capacity * capacity_factor
    
    def _calculate_autonomy_hours(self,
                                battery_soc: np.ndarray,
                                load_power: np.ndarray,
                                time_step_minutes: int) -> np.ndarray:
        """Calculate energy autonomy hours at each time step."""
        autonomy_hours = np.zeros_like(battery_soc)
        
        for i in range(len(battery_soc)):
            if load_power[i] > 0:
                available_energy = (battery_soc[i] * self.battery_system.nominal_capacity_kwh * 
                                  self.battery_system.dod_max)
                autonomy_hours[i] = available_energy / load_power[i]
            else:
                autonomy_hours[i] = np.inf
        
        return autonomy_hours
    
    def _calculate_performance_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        time_step_hours = self.config.simulation.time_step_minutes / 60
        
        metrics = {
            # Energy metrics
            'total_solar_generation_kwh': results['solar_power_kw'].sum() * time_step_hours,
            'total_load_consumption_kwh': results['load_power_kw'].sum() * time_step_hours,
            'total_unmet_load_kwh': results['unmet_load_kw'].sum() * time_step_hours,
            
            # Reliability metrics
            'system_availability_percent': (1 - (results['unmet_load_kw'] > 0).sum() / len(results)) * 100,
            'unmet_load_hours': (results['unmet_load_kw'] > 0).sum() * time_step_hours,
            'max_unmet_load_kw': results['unmet_load_kw'].max(),
            
            # Efficiency metrics
            'average_system_efficiency_percent': results['system_efficiency'].mean() * 100,
            'capacity_factor_percent': (results['solar_power_kw'].mean() / 
                                      self.solar_array.capacity_kw) * 100,
            
            # Battery metrics
            'average_battery_soc_percent': results['battery_soc'].mean() * 100,
            'min_battery_soc_percent': results['battery_soc'].min() * 100,
            'battery_cycles': self._estimate_battery_cycles(np.asarray(results['battery_power_kw'])),
            
            # Load metrics
            'peak_load_kw': results['load_power_kw'].max(),
            'average_load_kw': results['load_power_kw'].mean(),
            'load_factor': results['load_power_kw'].mean() / max(results['load_power_kw'].max(), 0.001),
        }
        
        return metrics
    
    def _calculate_economic_metrics(self, 
                                  results: pd.DataFrame,
                                  duration_years: float) -> Dict[str, float]:
        """Calculate economic performance metrics."""
        
        # Get economic parameters
        econ_data = self.economic_generator.generate_cost_projections(
            scenario_name='baseline', projection_years=25
        )
        
        baseline_costs = econ_data.iloc[0]
        
        # System costs
        solar_cost = (self.solar_array.capacity_kw * 1000 * 
                     baseline_costs['pv_cost_per_watt'])
        battery_cost = (self.battery_system.nominal_capacity_kwh * 
                       baseline_costs['battery_cost_per_kwh'])
        inverter_cost = (self.inverter.capacity_kw * 
                        baseline_costs['inverter_cost_per_kw'])
        installation_cost = (solar_cost + battery_cost + inverter_cost) * (baseline_costs['installation_multiplier']-1)
        
        total_system_cost = solar_cost + battery_cost + inverter_cost + installation_cost
        
        # Energy value (avoided utility costs)
        time_step_hours = self.config.simulation.time_step_minutes / 60
        annual_energy_kwh = results['load_power_kw'].sum() * time_step_hours * (365.25 / duration_years)
        utility_rate = 0.12  # $/kWh baseline
        annual_energy_value = annual_energy_kwh * utility_rate
        
        # Simple economic metrics
        simple_payback = total_system_cost / annual_energy_value if annual_energy_value > 0 else np.inf
        
        # LCOE calculation
        total_energy_generated = results['solar_power_kw'].sum() * time_step_hours
        lcoe = total_system_cost / total_energy_generated if total_energy_generated > 0 else np.inf
        
        economic_metrics = {
            'total_system_cost_usd': total_system_cost,
            'solar_cost_usd': solar_cost,
            'battery_cost_usd': battery_cost,
            'inverter_cost_usd': inverter_cost,
            'installation_cost_usd': installation_cost,
            'annual_energy_value_usd': annual_energy_value,
            'simple_payback_years': simple_payback,
            'lcoe_usd_per_kwh': lcoe,
            'cost_per_kw_installed': total_system_cost / self.solar_array.capacity_kw,
            'cost_per_kwh_storage': battery_cost / self.battery_system.nominal_capacity_kwh
        }
        
        return economic_metrics
    
    def _estimate_battery_cycles(self, battery_power: np.ndarray) -> float:
        """Estimate battery cycle count from power time series."""
        # Simple cycle counting - look for charge/discharge transitions
        charge_discharge_switches = np.diff(np.sign(battery_power))
        # Each full cycle = 2 switches (charge to discharge and back)
        estimated_cycles = np.sum(np.abs(charge_discharge_switches)) / 4
        return estimated_cycles
    
    def export_results(self, output_path: str, format: str = 'csv') -> None:
        """
        Export simulation results to files.
        
        Args:
            output_path: Directory to save results
            format: Export format ('csv', 'json', 'excel')
        """
        if self.results is None:
            raise ValueError("No simulation results to export. Run simulation first.")
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            # Export time series data
            if self.time_series_data is not None:
                self.time_series_data.to_csv(output_dir / 'time_series_data.csv')
            
            # Export metrics
            pd.DataFrame([self.performance_metrics]).to_csv(
                output_dir / 'performance_metrics.csv', index=False
            )
            pd.DataFrame([self.economic_metrics]).to_csv(
                output_dir / 'economic_metrics.csv', index=False
            )
            
        elif format == 'json':
            import json
            with open(output_dir / 'simulation_results.json', 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_results = self._convert_numpy_types(self.results)
                json.dump(json_results, f, indent=2, default=str)
        
        elif format == 'excel':
            with pd.ExcelWriter(output_dir / 'simulation_results.xlsx') as writer:
                if self.time_series_data is not None:
                    self.time_series_data.to_excel(writer, sheet_name='TimeSeries')
                pd.DataFrame([self.performance_metrics]).to_excel(
                    writer, sheet_name='Performance', index=False
                )
                pd.DataFrame([self.economic_metrics]).to_excel(
                    writer, sheet_name='Economics', index=False
                )
        
        self.logger.info(f"Results exported to {output_dir} in {format} format")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        else:
            return obj
    
    def get_summary_report(self) -> str:
        """Generate a text summary report of simulation results."""
        if not self.results:
            return "No simulation results available. Run simulation first."
        
        perf = self.performance_metrics
        econ = self.economic_metrics
        
        report = f"""
SolarSim Simulation Results Summary
==================================

System Configuration:
- Solar Array: {self.solar_array.capacity_kw:.1f} kW, {self.solar_array.technology}
- Battery: {self.battery_system.nominal_capacity_kwh:.1f} kWh, {self.battery_system.technology}
- Inverter: {self.inverter.capacity_kw:.1f} kW

Performance Metrics:
- Total Solar Generation: {perf['total_solar_generation_kwh']:.0f} kWh
- Total Load Consumption: {perf['total_load_consumption_kwh']:.0f} kWh
- System Availability: {perf['system_availability_percent']:.1f}%
- Average System Efficiency: {perf['average_system_efficiency_percent']:.1f}%
- Capacity Factor: {perf['capacity_factor_percent']:.1f}%
- Unmet Load Hours: {perf['unmet_load_hours']:.1f} hours

Battery Performance:
- Average SOC: {perf['average_battery_soc_percent']:.1f}%
- Minimum SOC: {perf['min_battery_soc_percent']:.1f}%
- Estimated Cycles: {perf['battery_cycles']:.0f}

Economic Metrics:
- Total System Cost: ${econ['total_system_cost_usd']:,.0f}
- Simple Payback: {econ['simple_payback_years']:.1f} years
- LCOE: ${econ['lcoe_usd_per_kwh']:.3f}/kWh
- Cost per kW Installed: ${econ['cost_per_kw_installed']:,.0f}/kW

Load Profile:
- Peak Load: {perf['peak_load_kw']:.2f} kW
- Average Load: {perf['average_load_kw']:.2f} kW
- Load Factor: {perf['load_factor']:.2f}
"""
        return report
