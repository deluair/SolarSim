"""
Basic SolarSim Usage Example

This script demonstrates how to set up and run a basic off-grid solar simulation
using the SolarSim framework.
"""

import sys
from pathlib import Path

# Add the parent directory to path to import solarsim
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from solarsim.utils.config import (
    SystemConfig, LocationConfig, PVConfig, BatteryConfig, 
    InverterConfig, LoadConfig, EconomicConfig, SimulationConfig
)
from solarsim.simulation import OffGridSolarSimulation
from solarsim.components.solar import SolarPVArray
from solarsim.components.battery import BatterySystem
from solarsim.components.inverter import Inverter
from solarsim.components.loads import LoadManager
from solarsim.data.weather import WeatherGenerator
from solarsim.data.loads import LoadProfileGenerator
from solarsim.data.economics import EconomicDataGenerator


def create_example_system_config():
    """Create an example system configuration for a medium household."""
    
    # Location: Charlotte, NC (temperate climate)
    location = LocationConfig(
        latitude=35.2271,
        longitude=-80.8431,
        altitude=200,
        timezone='America/New_York',
        climate_zone='temperate'
    )
    
    # 8 kW solar array with monocrystalline panels
    pv = PVConfig(
        technology='monocrystalline',
        capacity_kw=8.0,
        panel_power_w=400,
        tilt_angle=35,  # Optimal for this latitude
        azimuth_angle=180,  # South-facing
        mounting_type='roof',
        tracking=False
    )
    
    # 20 kWh LiFePO4 battery system
    battery = BatteryConfig(
        technology='lifepo4',
        capacity_kwh=20.0,
        voltage_system=48.0,
        thermal_management=True
    )
    
    # 8 kW inverter
    inverter = InverterConfig(
        capacity_kw=8.0,
        efficiency_rated=0.96,
        voltage_input=48.0,
        voltage_output=240.0,
        mppt_channels=2
    )
    
    # Medium household load profile
    load = LoadConfig(
        household_type='medium',
        daily_energy_kwh=25.0,  # 25 kWh/day
        peak_power_kw=4.0,
        seasonal_variation=0.3,  # ±30% seasonal variation
        weekend_factor=1.15,  # 15% higher on weekends
        growth_rate=0.02  # 2% annual growth
    )
    
    # Economic parameters
    economic = EconomicConfig(
        discount_rate=0.06,
        system_lifetime=25,
        federal_tax_credit=0.30,
        financing_type='cash'
    )
    
    # Simulation parameters
    simulation = SimulationConfig(
        start_date='2024-01-01',
        years=1,
        time_step_minutes=60,  # 1-hour time steps
        weather_uncertainty=True,
        monte_carlo_runs=1
    )
    
    return SystemConfig(
        location=location,
        pv=pv,
        battery=battery,
        inverter=inverter,
        load=load,
        economic=economic,
        simulation=simulation
    )


def run_component_demonstrations():
    """Demonstrate individual component functionality."""
    
    print("=== Component Demonstrations ===\n")
    
    # 1. Weather Data Generation
    print("1. Generating Weather Data...")
    location = LocationConfig(35.2271, -80.8431, climate_zone='temperate')
    weather_gen = WeatherGenerator(
        location={
            'latitude': location.latitude,
            'longitude': location.longitude,
            'altitude': location.altitude
        },
        climate_zone=location.climate_zone
    )
    
    weather_data = weather_gen.generate_annual_weather(
        year=2024,
        time_step_hours=1.0
    )
    
    print(f"Generated {len(weather_data)} hours of weather data")
    print(f"Average irradiance: {weather_data['ghi'].mean():.0f} W/m²")
    print(f"Average temperature: {weather_data['temp_air'].mean():.1f}°C")
    print()
    
    # 2. Solar Array Performance
    print("2. Solar Array Performance...")
    pv_config = PVConfig(capacity_kw=8.0, technology='monocrystalline')
    solar_array = SolarPVArray(pv_config)
    
    # Calculate power output for a sunny day
    irradiance = 800  # W/m²
    ambient_temp = 25  # °C
    wind_speed = 3    # m/s
    
    solar_output = solar_array.calculate_power_output(irradiance, ambient_temp, wind_speed)
    print(f"Solar output at {irradiance} W/m²: {solar_output['dc_power_kw']:.2f} kW")
    print(f"Panel efficiency: {solar_array.efficiency*100:.1f}%")
    print()
    
    # 3. Battery System
    print("3. Battery System...")
    battery_config = BatteryConfig(capacity_kwh=20.0, technology='lifepo4')
    battery = BatterySystem(battery_config)
    
    print(f"Battery capacity: {battery.nominal_capacity_kwh} kWh")
    print(f"Usable capacity: {battery.get_usable_capacity():.1f} kWh")
    print(f"Round-trip efficiency: {battery.round_trip_efficiency*100:.0f}%")
    print()
    
    # 4. Load Profile Generation
    print("4. Load Profile Generation...")
    load_config = LoadConfig(household_type='medium')
    load_manager = LoadManager(load_config)
    
    # Generate load for a specific time
    timestamp: pd.Timestamp = pd.Timestamp('2024-07-15 12:00:00')
    load_result = load_manager.calculate_instantaneous_load(timestamp, ambient_temp=30)
    
    print(f"Midday load in summer: {load_result['total_load_kw']:.2f} kW")
    print(f"Critical loads: {load_result['critical_load_kw']:.2f} kW")
    print()
    
    # 5. Economic Data
    print("5. Economic Analysis...")
    econ_gen = EconomicDataGenerator()
    econ_data = econ_gen.generate_cost_projections(scenario_name='baseline', projection_years=5)
    
    current_costs = econ_data.iloc[0]
    future_costs = econ_data.iloc[-1]
    print(f"Current solar cost: ${current_costs['pv_cost_per_watt']:.2f}/W")
    print(f"Current battery cost: ${current_costs['battery_cost_per_kwh']:.0f}/kWh")
    print(f"Projected {future_costs['year']} solar cost: ${future_costs['pv_cost_per_watt']:.2f}/W")
    print()


def run_basic_simulation():
    """Run a basic simulation and display results."""
    
    print("=== Running Basic Simulation ===\n")
    
    # Create system configuration
    config = create_example_system_config()
    print("System Configuration:")
    print(config.summary())
    
    # Initialize simulation
    sim = OffGridSolarSimulation(config)
    
    # Run simulation for 3 months
    print("Running 3-month simulation...")
    results = sim.run_simulation(
        start_date='2024-06-01',
        duration_years=0.25,  # 3 months
        time_step_minutes=60,
        include_degradation=False,
        monte_carlo_runs=1
    )
    
    # Display results
    print("\n" + sim.get_summary_report())
    
    # Export results
    output_dir = Path(__file__).parent / 'simulation_results'
    sim.export_results(str(output_dir), format='csv')
    print(f"\nResults exported to: {output_dir}")
    
    return sim, results


def create_visualization_plots(sim, results):
    """Create visualization plots of simulation results."""
    
    if sim.time_series_data is None:
        print("No time series data available for plotting")
        return
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SolarSim Results - 3 Month Simulation', fontsize=16)
    
    data = sim.time_series_data
    
    # Plot 1: Power flows
    axes[0, 0].plot(data.index, data['solar_power_kw'], label='Solar Generation', alpha=0.8)
    axes[0, 0].plot(data.index, data['load_power_kw'], label='Load Demand', alpha=0.8)
    axes[0, 0].set_ylabel('Power (kW)')
    axes[0, 0].set_title('Power Generation vs Load')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Battery state of charge
    axes[0, 1].plot(data.index, data['battery_soc'] * 100, color='green', alpha=0.8)
    axes[0, 1].set_ylabel('Battery SOC (%)')
    axes[0, 1].set_title('Battery State of Charge')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 100)
    
    # Plot 3: System efficiency
    axes[1, 0].plot(data.index, data['system_efficiency'] * 100, color='orange', alpha=0.8)
    axes[1, 0].set_ylabel('Efficiency (%)')
    axes[1, 0].set_title('System Efficiency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Weather conditions
    axes[1, 1].plot(data.index, data['ghi'], label='Solar Irradiance', alpha=0.8)
    ax2 = axes[1, 1].twinx()
    ax2.plot(data.index, data['temp_air'], color='red', label='Temperature', alpha=0.8)
    axes[1, 1].set_ylabel('Irradiance (W/m²)')
    ax2.set_ylabel('Temperature (°C)', color='red')
    axes[1, 1].set_title('Weather Conditions')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(__file__).parent / 'simulation_results' / 'simulation_plots.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {output_path}")
    
    return fig


def compare_scenarios():
    """Compare different system configurations."""
    
    print("\n=== Scenario Comparison ===\n")
    
    base_config = create_example_system_config()
    scenarios = {}
    
    # Scenario 1: Small system
    small_config = create_example_system_config()
    small_config.pv.capacity_kw = 4.0
    small_config.battery.capacity_kwh = 10.0
    small_config.inverter.capacity_kw = 4.0
    small_config.load.household_type = 'small'
    small_config.load.daily_energy_kwh = 12.0
    
    # Scenario 2: Large system  
    large_config = create_example_system_config()
    large_config.pv.capacity_kw = 12.0
    large_config.battery.capacity_kwh = 30.0
    large_config.inverter.capacity_kw = 10.0
    large_config.load.household_type = 'large'
    large_config.load.daily_energy_kwh = 40.0
    
    scenarios = {
        'Small System (4kW/10kWh)': small_config,
        'Medium System (8kW/20kWh)': base_config,
        'Large System (12kW/30kWh)': large_config
    }
    
    results_comparison = []
    
    for scenario_name, config in scenarios.items():
        print(f"Running {scenario_name}...")
        sim = OffGridSolarSimulation(config)
        
        # Quick 1-month simulation
        results = sim.run_simulation(
            start_date='2024-07-01',
            duration_years=1/12,  # 1 month
            time_step_minutes=60,
            include_degradation=False
        )
        
        # Collect key metrics
        perf = sim.performance_metrics
        econ = sim.economic_metrics
        
        scenario_results = {
            'Scenario': scenario_name,
            'PV Capacity (kW)': config.pv.capacity_kw,
            'Battery Capacity (kWh)': config.battery.capacity_kwh,
            'System Availability (%)': perf['system_availability_percent'],
            'Capacity Factor (%)': perf['capacity_factor_percent'],
            'Total Cost ($)': econ['total_system_cost_usd'],
            'Cost per kW ($/kW)': econ['cost_per_kw_installed'],
            'Simple Payback (years)': econ['simple_payback_years']
        }
        
        results_comparison.append(scenario_results)
    
    # Display comparison table
    comparison_df = pd.DataFrame(results_comparison)
    print("\nScenario Comparison Results:")
    print("=" * 80)
    print(comparison_df.to_string(index=False, float_format='%.1f'))
    
    return comparison_df


def main():
    """Main execution function."""
    
    print("SolarSim Framework Demonstration")
    print("=" * 40)
    print(f"Simulation started at: {datetime.now()}")
    print()
    
    try:
        # 1. Component demonstrations
        run_component_demonstrations()
        
        # 2. Basic simulation
        sim, results = run_basic_simulation()
        
        # 3. Create visualizations
        try:
            create_visualization_plots(sim, results)
        except Exception as e:
            print(f"Visualization error (matplotlib not available?): {e}")
        
        # 4. Scenario comparison
        comparison_df = compare_scenarios()
        
        # 5. Save comparison results
        output_dir = Path(__file__).parent / 'simulation_results'
        comparison_df.to_csv(output_dir / 'scenario_comparison.csv', index=False)
        
        print(f"\nDemonstration completed successfully!")
        print(f"All results saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()