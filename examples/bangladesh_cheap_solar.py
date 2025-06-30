import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from solarsim import OffGridSolarSimulation
from solarsim.utils.config import SystemConfig, LocationConfig, PVConfig, BatteryConfig, InverterConfig, LoadConfig


def create_bangladesh_cheap_config():
    config = SystemConfig()
    config.location = LocationConfig(
        latitude=23.8103, longitude=90.4125, altitude=4, climate_zone='tropical'
    )
    config.pv = PVConfig(
        capacity_kw=6.0,
        technology='polycrystalline',
        tilt_angle=20,
        azimuth_angle=180
    )
    config.battery = BatteryConfig(
        capacity_kwh=10.0,
        technology='lifepo4'
    )
    config.inverter = InverterConfig(
        capacity_kw=4.0
    )
    config.load = LoadConfig(
        household_type='large',
        daily_energy_kwh=18.0,
        peak_power_kw=3.5,
        seasonal_variation=True
    )
    return config


def plot_bangladesh_results(sim):
    data = sim.time_series_data
    if data is None:
        print("No time series data available for plotting")
        return
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle('Bangladesh Cheap Solar System - 3 Month Simulation', fontsize=16)
    # Solar and Load
    axes[0].plot(data.index, data['solar_power_kw'], label='Solar Generation', color='gold', alpha=0.8)
    axes[0].plot(data.index, data['load_power_kw'], label='Load Demand', color='blue', alpha=0.7)
    axes[0].set_ylabel('Power (kW)')
    axes[0].set_title('Power Generation vs Load')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # Battery SOC
    axes[1].plot(data.index, data['battery_soc'] * 100, color='green', alpha=0.8)
    axes[1].set_ylabel('Battery SOC (%)')
    axes[1].set_title('Battery State of Charge')
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)
    # Weather
    axes[2].plot(data.index, data['ghi'], label='Solar Irradiance', color='orange', alpha=0.8)
    ax2 = axes[2].twinx()
    ax2.plot(data.index, data['temp_air'], color='red', label='Temperature', alpha=0.7)
    axes[2].set_ylabel('Irradiance (W/m²)')
    ax2.set_ylabel('Temperature (°C)', color='red')
    axes[2].set_title('Weather Conditions')
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = Path(__file__).parent / 'simulation_results' / 'bangladesh_cheap_solar_plots.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Bangladesh scenario plots saved to: {output_path}")
    return fig


def main():
    print("Bangladesh Cheap Solar Example\n" + "="*40)
    config = create_bangladesh_cheap_config()
    print(config.summary())
    sim = OffGridSolarSimulation(config)
    print("Running 3-month simulation...")
    results = sim.run_simulation(
        start_date='2024-06-01',
        duration_years=0.25,
        time_step_minutes=60,
        include_degradation=False,
        monte_carlo_runs=1
    )
    print("\n" + sim.get_summary_report())
    output_dir = Path(__file__).parent / 'simulation_results'
    sim.export_results(str(output_dir), format='csv')
    print(f"Results exported to: {output_dir}")
    plot_bangladesh_results(sim)

if __name__ == "__main__":
    main() 