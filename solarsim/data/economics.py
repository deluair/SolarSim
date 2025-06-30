"""
Economic data generation module for SolarSim.

Contains classes and functions for generating economic parameters,
cost projections, and financial scenarios for solar system analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..utils.constants import ECONOMIC_PARAMETERS, PV_TECHNOLOGIES, BATTERY_TECHNOLOGIES
from ..utils.helpers import economic_escalation, present_value


@dataclass
class EconomicScenario:
    """Economic scenario with cost and incentive parameters."""
    name: str
    pv_cost_per_watt: float
    battery_cost_per_kwh: float
    inverter_cost_per_kw: float
    installation_cost_multiplier: float
    federal_tax_credit: float
    state_incentives: float
    electricity_rate: float
    net_metering_rate: float
    discount_rate: float
    inflation_rate: float


class EconomicDataGenerator:
    """
    Generator for economic data and cost projections.
    
    Creates realistic economic scenarios including:
    - Component cost projections
    - Electricity rate forecasts
    - Incentive schedules
    - Financing options
    """
    
    def __init__(self, base_year: int = 2025):
        """
        Initialize economic data generator.
        
        Args:
            base_year: Base year for cost projections
        """
        self.base_year = base_year
        self.scenarios = self._initialize_scenarios()
        
    def _initialize_scenarios(self) -> Dict[str, EconomicScenario]:
        """Initialize predefined economic scenarios."""
        scenarios = {
            'optimistic': EconomicScenario(
                name='Optimistic',
                pv_cost_per_watt=0.35,
                battery_cost_per_kwh=150,
                inverter_cost_per_kw=200,
                installation_cost_multiplier=1.5,
                federal_tax_credit=0.30,
                state_incentives=0.10,
                electricity_rate=0.12,
                net_metering_rate=1.0,
                discount_rate=0.05,
                inflation_rate=0.02
            ),
            'baseline': EconomicScenario(
                name='Baseline',
                pv_cost_per_watt=0.45,
                battery_cost_per_kwh=200,
                inverter_cost_per_kw=250,
                installation_cost_multiplier=2.0,
                federal_tax_credit=0.30,
                state_incentives=0.05,
                electricity_rate=0.15,
                net_metering_rate=0.95,
                discount_rate=0.06,
                inflation_rate=0.025
            ),
            'pessimistic': EconomicScenario(
                name='Pessimistic',
                pv_cost_per_watt=0.60,
                battery_cost_per_kwh=300,
                inverter_cost_per_kw=350,
                installation_cost_multiplier=2.5,
                federal_tax_credit=0.30,
                state_incentives=0.00,
                electricity_rate=0.18,
                net_metering_rate=0.75,
                discount_rate=0.08,
                inflation_rate=0.03
            )
        }
        return scenarios
    
    def generate_cost_projections(self, 
                                scenario_name: str = 'baseline',
                                projection_years: int = 25) -> pd.DataFrame:
        """
        Generate cost projections over time.
        
        Args:
            scenario_name: Economic scenario to use
            projection_years: Number of years to project
            
        Returns:
            DataFrame with cost projections by year
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.scenarios[scenario_name]
        
        years = range(self.base_year, self.base_year + projection_years)
        projections = []
        
        for year in years:
            year_offset = year - self.base_year
            
            # Technology learning curves (costs decrease over time)
            pv_learning_rate = 0.08  # 8% cost reduction per year
            battery_learning_rate = 0.12  # 12% cost reduction per year
            inverter_learning_rate = 0.05  # 5% cost reduction per year
            
            # Calculate projected costs
            pv_cost = scenario.pv_cost_per_watt * (1 - pv_learning_rate) ** year_offset
            battery_cost = scenario.battery_cost_per_kwh * (1 - battery_learning_rate) ** year_offset
            inverter_cost = scenario.inverter_cost_per_kw * (1 - inverter_learning_rate) ** year_offset
            
            # Installation costs increase with inflation
            installation_multiplier = scenario.installation_cost_multiplier * (
                1 + scenario.inflation_rate
            ) ** year_offset
            
            # Electricity rates increase with escalation
            electricity_rate = scenario.electricity_rate * (
                1 + 0.03  # 3% annual escalation
            ) ** year_offset
            
            # Net metering policies may change
            net_metering_rate = scenario.net_metering_rate
            if year_offset > 10:  # Policy changes after 10 years
                net_metering_rate *= 0.9  # 10% reduction
            
            projections.append({
                'year': year,
                'pv_cost_per_watt': pv_cost,
                'battery_cost_per_kwh': battery_cost,
                'inverter_cost_per_kw': inverter_cost,
                'installation_multiplier': installation_multiplier,
                'electricity_rate': electricity_rate,
                'net_metering_rate': net_metering_rate,
                'federal_tax_credit': scenario.federal_tax_credit if year_offset < 8 else 0.0,  # Expires 2032
                'state_incentives': scenario.state_incentives,
                'discount_rate': scenario.discount_rate,
                'inflation_rate': scenario.inflation_rate
            })
        
        return pd.DataFrame(projections)
    
    def calculate_system_costs(self,
                             pv_capacity_kw: float,
                             battery_capacity_kwh: float,
                             inverter_capacity_kw: float,
                             scenario_name: str = 'baseline',
                             installation_year: int = 2025) -> Dict[str, float]:
        """
        Calculate total system costs for given configuration.
        
        Args:
            pv_capacity_kw: PV system capacity in kW
            battery_capacity_kwh: Battery capacity in kWh
            inverter_capacity_kw: Inverter capacity in kW
            scenario_name: Economic scenario
            installation_year: Year of installation
            
        Returns:
            Dictionary with cost breakdown
        """
        # Get cost projections for installation year
        projections = self.generate_cost_projections(scenario_name)
        year_data = projections[projections['year'] == installation_year].iloc[0]
        
        # Component costs
        pv_cost = pv_capacity_kw * 1000 * year_data['pv_cost_per_watt']  # Convert kW to W
        battery_cost = battery_capacity_kwh * year_data['battery_cost_per_kwh']
        inverter_cost = inverter_capacity_kw * year_data['inverter_cost_per_kw']
        
        # Hardware subtotal
        hardware_cost = pv_cost + battery_cost + inverter_cost
        
        # Installation and soft costs
        installation_cost = hardware_cost * year_data['installation_multiplier']
        
        # Total system cost
        total_cost = hardware_cost + installation_cost
        
        # Incentives
        federal_credit = total_cost * year_data['federal_tax_credit']
        state_incentive = total_cost * year_data['state_incentives']
        total_incentives = federal_credit + state_incentive
        
        # Net cost after incentives
        net_cost = total_cost - total_incentives
        
        return {
            'pv_cost': pv_cost,
            'battery_cost': battery_cost,
            'inverter_cost': inverter_cost,
            'hardware_cost': hardware_cost,
            'installation_cost': installation_cost,
            'total_cost': total_cost,
            'federal_tax_credit': federal_credit,
            'state_incentives': state_incentive,
            'total_incentives': total_incentives,
            'net_cost': net_cost,
            'cost_per_watt': total_cost / (pv_capacity_kw * 1000),
            'net_cost_per_watt': net_cost / (pv_capacity_kw * 1000)
        }
    
    def generate_electricity_rate_forecast(self,
                                         base_rate: float,
                                         years: int = 25,
                                         escalation_rate: float = 0.03,
                                         include_variability: bool = True) -> pd.DataFrame:
        """
        Generate electricity rate forecast with time-of-use structure.
        
        Args:
            base_rate: Base electricity rate in $/kWh
            years: Number of years to forecast
            escalation_rate: Annual rate escalation
            include_variability: Include rate variability
            
        Returns:
            DataFrame with electricity rate forecast
        """
        rate_data = []
        
        for year in range(years):
            # Base rate with escalation
            escalated_rate = base_rate * (1 + escalation_rate) ** year
            
            # Time-of-use rates (typical utility structure)
            peak_rate = escalated_rate * 1.5      # Peak hours (4-9 PM)
            shoulder_rate = escalated_rate * 1.2  # Shoulder hours
            off_peak_rate = escalated_rate * 0.8  # Off-peak hours
            
            # Add variability if requested
            if include_variability:
                variability = np.random.normal(1.0, 0.05)  # ±5% variability
                peak_rate *= variability
                shoulder_rate *= variability
                off_peak_rate *= variability
            
            rate_data.append({
                'year': self.base_year + year,
                'base_rate': escalated_rate,
                'peak_rate': peak_rate,
                'shoulder_rate': shoulder_rate,
                'off_peak_rate': off_peak_rate,
                'demand_charge': 15.0 * (1 + escalation_rate) ** year  # $/kW
            })
        
        return pd.DataFrame(rate_data)
    
    def calculate_financing_options(self,
                                  system_cost: float,
                                  loan_term_years: int = 15,
                                  loan_rate: float = 0.055) -> Dict[str, Dict]:
        """
        Calculate financing options for solar system.
        
        Args:
            system_cost: Total system cost
            loan_term_years: Loan term in years
            loan_rate: Annual loan interest rate
            
        Returns:
            Dictionary with financing options
        """
        financing_options = {}
        
        # Cash purchase
        financing_options['cash'] = {
            'down_payment': system_cost,
            'monthly_payment': 0,
            'total_payments': system_cost,
            'total_interest': 0
        }
        
        # Solar loan options
        for down_payment_pct in [0.0, 0.10, 0.20]:
            down_payment = system_cost * down_payment_pct
            loan_amount = system_cost - down_payment
            
            # Calculate monthly payment
            monthly_rate = loan_rate / 12
            num_payments = loan_term_years * 12
            
            if loan_amount > 0:
                monthly_payment = loan_amount * (
                    monthly_rate * (1 + monthly_rate) ** num_payments
                ) / ((1 + monthly_rate) ** num_payments - 1)
                
                total_payments = down_payment + monthly_payment * num_payments
                total_interest = total_payments - system_cost
            else:
                monthly_payment = 0
                total_payments = system_cost
                total_interest = 0
            
            option_name = f'loan_{int(down_payment_pct*100)}pct_down'
            financing_options[option_name] = {
                'down_payment': down_payment,
                'loan_amount': loan_amount,
                'monthly_payment': monthly_payment,
                'total_payments': total_payments,
                'total_interest': total_interest,
                'loan_term_years': loan_term_years,
                'loan_rate': loan_rate
            }
        
        return financing_options
    
    def generate_economic_sensitivity_analysis(self,
                                             base_params: Dict[str, float],
                                             sensitivity_range: float = 0.2) -> pd.DataFrame:
        """
        Generate sensitivity analysis for key economic parameters.
        
        Args:
            base_params: Base economic parameters
            sensitivity_range: Range for sensitivity analysis (±20% default)
            
        Returns:
            DataFrame with sensitivity scenarios
        """
        # Key parameters for sensitivity analysis
        key_params = [
            'pv_cost_per_watt',
            'battery_cost_per_kwh',
            'electricity_rate',
            'discount_rate',
            'federal_tax_credit'
        ]
        
        scenarios = []
        
        # Base case
        base_scenario = base_params.copy()
        base_scenario['scenario'] = 'base_case'
        scenarios.append(base_scenario)
        
        # Individual parameter variations
        for param in key_params:
            if param in base_params:
                base_value = base_params[param]
                
                # Low scenario
                low_scenario = base_params.copy()
                low_scenario[param] = base_value * (1 - sensitivity_range)
                low_scenario['scenario'] = f'{param}_low'
                scenarios.append(low_scenario)
                
                # High scenario
                high_scenario = base_params.copy()
                high_scenario[param] = base_value * (1 + sensitivity_range)
                high_scenario['scenario'] = f'{param}_high'
                scenarios.append(high_scenario)
        
        return pd.DataFrame(scenarios)
    
    def get_scenario(self, scenario_name: str) -> EconomicScenario:
        """Get economic scenario by name."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        return self.scenarios[scenario_name]
    
    def add_custom_scenario(self, scenario: EconomicScenario) -> None:
        """Add custom economic scenario."""
        self.scenarios[scenario.name.lower()] = scenario
    
    def list_scenarios(self) -> List[str]:
        """List available economic scenarios."""
        return list(self.scenarios.keys())


def calculate_utility_bill_offset(annual_generation_kwh: float,
                                annual_consumption_kwh: float,
                                electricity_rate: float,
                                net_metering_rate: float = 1.0) -> Dict[str, float]:
    """
    Calculate utility bill offset from solar generation.
    
    Args:
        annual_generation_kwh: Annual solar generation
        annual_consumption_kwh: Annual electricity consumption
        electricity_rate: Electricity rate in $/kWh
        net_metering_rate: Net metering credit rate (fraction of retail rate)
        
    Returns:
        Dictionary with bill offset calculations
    """
    # Without solar
    annual_bill_without_solar = annual_consumption_kwh * electricity_rate
    
    # With solar
    net_usage = annual_consumption_kwh - annual_generation_kwh
    
    if net_usage > 0:
        # Still purchasing electricity
        annual_bill_with_solar = net_usage * electricity_rate
        excess_generation_credit = 0
    else:
        # Excess generation
        annual_bill_with_solar = 0
        excess_generation_kwh = abs(net_usage)
        excess_generation_credit = excess_generation_kwh * electricity_rate * net_metering_rate
    
    # Calculate savings
    annual_savings = annual_bill_without_solar - annual_bill_with_solar + excess_generation_credit
    percent_offset = (annual_savings / annual_bill_without_solar) * 100
    
    return {
        'annual_bill_without_solar': annual_bill_without_solar,
        'annual_bill_with_solar': annual_bill_with_solar,
        'excess_generation_credit': excess_generation_credit,
        'annual_savings': annual_savings,
        'percent_offset': min(percent_offset, 100),  # Cap at 100%
        'net_usage_kwh': max(net_usage, 0),
        'excess_generation_kwh': max(-net_usage, 0) if net_usage < 0 else 0
    }
