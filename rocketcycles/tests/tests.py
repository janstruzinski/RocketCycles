from rocketcycles import elements
from rocketcycles.fluid import RocketCycleFluid, reformat_CEA_mass_fractions
import rocketcea.cea_obj as rcea
from rocketcea.cea_obj_w_units import CEA_Obj
import unittest
import numpy as np


class TestRocketCycleFluid(unittest.TestCase):
    def test_initialization(self):
        """Test initialization by comparing RocketCycleFluid properties with CEA output"""
        # Create RocketCEA object and get properties of the mixture in the chamber
        CC = CEA_Obj(oxName="LOX", fuelName="CH4", isp_units='sec', cstar_units='m/s',
                     pressure_units='bar', temperature_units='K', sonic_velocity_units='m/s',
                     enthalpy_units='kJ/kg', density_units='kg/m^3', specific_heat_units='kJ/kg-K',
                     viscosity_units='millipoise', thermal_cond_units='W/cm-degC')

        # Cp and enthalpy are enough to check whether most calculations in RocketCycleFluid initialization are
        # correct. Gamma is not used, because it is calculated for equilibrium conditions, whereas RocketCycleFluid
        # calculates it for frozen ones.
        cea_Cp_frozen = CC.get_Chamber_Cp(Pc=800, MR=0.2, frozen=1) * 1e3  # J/kg-K
        cea_h0 = CC.get_Chamber_H(Pc=800, MR=0.2)  # kJ / kg
        T = CC.get_Tcomb(Pc=800, MR=0.2)

        # Create RocketCycleFluid object based on the products and temperature, and get their properties. This will
        # also check reformat_CEA_mass_fractions() function.
        mass_fractions = CC.get_SpeciesMassFractions(Pc=800, MR=0.2, min_fraction=1e-6)[1]
        mass_fractions = reformat_CEA_mass_fractions(mass_fractions)
        fluid = RocketCycleFluid(species=list(mass_fractions.keys()), mass_fractions=list(mass_fractions.values()),
                                 temperature=T, type="fuel", phase="gas")
        h0 = fluid.h0 / (fluid.MW * 1e-3)  # kJ / mol
        np.testing.assert_allclose([fluid.mass_Cp_frozen, h0], [cea_Cp_frozen, cea_h0], rtol=1e-2)

    def test_total_properties_functions(self):
        """A function to test all functions related to calculating total properties"""
        # To test total temperature function, we will overwrite Cp in RocketEngineFluid with some fictional value and
        # compare the results with those manually calculated
        fluid = RocketCycleFluid(species=["CO2"], mass_fractions=[1], temperature=456, type="name", phase="gas")
        fluid.mass_Cp_frozen = 1234  # J / (kg * K)
        fluid.velocity = 321  # m/s

        # Calculate desired value:
        desired_Tt = 456 + (321 ** 2) / (2 * 1234)  # K
        # Actual value:
        fluid.calculate_total_temperature()

        # Calculate total from static pressure for fictional gamma of 1.3
        fluid.gamma = 1.3
        fluid.Ps = 2.0  # bar
        # Desired value
        desired_Pt = 2 / ((456 / desired_Tt) ** (1.3 / (1.3 - 1)))  # bar
        # Actual value
        fluid.calculate_total_from_static_pressure()

        # Now calculate static from total pressure, which should not change from the assigned one earlier on
        # Desired one
        desired_Ps = fluid.Ps
        # Actual one
        fluid.Ps = None
        fluid.calculate_static_from_total_pressure()

        # Compare the values
        np.testing.assert_allclose([fluid.Tt, fluid.Pt, fluid.Ps], [desired_Tt, desired_Pt, desired_Ps])

    def test_calculate_gas_density(self):
        """A function to test gas density calculations"""
        # In CEA, gas density is calculated with ideal gas model that only takes solids/consendibles into account.
        # In RocketCycleFluid, it is also takes into account gas compressibility. However, they should be equal
        # for low pressure, high temperature combustion.

        combustion = CEA_Obj(oxName="LOX", fuelName="CH4", isp_units='sec', cstar_units='m/s',
                             pressure_units='bar', temperature_units='K', sonic_velocity_units='m/s',
                             enthalpy_units='kJ/kg', density_units='kg/m^3', specific_heat_units='kJ/kg-K',
                             viscosity_units='millipoise', thermal_cond_units='W/cm-degC')

        CEA_density = combustion.get_Chamber_Density(Pc=1, MR=0.2)
        T_combustion = combustion.get_Tcomb(Pc=1, MR=0.2)

        # Get mass fractions and create an object with them
        mass_fractions = combustion.get_SpeciesMassFractions(Pc=1, MR=0.2)[1]
        mass_fractions = reformat_CEA_mass_fractions(mass_fractions)

        # Create RocketCycleFluid object
        fluid = RocketCycleFluid(species=list(mass_fractions.keys()), mass_fractions=list(mass_fractions.values()),
                                 temperature=T_combustion, type="fuel", phase="gas")
        fluid.Ps = 1  # bar
        density = fluid.calculate_gas_density()

        np.testing.assert_allclose(density, CEA_density, rtol=1e-2)

    def test_equilibrate(self):
        """A function to test equilibrate function"""
        # This will happen by comparing combustion temperatures from CEA with separate oxygen and fuel propellants and
        # from RocketCycleFluid with mixed oxygen and fuel as monopropellants. Both problems are equilibrium problems
        # at constant hp, so for the same reactants should result in the same products (hence temperature of combustion)

        # Get temperature of combustion from CEA
        combustion = CEA_Obj(oxName="GOX", fuelName="GH2", isp_units='sec', cstar_units='m/s',
                             pressure_units='bar', temperature_units='K', sonic_velocity_units='m/s',
                             enthalpy_units='kJ/kg', density_units='kg/m^3', specific_heat_units='kJ/kg-K',
                             viscosity_units='millipoise', thermal_cond_units='W/cm-degC')
        T_combustion = combustion.get_Tcomb(Pc=1, MR=1)

        # Get temperature of combustion from RocketCycleFluid
        fluid = RocketCycleFluid(species=["O2", "H2"], mass_fractions=[0.5, 0.5], temperature=298.15, type="name",
                                 phase="gas")
        fluid.Pt = 1  # bar
        equilibrium_fluid, equilibrium_output = fluid.equilibrate()

        # Compare temperatures
        np.testing.assert_allclose(equilibrium_fluid.Ts, T_combustion, rtol=1e-2)


class TestRocketCycleElements(unittest.TestCase):
    def test_calculate_state_after_pump(self):
        """A function to test calculation of the state after pump function"""
        # We are going to assume some fictional thermophysical and compare the results with manual calculations.

        # Calculate actual values
        fluid = RocketCycleFluid(species=["H2O(L)"], mass_fractions=[1], temperature=298.15, type="fuel",
                                 phase="liquid", liquid_elasticity=1.43 * 10 ** 9,
                                 volumetric_expansion_coefficient=990 * 1e-6, density=940)
        fluid.Pt = 3  # bar
        fluid.Ps = fluid.Pt  # bar
        fluid.mass_Cp_frozen = 1000  # J / (kg * K)
        pumped_fluid, enthalpy_change = elements.calculate_state_after_pump(
            fluid=fluid, delta_P=60, efficiency=0.5)

        # Calculate desired values
        outlet_density_isothermal = fluid.density / (1 - 60e5 / fluid.liquid_elasticity)  # kg / m^3
        w_useful = (63e5 / outlet_density_isothermal) - (3e5 / fluid.density)  # J / kg
        w_total = w_useful / 0.5  # J / kg
        w_wasted = w_total - w_useful  # J / kg
        T_outlet = fluid.Ts + w_wasted / fluid.mass_Cp_frozen  # K
        outlet_density = outlet_density_isothermal / (
                1 + (T_outlet - fluid.Ts) * fluid.volumetric_expansion_coefficient)

        # Compare actual and desired values
        np.testing.assert_allclose([pumped_fluid.Ts, pumped_fluid.Pt, enthalpy_change, pumped_fluid.density],
                                   [T_outlet, 63, w_total, outlet_density])

    def test_calculate_state_after_preburner(self):
        """A function to test calculation of the state after preburner"""
        # This will be done by comparing CEA output with the function output
        # Get CEA results
        preburner = CEA_Obj(oxName="LOX", fuelName="C3H8", isp_units='sec', cstar_units='m/s',
                            pressure_units='bar', temperature_units='K', sonic_velocity_units='m/s',
                            enthalpy_units='kJ/kg', density_units='kg/m^3', specific_heat_units='kJ/kg-K',
                            viscosity_units='millipoise', thermal_cond_units='W/cm-degC', fac_CR=1.5)
        T_combustion = preburner.get_Tcomb(Pc=600, MR=50)  # K
        Cp_frozen = preburner.get_Chamber_Transport(Pc=600, MR=50, frozen=1)[0] * 1e3  # J / (kg * K)

        # Get results from the function.
        LOX = RocketCycleFluid(species=["O2(L)"], mass_fractions=[1], temperature=90.17, type="oxid",
                               phase="liquid", species_molar_Cp=[50.180])
        Propane = RocketCycleFluid(species=["C3H8(L)"], mass_fractions=[1], temperature=231.08, type="fuel",
                                   phase="liquid", species_molar_Cp=[92.974])
        preburner_CEA_output, preburner_products = elements.calculate_state_after_preburner(
            fuel=Propane, oxidizer=LOX, OF=50, preburner_inj_pressure=600, CR=1.5, preburner_eta=1)

        # Compare the results
        np.testing.assert_allclose([preburner_products.Ts, preburner_products.mass_Cp_frozen],
                                   [T_combustion, Cp_frozen], rtol=1e-2)

    def test_calculate_state_after_turbine(self):
        """A function to test calculation of the state after turbine"""
        # This will be done by comparing the difference in enthalpies between inlet and outlet gas.

        # Get the result of the calculations
        inlet_gas = RocketCycleFluid(species=["CH4"], mass_fractions=[1], temperature=800, type="fuel", phase="gas")
        inlet_gas.velocity = 223.293
        inlet_gas.calculate_total_temperature()
        inlet_gas.Ps = 600  # bar
        inlet_gas.calculate_total_from_static_pressure()
        beta_tt, outlet_gas, equilibrium_gas, equilibrium_output, average_molar_Cp, gamma_average = (
            elements.calculate_state_after_turbine(massflow=149, turbine_power=37e6,
                                                   turbine_polytropic_efficiency=0.85,
                                                   preburner_products=inlet_gas,
                                                   turbine_axial_velocity=223.293))
        actual_w_specific = outlet_gas.h0 - inlet_gas.h0  # kJ / mol

        # Calculate manually the desired enthalpy difference
        desired_w_specific = -(37e6 * 1e-3 / 149) * (16.04246 * 1e-3)  # kJ / mol

        # Compare the results
        np.testing.assert_allclose(actual_w_specific, desired_w_specific)

    def test_calculate_state_after_cooling_channels(self):
        """A function to test calculation of the state after cooling channels."""
        # This will be simply done by checking if the temperature and massflows are in order.
        coolant = RocketCycleFluid(species=["CH4(L)"], mass_fractions=[1], temperature=91, type="fuel", phase="liquid",
                                   species_molar_Cp=[54.037])
        coolant.Ps = 800  # bar
        coolant.Pt = 800  # bar

        # Get fluid after cooling channels
        coolant, mdot = elements.calculate_state_after_cooling_channels(
            fluid=coolant, mdot_coolant=149, mdot_film=20, pressure_drop=150,
            temperature_rise=100)

        # Compare the results
        np.testing.assert_allclose([mdot, coolant.Ps, coolant.Ts], [129, 650, 191])

    def test_calculate_combustion_chamber_performance(self):
        """A function to test the calculation of CC performance"""
        # This will be done by comparing the results from the function and CEA
        fuel = RocketCycleFluid(species=["CH4(L)"], mass_fractions=[1], temperature=111.66, type="fuel", phase="liquid",
                                species_molar_Cp=[54.037])
        oxidizer = RocketCycleFluid(species=["O2(L)"], mass_fractions=[1], temperature=90.18, type="oxidizer",
                                    phase="liquid", species_molar_Cp=[54.361])
        # Get propellant cards and add them to CEA
        rcea.add_new_fuel("fuel_card", card_str=fuel.CEA_card)
        rcea.add_new_oxidizer("oxidizer_card", card_str=oxidizer.CEA_card)
        # Make analysis and get desired values
        CC = CEA_Obj(oxName="oxidizer_card", fuelName="fuel_card", isp_units='sec', cstar_units='m/s',
                     pressure_units='bar', temperature_units='K', sonic_velocity_units='m/s',
                     enthalpy_units='kJ/kg', density_units='kg/m^3', specific_heat_units='kJ/kg-K',
                     viscosity_units='millipoise', thermal_cond_units='W/cm-degC', fac_CR=2.5)
        mdot_fuel = 100
        mdot_oxidizer = 400
        desired_IspVac, desired_C_star, desired_T_comb = CC.get_IvacCstrTc(Pc=300, MR=4, eps=100)
        CC_plenum_pressure = 300e5 / CC.get_Pinj_over_Pcomb(Pc=300, MR=4)  # Pa
        desired_A_t = (mdot_fuel + mdot_oxidizer) * desired_C_star / CC_plenum_pressure
        desired_A_e = 100 * desired_A_t
        desired_T_vac = desired_IspVac * 9.80665 * (mdot_fuel + mdot_oxidizer)
        desired_T_sea = desired_T_vac - desired_A_e * 1.01325e5
        desired_IspSea = desired_T_sea / ((mdot_fuel + mdot_oxidizer) * 9.80665)

        # Get actual values
        CC_output, CC_plenum_pressure, IspVac, IspSea, Tcomb, ThrustVac, ThrustSea, A_t, A_e = (
            elements.calculate_combustion_chamber_performance(
                mdot_oxidizer=400, mdot_fuel=100, oxidizer=oxidizer, fuel=fuel, CC_pressure_at_injector=300, CR=2.5,
                eps=100, eta_cstar=1, eta_isp=1))

        # Compare the results
        np.testing.assert_allclose([IspVac, IspSea, ThrustVac, ThrustSea], [desired_IspVac, desired_IspSea,
                                                                            desired_T_vac / 1e3, desired_T_sea / 1e3])
