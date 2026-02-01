import time

from rocketcea.cea_obj_w_units import CEA_Obj
import rocketcea.cea_obj as rcea
import scipy.optimize as opt
from rocketcycles.fluid import RocketCycleFluid, reformat_CEA_mass_fractions
import warnings
import re


def calculate_state_after_pump_for_pyfluids(fluid, delta_P, efficiency):
    """A function to calculate the state of the propellant after it passes the pump based on the pressure change and
        pump efficiency.

        :param pyfluids.Fluid fluid: PyFluids' Fluid object
        :param float or int delta_P: pressure increase across the pump (in bar)
        :param float or int efficiency: isentropic efficiency of the pump (-)
        :return: a new fluid class after compression process, specific work required
        """

    # For PyFluid, a built-in function can be used.
    old_enthalpy = fluid.enthalpy  # J / kg
    try:
        fluid = fluid.compression_to_pressure(pressure=fluid.pressure + delta_P * 1e5,
                                              isentropic_efficiency=efficiency * 100)
    except:
        warnings.simplefilter("error", UserWarning)
        warnings.warn("Slurry is created in the pump. Please increase propellant inlet temperature or decrease pressure"
                      " rise in the pump")
    w_total = fluid.enthalpy - old_enthalpy  # J / kg
    return fluid, w_total


def calculate_state_after_pump(fluid, delta_P, efficiency):
    """A function to calculate the state of the propellant after it passes the pump based on the pressure change and
    pump efficiency.

    :param RocketCycleFluid fluid: RocketCycleFluid object
    :param float or int delta_P: pressure increase across the pump (in bar)
    :param float or int efficiency: isentropic efficiency of the pump (-)
    :return: a new fluid class after compression process, specific work required
    """
    # The process is divided into two paths. The first is the isothermal compression due to pressure change. The
    # second one is isobaric heating (due to pump inefficiency). This can be done as liquid state only depends on
    # the final state parameters (pressure and temperature), which stay the same regardless of path taken.

    # First isothermal compression. Calculate new density at the outlet (due to compression) and the specific work.
    inlet_pressure = fluid.Pt * 1e5  # change bar to Pa; total pressure used to take into account fluid velocity
    delta_P *= 1e5  # change bar to Pa
    outlet_pressure = inlet_pressure + delta_P  # Pa
    outlet_density_isothermal = (fluid.density /
                                 (1 - ((outlet_pressure - inlet_pressure) / fluid.liquid_elasticity)))  # kg/m^3
    w_useful = (outlet_pressure / outlet_density_isothermal) - (inlet_pressure / fluid.density)  # J / kg
    w_total = w_useful / efficiency  # J / kg

    # Now calculate heating of the liquid and new density (due to new pressure). For such small temperature
    # rise like in a pumped fluid, Cp can be assumed to be constant
    w_wasted = w_total - w_useful  # J / kg
    delta_T = w_wasted / fluid.mass_Cp_frozen  # K
    outlet_temperature = fluid.Ts + delta_T  # K
    outlet_density = (outlet_density_isothermal /
                      (1 + ((outlet_temperature - fluid.Ts) * fluid.volumetric_expansion_coefficient)))  # kg / m^3

    # Assign new properties to the fluid, return the results
    fluid = RocketCycleFluid(species=fluid.species, mass_fractions=fluid.mass_fractions,
                             temperature=outlet_temperature, type=fluid.type, phase="liquid",
                             volumetric_expansion_coefficient=fluid.volumetric_expansion_coefficient,
                             liquid_elasticity=fluid.liquid_elasticity, density=outlet_density,
                             species_molar_Cp=fluid.species_molar_Cp)
    fluid.Pt = outlet_pressure / 1e5  # bar
    fluid.Ps = fluid.Pt  # bar

    return fluid, w_total


def calculate_state_after_preburner(preburner_inj_pressure, CR, fuel=None, oxidizer=None, monopropellant=None, OF=None):
    """A function to calculate the state of the combustion product mixture in the preburner based on inflow
    propellants.

    :param RocketCycleFluid fuel: RocketCycleFluid object representing fuel
    :param RocketCycleFluid oxidizer: RocketCycleFluid object representing oxidizer
    :param RocketCycleFluid monopropellant: RocketCycleFluid object representing monopropellant (if used instead of
     oxidizer and fuel)
    :param float or int OF: Oxidizer-to-Fuel ratio. Only needed if fuel and oxidizer are used.
    :param float or int preburner_inj_pressure: Preburner pressure at the injector face (bar)
    :param float or int CR: "Contraction ratio" of the preburner, which is used as a measure of its cross-sectional
     area in order to model Rayleigh line loss.
    :return: Preburner CEA full output, RocketEngineFluid representing its products
    """

    # First get the CEA propellant cards. Joule - Thomson effect (temperature
    # increases) in injector is neglected, as even for extreme cases (pressure drops of ~100 bar) it on the order of few
    # degrees for both liquids or hot gases (the only fluids coming into combustion chambers in closed cycles),
    # so irrelevant.
    if fuel is not None and oxidizer is not None:
        rcea.add_new_fuel("fuel card", fuel.CEA_card)
        rcea.add_new_oxidizer("oxidizer card", oxidizer.CEA_card)

    elif monopropellant is not None:
        # The monopropellant CEA card name needs to have name instead of oxid or fuel.
        monoprop_card = monopropellant.CEA_card.replace("oxid", "name")
        monoprop_card = monoprop_card.replace("fuel", "name")
        rcea.add_new_propellant("monoprop card", monoprop_card)

    # Raise an error if fuel, oxidizer or monopropellant not assigned correctly
    elif fuel is not None and oxidizer is not None and monopropellant is not None:
        warnings.simplefilter("error", UserWarning)
        warnings.warn("Both oxidizer - fuel combination and monopropellant were assigned.")

    else:
        warnings.simplefilter("error", UserWarning)
        warnings.warn("Wrong input for fuel and oxidizer, or monopropellant")

    # Raise an error if fuel and oxidizer are used, but OF is not assigned
    if fuel is not None and oxidizer is not None and OF is None:
        warnings.simplefilter("error", UserWarning)
        warnings.warn("OF was not assigned for given fuel - oxidizer combination")

    # OF ratio of the preburner is used to determine if products gases are fuel or oxidizer rich. However,
    # for monopropellant, OF is None. Therefore, to keep the same code for both fuel-oxidizer and monopropellant,
    # OF needs to be changed to above 1 or below 1 depending on monopropellant. It is just an artificial value then
    # that allows to determine if the condition is True or False, without any physical meaning. It is still passed to
    # all CEA functions (again to use the same code everywhere), but it is not used inside of them.
    if monopropellant is not None:
        if monopropellant.type == "fuel":
            OF = 0.5
        elif monopropellant.type == "oxid":
            OF = 2

    # Get the full CEA output, the combustion products' composition, plenum pressure, temperature and specific heat
    # in the combustor. To get full CEA output as a string, CEA object that is not a SI units wrapper needs to be
    # created (as the wrapper does not have such function).  It is also used to get products velocity at the end of
    # preburner.
    rcea.clearCache(show_size=False)
    if fuel is not None and oxidizer is not None:
        preburner = rcea.CEA_Obj(oxName="oxidizer card", fuelName="fuel card", fac_CR=CR)
    elif monopropellant is not None:
        preburner = rcea.CEA_Obj(propName="monoprop card", fac_CR=CR)
    preburner_CEA_output = preburner.get_full_cea_output(MR=OF, Pc=preburner_inj_pressure, pc_units="bar", output="si",
                                                         short_output=1, show_mass_frac=1)
    # To get sonic velocity and Mach number at the end of preburner, pressure in psia needs to be used, as SI Units
    # wrapper does not have these functions.
    a = preburner.get_SonicVelocities(Pc=preburner_inj_pressure * 14.5037738, MR=OF)[0] * 0.3048  # m/s
    M = preburner.get_Chamber_MachNumber(Pc=preburner_inj_pressure * 14.5037738, MR=OF)
    products_velocity = M * a

    # Afterward SI units CEA object is created, so that variables with regular units are returned
    if fuel is not None and oxidizer is not None:
        preburner = CEA_Obj(oxName="oxidizer card", fuelName="fuel card", isp_units='sec', cstar_units='m/s',
                            pressure_units='bar', temperature_units='K', sonic_velocity_units='m/s',
                            enthalpy_units='kJ/kg', density_units='kg/m^3', specific_heat_units='kJ/kg-K',
                            viscosity_units='millipoise', thermal_cond_units='W/cm-degC', fac_CR=CR)
    elif monopropellant is not None:
        preburner = CEA_Obj(propName="monoprop card", isp_units='sec', cstar_units='m/s',
                            pressure_units='bar', temperature_units='K', sonic_velocity_units='m/s',
                            enthalpy_units='kJ/kg', density_units='kg/m^3', specific_heat_units='kJ/kg-K',
                            viscosity_units='millipoise', thermal_cond_units='W/cm-degC', fac_CR=CR)

    # Get temperature in the preburner and gamma of the products. It is assumed that preburner efficiency only affects
    # temperature but not gas composition, as the difference between the two temperatures is very small, so composition
    # and other properties should be similar.
    preburner_temperature = preburner.get_Tcomb(Pc=preburner_inj_pressure, MR=OF)  # K

    # Get preburner pressure at its end
    preburner_plenum_pressure = (preburner_inj_pressure /
                                 preburner.get_Pinj_over_Pcomb(Pc=preburner_inj_pressure, MR=OF))  # bar

    # Get preburner heat capacity and visocity. Heat capacity is equilibrium, not frozen heat capacity.
    transport_properties = preburner.get_Chamber_Transport(Pc=preburner_inj_pressure, MR=OF)[0:2]
    products_Cp_equilibrium = transport_properties[0] * 1e3  # J / (K * kg)
    viscosity = transport_properties[1]  # milipoise

    # Get preburner mass fractions
    products_mass_fractions = preburner.get_SpeciesMassFractions(Pc=preburner_inj_pressure, MR=OF, min_fraction=1e-6)[1]
    products_mass_fractions = reformat_CEA_mass_fractions(products_mass_fractions)

    # Create a RocketCycleFluid object to store the results. Determine if they are fuel or oxygen rich.
    if OF >= 1:
        products_type = "oxidizer"
    else:
        products_type = "fuel"
    preburner_products = RocketCycleFluid(species=list(products_mass_fractions.keys()),
                                          mass_fractions=list(products_mass_fractions.values()),
                                          temperature=preburner_temperature, type=products_type, phase="gas")
    preburner_products.Ps = preburner_plenum_pressure  # bar
    preburner_products.mass_Cp_equilibrium = products_Cp_equilibrium  # J / (kg * K)
    preburner_products.viscosity = viscosity  # milipoise

    # Calculate total temperature and pressure
    preburner_products.velocity = products_velocity  # m/s
    preburner_products.calculate_total_temperature()
    preburner_products.calculate_total_from_static_pressure()

    # Return the results
    return preburner_CEA_output, preburner_products


def calculate_state_after_turbine(massflow, turbine_power, turbine_polytropic_efficiency, preburner_products,
                                  turbine_axial_velocity, pressure_recovery_factor):
    """A function to calculate the pressure ratio of the turbine and the state of the propellant after it passes
    the turbine.

    :param int or float massflow: A massflow through the turbine (kg/s)
    :param int or float turbine_power:  A shaft power turbine needs to deliver (W)
    :param int or float turbine_polytropic_efficiency:  Polytropic efficiency of the turbine (from 0 to 1)
    :param RocketCycleFluid preburner_products:  RocketCycleFluid object representing preburner products at its end
    :param int or float turbine_axial_velocity: Axial velocity (m/s) through the turbine
    :param float pressure_recovery_factor: Static to total pressure ratio that the diffuser and manifold after turbine
     allow to recover.

    :return: Turbine pressure ratio, RocketCycleFluid representing outlet gas,
    RocketCycleFluid representing equilibrated outlet gas, CEA full output for equilibrated outlet gas
    """

    # It is assumed that gas is in frozen conditions in the turbine. Firstly, this is because it is relatively short
    # and velocities are high, meaning that residence time is small (meaning that equilibrium will not be reached).
    # Secondly, the pressure ratio in the turbine is very small compared to typical rocket engines nozzles
    # (both turbine and nozzle expand gases), so it can be thought of as a nozzle with very small area ratio - and for
    # these frozen conditions are used (again because of short residence time). Thirdly, temperatures
    # (and their changes) are relatively small, so change in Gibbs energy will not be large, hence equilibrium will be
    # almost constant anyway.  Nevertheless, at some point after turbine, gases will eventually reach equilibrium which
    # will be accounted for.

    # Furthermore, it is assumed that turbine axial velocity is constant along the turbine, which would be typical
    # for stages with constant flow coefficient. It is also assumed that outlet velocity is only axial (which is
    # almost always the case in LRE turbines). Therefore, the velocity component drops out from total enthalpy
    # conservation equation. Now, based on initial static enthalpy and specific power of turbine, static enthalpy (
    # and thus static temperature) can be found at turbine outlet. Enthalpy is used, because there is no energy error
    # (no assumption about thermally or calorically perfect gas / Cp averaging), which is deemed most important for
    # accurate prediction of system performance and for numerical convergence. Afterward, total temperature can be
    # calculated at stage outlet. Assuming polytropic expansion (as polytropic efficiency does not depend on
    # presssure ratio like isentropic one), required total-to-total pressure ratio can be obtained for each stage,
    # together with static and total pressures at the outlet. Furthermore, it the heat ratio for any
    # expansion process in the turbine is based on average specific heat for it (as it depends on temperature).

    # We have total and static properties of gas at the end of preburner. It will accelerate before the nozzle, hence
    # the new static temperature and pressure need to be found using turbine axial velocity.
    # Cp is calculated using static temperature, which is unknown. Therefore, it cannot be calculated from total
    # temperature. Instead, static temperature needs to be iteratively found, such that total temperature is the same.

    def calculate_temperature_residual(inlet_Ts):
        inlet_gas = RocketCycleFluid(species=preburner_products.species,
                                     mass_fractions=preburner_products.mass_fractions,
                                     temperature=inlet_Ts, type=preburner_products.type, phase="gas")
        inlet_gas.velocity = turbine_axial_velocity
        inlet_gas.calculate_total_temperature()
        return inlet_gas.Tt - preburner_products.Tt

    Ts = opt.toms748(calculate_temperature_residual, a=288.15, b=preburner_products.Ts, maxiter=1000)
    inlet_gas = RocketCycleFluid(species=preburner_products.species, mass_fractions=preburner_products.mass_fractions,
                                 temperature=float(Ts), type=preburner_products.type, phase="gas")
    inlet_gas.Pt = preburner_products.Pt
    inlet_gas.velocity = turbine_axial_velocity
    inlet_gas.calculate_total_temperature()
    inlet_gas.calculate_static_from_total_pressure()

    # Calculate specific work of the turbine
    specific_work = ((turbine_power * 1e-3) / massflow) * (inlet_gas.MW * 1e-3)  # kJ / mol
    outlet_hs = inlet_gas.h0 - specific_work  # kJ / mol

    # Define a function which will be solved to find static temperature at the outlet that results in
    # the right work extraction
    def calc_enthalpy_residual(outlet_Ts):
        """Calculate enthalpy residual based on outlet static temperature.

        :param float or int outlet_Ts: static temperature at the outlet.

        :return: enthalpy residual between desired and obtained enthalpy
        """
        outlet_gas = RocketCycleFluid(species=inlet_gas.species, mass_fractions=inlet_gas.mass_fractions,
                                      temperature=outlet_Ts, type=inlet_gas.type, phase=inlet_gas.phase)
        return outlet_hs - outlet_gas.h0  # kJ / mol

    # Solve the function above. Bisection algorithm will be again used for guaranteed convergence. The lower limit is
    # 0 deg C, the higher limit is inlet gas static temperature.
    Ts = opt.toms748(calc_enthalpy_residual, a=288.15, b=inlet_gas.Ts, maxiter=1000)  # K

    # Define gas at the outlet of the current stage
    outlet_gas = RocketCycleFluid(species=inlet_gas.species, mass_fractions=inlet_gas.mass_fractions,
                                  temperature=float(Ts), type=inlet_gas.type, phase=inlet_gas.phase)

    # Calculate total temperature
    outlet_gas.velocity = turbine_axial_velocity  # m / s
    outlet_gas.calculate_total_temperature()

    # Calculate gamma and calculate total-to-total pressure ratio. Gamma is calculated based on average specific heat
    # for the process
    average_molar_Cp = (((inlet_gas.h0 - outlet_gas.h0) * 1e3) /
                        (inlet_gas.Ts - outlet_gas.Ts))  # J / (K * mol)
    gamma_average = average_molar_Cp / (average_molar_Cp - inlet_gas.R)

    # Calculate pressure ratio
    beta_tt = (inlet_gas.Tt / outlet_gas.Tt) ** (gamma_average /
                                                 (turbine_polytropic_efficiency * (gamma_average - 1)))

    # Calculate pressures at the outlet using beta_tt and pressure recovery factor.
    outlet_gas.Pt = inlet_gas.Pt / beta_tt  # bar
    outlet_gas.Ps = outlet_gas.Pt * pressure_recovery_factor

    # After the turbine outlet, apply pressure recovery factor and perform equilibrium

    equilibrium_gas, equilibrium_CEA_output = outlet_gas.equilibrate()

    # Return turbine calculations results
    return beta_tt, outlet_gas, equilibrium_gas, equilibrium_CEA_output, average_molar_Cp, gamma_average


def calculate_state_after_cooling_channels_for_Pyfluids(fluid, mdot_coolant, mdot_film, pressure_drop,
                                                        temperature_rise):
    """A function to calculate the state of the coolant propellant after it passes through the cooling channels.

    :param pyfluids.Fluid fluid: An object representing cooling fluid
    :param float or int mdot_coolant: Inlet coolant massflow (in kg/s)
    :param float or int mdot_film: Film cooling massflow (in kg/s)
    :param float or int pressure_drop: Pressure drop (in bar) across cooling channels
    :param float or int temperature_rise: Temperature rise (in K) across cooling channels

    :return: Object representing outlet fluid, outlet massflow (in kg/s), heat flow picked up by the coolant (W)
    """

    # First calculate outlet massflow
    mdot_outlet = mdot_coolant - mdot_film  # kg / s

    # Get enthalpy at the inlet
    enthalpy_inlet = fluid.enthalpy # J / kg

    # For PyFluids' Fluid, object method can be used
    fluid = fluid.heating_to_temperature(temperature=fluid.temperature + temperature_rise,
                                         pressure_drop=pressure_drop * 1e5)

    # Get enthalpy at the outlet
    enthalpy_outlet = fluid.enthalpy # J / kg

    # Total heat flow picked up by the coolant
    heat_coolant = (enthalpy_outlet - enthalpy_inlet) * mdot_outlet # W

    return fluid, mdot_outlet, heat_coolant


def calculate_state_after_cooling_channels(fluid, mdot_coolant, mdot_film, pressure_drop,
                                           temperature_rise):
    """A function to calculate the state of the coolant propellant after it passes through the cooling channels.

    :param RocketCycleFluid fluid: An object representing cooling fluid
    :param float or int mdot_coolant: Inlet coolant massflow (in kg/s)
    :param float or int mdot_film: Film cooling massflow (in kg/s)
    :param float or int pressure_drop: Pressure drop (in bar) across cooling channels
    :param float or int temperature_rise: Temperature rise (in K) across cooling channels

    :return: Object representing outlet fluid, outlet massflow (in kg/s), heat flow picked up by the coolant (W)
    """

    # First calculate outlet massflow
    mdot_outlet = mdot_coolant - mdot_film  # kg / s

    # Get enthalpy at the inlet
    enthalpy_inlet = (fluid.h0 / (fluid.MW * 1e-3)) * 1e3  # J / kg

    # Create a new object with new temperature and pressure.
    new_Ps = fluid.Ps - pressure_drop
    fluid = RocketCycleFluid(species=fluid.species, mass_fractions=fluid.mass_fractions,
                             temperature=fluid.Ts + temperature_rise, type=fluid.type, phase="liquid",
                             volumetric_expansion_coefficient=fluid.volumetric_expansion_coefficient,
                             liquid_elasticity=fluid.liquid_elasticity)
    fluid.Ps = new_Ps  # bar
    # Dynamic head is small, so total pressure is the same as the static one
    fluid.Pt = new_Ps  # bar

    # Get enthalpy at the outlet
    enthalpy_outlet = (fluid.h0 / (fluid.MW * 1e-3)) * 1e3 # J / kg

    # Total heat flow picked up by the coolant
    heat_coolant = (enthalpy_outlet - enthalpy_inlet) * mdot_outlet # W

    return fluid, mdot_outlet, heat_coolant


def calculate_combustion_chamber_performance(mdot_oxidizer, mdot_fuel, oxidizer, fuel, CC_pressure_at_injector, CR,
                                             eps, eta_cstar, eta_cf, mdot_film=0, coolant_CEA_card=None,
                                             include_film_in_cstar=False, heat_coolant=0):
    """A function to calculate the combustion chamber performance.

    :param float or int mdot_oxidizer: Oxidizer massflow (kg/s)
    :param float or int mdot_fuel: Fuel massflow (kg/s)
    :param RocketCycleFluid oxidizer: RocketCycleFluid representing oxidizer
    :param RocketCycleFluid fuel: RocketCycleFluid representing fuel
    :param float or int CC_pressure_at_injector: CC pressure (in bar) at the injector plate
    :param float or int CR: CC contraction ratio
    :param float or int eps: CC expansion ratio
    :param float or int eta_cstar: C* efficiency (-)
    :param float or int eta_cf: Cf efficiency in vacuum (-)
    :param float or int mdot_film: Film coolant massflow (kg/s)
    :param str coolant_CEA_card: Film coolant CEA card
    :param boolean include_film_in_cstar: Boolean whether film is included in CEA calculations to get C* and Isp.
     If True, eta_cstar should refer to real C* over CEA C* of core flow mixed with film.
      eta_cf should refer to real Cf over CEA Cf of core flow mixed with film.
     If False, eta_cstar should refer to real C* over CEA C* of core flow alone.
      eta_cf should refer to real Cf over CEA Cf of core flow alone.
    :param float or int heat_coolant: Heat flow picked up by the coolant from combustion gases. Enthalpy of the reactants
     will be lowered by adequate enthalpy value calculated from heat_coolant to keep energy balance.

    :return: Full CEA output, CC plenum pressure, real vacuum and sea level specific impulse,
     ideal combustion temperature, real vacuum and sea level thrust, throat and exit areas
    """

    # Get total massflow and OF of the core flow (without film coolant)
    mdot_total = mdot_oxidizer + mdot_fuel + mdot_film  # kg / s
    mdot_total_wo_film = mdot_oxidizer + mdot_fuel  # kg / s

    # Calculate OF without film
    OF_wo_film = mdot_oxidizer / mdot_fuel

    # First calculate how much heat must be taken away from each reactant
    heat_lost_ox = heat_coolant * OF_wo_film / (1 + OF_wo_film) # W
    heat_lost_fuel = heat_coolant - heat_lost_ox # W

    # Now calculate how much enthalpy must be removed. Get temperature change from it.
    enthalpy_lost_ox = heat_lost_ox / mdot_oxidizer # J / kg
    enthalpy_lost_fuel = heat_lost_fuel / mdot_fuel # J / kg
    temperature_decrease_ox = enthalpy_lost_ox / oxidizer.mass_Cp_frozen # K
    temperature_decrease_fuel = enthalpy_lost_fuel / fuel.mass_Cp_frozen # K

    # Then modify propellant CEA cards
    fuel = RocketCycleFluid(species=fuel.species, mass_fractions=fuel.mass_fractions,
                            temperature=fuel.Ts - temperature_decrease_fuel, type=fuel.type, phase=fuel.phase,
                            species_molar_Cp=fuel.species_molar_Cp)
    oxidizer = RocketCycleFluid(species=oxidizer.species, mass_fractions=oxidizer.mass_fractions,
                                temperature=oxidizer.Ts - temperature_decrease_ox, type=oxidizer.type,
                                phase=oxidizer.phase, species_molar_Cp=oxidizer.species_molar_Cp)

    # Get and save CEA analysis results for core flow only. Create CEA object with Imperial units to be able to get
    # full output.
    rcea.add_new_fuel(name="fuel card w/o coolant", card_str=fuel.CEA_card)
    rcea.add_new_oxidizer(name="oxidizer card w/o coolant", card_str=oxidizer.CEA_card)
    CC_core = rcea.CEA_Obj(oxName="oxidizer card w/o coolant", fuelName="fuel card w/o coolant", fac_CR=CR)
    CC_core_CEA_output = CC_core.get_full_cea_output(Pc=CC_pressure_at_injector, MR=OF_wo_film, eps=eps,
                                                     pc_units="bar", output="si", short_output=1, show_mass_frac=1)
    # Also get temperature of combustion. Input is given in Psia and output is in Rankine degrees, hence conversions
    Tcomb = CC_core.get_Tcomb(Pc=CC_pressure_at_injector * 14.5038, MR=OF_wo_film) * 5 / 9

    # It may be beneficial to have CEA results for the case where core flow and film are lumped together (for example
    # to assess impact of the film on the performance). Therefore, a mixed CEA case with core flow and film lumped
    # together is also calculated.
    # If there is no film coolant, there will be no CEA results for the mixed case.
    if mdot_film == 0 and coolant_CEA_card is None:
        CC_with_film_CEA_output = None

    # If there is film coolant, do analysis for the mixed case.
    elif mdot_film != 0 and coolant_CEA_card is not None:
        # Recognize whether fuel or oxygen is used for coolant and depending on the results, calculate total OF and add
        # cards to CEA.
        if "fuel" in coolant_CEA_card:
            # Calculate OF
            OF_w_film = mdot_oxidizer / (mdot_fuel + mdot_film)
            # Adjust weight fractions in CEA cards
            mdot_total_fuel = mdot_fuel + mdot_film
            fuel_CEA_card = modify_wt_percent(CEA_card=fuel.CEA_card,
                                              adjust_func=lambda wt: wt * mdot_fuel / mdot_total_fuel)
            coolant_CEA_card = modify_wt_percent(CEA_card=coolant_CEA_card,
                                                 adjust_func=lambda wt: wt * mdot_film / mdot_total_fuel)
            # Merge film and fuel cards
            fuel_CEA_card = fuel_CEA_card + coolant_CEA_card
            # Add cards to CEA
            rcea.add_new_fuel(name="fuel card", card_str=fuel_CEA_card)
            rcea.add_new_oxidizer(name="oxidizer card", card_str=oxidizer.CEA_card)
        elif "oxid" in coolant_CEA_card:
            # Calculate OF
            OF_w_film = (mdot_oxidizer + mdot_film) / mdot_fuel
            # Adjust weight fractions in CEA cards
            mdot_total_ox = mdot_oxidizer + mdot_film
            ox_CEA_card = modify_wt_percent(CEA_card=oxidizer.CEA_card,
                                            adjust_func=lambda wt: wt * mdot_oxidizer / mdot_total_ox)
            coolant_CEA_card = modify_wt_percent(CEA_card=coolant_CEA_card,
                                                 adjust_func=lambda wt: wt * mdot_film / mdot_total_ox)
            # Merge film and fuel cards
            ox_CEA_card = ox_CEA_card + coolant_CEA_card
            # Add cards to CEA
            rcea.add_new_fuel(name="fuel card", card_str=fuel.CEA_card)
            rcea.add_new_oxidizer(name="oxidizer card", card_str=ox_CEA_card)

        # Create CEA object with Imperial units to be able to get and save full output.
        CC_with_film = rcea.CEA_Obj(oxName="oxidizer card", fuelName="fuel card", fac_CR=CR)
        CC_with_film_CEA_output = CC_with_film.get_full_cea_output(Pc=CC_pressure_at_injector, MR=OF_w_film, eps=eps,
                                                                   pc_units="bar", output="si", short_output=1,
                                                                   show_mass_frac=1)

    # If there is no film or if there is film, but it is not included in performance calculations,
    # core flow only will be used for them.
    if mdot_film == 0 or not include_film_in_cstar:
        # Create CEA object with SI units for further calculations and assign OF
        CC = CEA_Obj(oxName="oxidizer card w/o coolant", fuelName="fuel card w/o coolant", isp_units='sec',
                      cstar_units='m/s', pressure_units='bar', temperature_units='K', sonic_velocity_units='m/s',
                      enthalpy_units='kJ/kg', density_units='kg/m^3', specific_heat_units='kJ/kg-K',
                      viscosity_units='millipoise', thermal_cond_units='W/cm-degC', fac_CR=CR)
        OF = OF_wo_film
    # If there is film and if it is included in performance calculations, then core flow mixed with film will be used
    # for them
    elif mdot_film != 0 and include_film_in_cstar:
        # Create CEA object with SI units for further calculations and assign OF
        CC = CEA_Obj(oxName="oxidizer card", fuelName="fuel card", isp_units='sec', cstar_units='m/s',
                     pressure_units='bar', temperature_units='K', sonic_velocity_units='m/s',
                     enthalpy_units='kJ/kg', density_units='kg/m^3', specific_heat_units='kJ/kg-K',
                     viscosity_units='millipoise', thermal_cond_units='W/cm-degC', fac_CR=CR)
        OF = OF_w_film

    # Get ideal specific impulse, C* and CC plenum pressure
    (IspVac_ideal, Cstar) = CC.get_IvacCstrTc(Pc=CC_pressure_at_injector, MR=OF, eps=eps)[0:2]
    CC_plenum_pressure = CC_pressure_at_injector / CC.get_Pinj_over_Pcomb(Pc=CC_pressure_at_injector, MR=OF)  # bar

    # If film is not included in CEA calculations, IspVac must be decreased proportionally to its amount
    if not include_film_in_cstar:
        IspVac_ideal = IspVac_ideal * mdot_total_wo_film / mdot_total

    # Get throat and exit area
    A_t = Cstar * mdot_total * eta_cstar / (CC_plenum_pressure * 1e5)  # m^2
    A_e = A_t * eps  # m^2

    # Calculate thrust and real Isp in vacuum
    IspVac_real = IspVac_ideal * eta_cf * eta_cstar
    ThrustVac = IspVac_real * mdot_total * 9.80665

    # Calculate how much Isp is lost. Approximately the same amount will be lost at the sea level
    # (neglecting different nozzle phenomena like separation) as well.
    Isp_losses = IspVac_ideal - IspVac_real

    # Then calculate Isp at the sea level including separation
    IspSea_ideal, sea_level_operation_mode = CC.estimate_Ambient_Isp(Pc=CC_pressure_at_injector, MR=OF, eps=eps,
                                                                    Pamb=1.01325)
    # If film is not included in CEA calculations, IspSea must be decreased proportionally to its amount
    if not include_film_in_cstar:
        IspSea_ideal = IspSea_ideal * mdot_total_wo_film / mdot_total
    # Subtract the loses
    IspSea_real = IspSea_ideal - Isp_losses
    # Calculate sea level thrust
    ThrustSea = mdot_total * 9.80665 * IspSea_real

    return (CC_core_CEA_output, CC_with_film_CEA_output, CC_plenum_pressure, IspVac_real, IspSea_real, Tcomb,
            ThrustVac / 1e3, ThrustSea / 1e3, A_t, A_e, sea_level_operation_mode)


def modify_wt_percent(CEA_card, adjust_func):
    """A function to change weight fractions in CEA card based on given transformation function.

    :param string CEA_card: CEA card in which weight fractions need to be modified.
    :param callable function adjust_func: A transformation function to adjust weight fractions.

    :return: The modified CEA card string with updated weight fractions.
    """

    def replace_wt(match):
        prefix = match.group(1)  # 'wt%='
        current_value = float(match.group(2))  # Extract the number as float
        # Adjust the value using the provided function
        new_value = adjust_func(current_value)
        # Format the new value and return it
        return f"{prefix}{new_value}"

    # Regular expression to find wt% values
    pattern = r'(wt%=\s*)(\d+(\.\d+)?)'
    # Replace all occurrences using the replace_wt function
    return re.sub(pattern, replace_wt, CEA_card)
