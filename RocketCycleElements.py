from rocketcea.cea_obj_w_units import CEA_Obj
import rocketcea.cea_obj as rcea
import scipy.optimize as opt
from RocketCycleFluid import RocketCycleFluid, reformat_CEA_mass_fractions


def calculate_state_after_pump(fluid, fluid_object, delta_P, efficiency):
    """A function to calculate the state of the propellant after it passes the pump based on the pressure change and
    pump efficiency.

    :param pyfluids.Fluid or RocketCycleFluid fluid: PyFluids' Fluid or RocketCycleFluid object
    :param str fluid_object: a string "PyFluid" or "RocketEngineFluid" indicating fluid object used
    :param float or int delta_P: pressure increase across the pump (in bar)
    :param float or int efficiency: isentropic efficiency of the pump (-)
    :return: a new fluid class after compression process, specific work required
    """

    # For PyFluid, a built-in function can be used.
    if fluid_object == "PyFluid":
        old_enthalpy = fluid.enthalpy  # J / kg
        fluid.compression_to_pressure(pressure=fluid.pressure + delta_P * 1e5, isentropic_efficiency=efficiency)
        w_total = fluid.enthalpy - old_enthalpy  # J / kg
        return fluid, w_total

    elif fluid_object == "RocketCycleFluid":
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
                                 liquid_elasticity=fluid.liquid_elasticity, density=outlet_density)
        fluid.Pt = outlet_pressure / 1e5  # bar
        fluid.Ps = fluid.Pt  # bar

        return fluid, w_total


def calculate_state_after_preburner(fuel, oxidizer, OF, preburner_inj_pressure, products_velocity):
    """A function to calculate the state of the combustion product mixture in the preburner based on inflow
    propellants.

    :param RocketCycleFluid fuel: RocketCycleFluid object representing fuel
    :param RocketCycleFluid oxidizer: RocketCycleFluid object representing oxidizer
    :param float or int OF: Oxidizer-to-Fuel ratio
    :param float or int preburner_inj_pressure: Preburner pressure at the injector face
    :param float or int products_velocity: Velocity of the combustion products when they enter turbine
    :return: Preburner CEA full output, RocketEngineFluid representing its products
    """

    # First get the CEA propellant cards. Joule - Thomson effect (temperature
    # increases) in injector is neglected, as even for extreme cases (pressure drops of ~100 bar) it on the order of few
    # degrees for both liquids or hot gases (the only fluids coming into combustion chamber in closed cycles),
    # so irrelevant.
    rcea.add_new_fuel("fuel card", fuel.CEA_card)
    rcea.add_new_oxidizer("oxidizer card", oxidizer.CEA_card)

    # Get CEA output, but first find the right CR.
    # Explanation: Preburner products need to have the right velocity when they come into turbine inlet (as given
    # by its flow coefficient and rotational speed). There will be also some correlated Rayleigh line loss due to
    # non-adiabatic flow in the preburner. To model this, we are going to vary contraction ratio of the CEA_Obj
    # class, even though preburner does not have a nozzle or any contraction. However, this will vary preburner
    # cross-sectional area (as there is no difference preburner and cylindrical section of combustion chamber) and
    # allow to achieve desired velocity at its end and consequently get the pressure loss.

    # First convert preburner pressure to Imperial units, as SI wrapper around rocketCEA does not have a function to get
    # Mach number at the end of chamber
    pressure_preburner_inj_psia = preburner_inj_pressure * 14.5037738  # convert pressure in bar to psia

    # Then create a function to solve numerically that gives the right CR
    def calc_velocity_residual(CR):
        """A function to calculate a residual between obtained and desired velocity at the end of preburner

        :param float or int CR: Contraction ratio of the combustor. In this case it is not an actual contraction ratio,
            but a measure of its cross-sectional area
        :return: A residual between obtained and desired velocity at the end of preburner (in m/s)
        """
        preburner = rcea.CEA_Obj(oxName="oxidizer card", fuelName="fuel card", fac_CR=CR)
        a = preburner.get_SonicVelocities(Pc=pressure_preburner_inj_psia, MR=OF)[0] * 0.3048  # m/s
        M = preburner.get_Chamber_MachNumber(Pc=pressure_preburner_inj_psia, MR=OF)
        return M * a - products_velocity  # m/s

    # Solve the function for CR. Use bracketing method from FR 1 to 5. 1 is the lower physical limit for CR.
    # 10 should correspond to a very slow turbine inlet velocity, so should be able to always converge.
    CR = opt.toms748(calc_velocity_residual, a=1, b=10, maxiter=1000)

    # Get the full CEA output, the combustion products' composition, plenum pressure, temperature and specific heat
    # in the combustor. To get full CEA output as a string, CEA object that is not a SI units wrapper needs to be
    # created (as the wrapper does not have such function).
    preburner = rcea.CEA_Obj(oxName="oxidizer card", fuelName="fuel card", fac_CR=CR)
    preburner_CEA_output = preburner.get_full_cea_output(Pc=preburner_inj_pressure, MR=OF, pc_units="bar", output="si",
                                                         short_output=1)

    # Afterward SI units CEA object is created, so that variables with regular units are returned
    preburner = CEA_Obj(oxName="oxidizer card", fuelName="fuel card", isp_units='sec', cstar_units='m/s',
                        pressure_units='bar', temperature_units='K', sonic_velocity_units='m/s', enthalpy_units='kJ/kg',
                        density_units='kg/m^3', specific_heat_units='kJ/kg-K', viscosity_units='millipoise',
                        thermal_cond_units='W/cm-degC', fac_CR=CR)

    # Get temperature in the preburner and gamma of the products
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


def calculate_state_after_turbine(massflow, turbine_power, turbine_polytropic_efficiency, inlet_gas,
                                  turbine_axial_velocity):
    """A function to calculate the pressure ratio of the turbine and the state of the propellant after it passes
    the turbine.

    :param int or float massflow: A massflow through the turbine (kg/s)
    :param int or float turbine_power:  A shaft power turbine needs to deliver (W)
    :param int or float turbine_polytropic_efficiency:  Polytropic efficiency of the turbine (from 0 to 1)
    :param RocketCycleFluid inlet_gas:  RocketCycleFluid object representing fluid driving the turbine
    :param int or float turbine_axial_velocity: Axial velocity through the turbine

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
    # room temperature, the higher limit is inlet gas static temperature.
    Ts = opt.toms748(calc_enthalpy_residual, a=288.15, b=inlet_gas.Ts, maxiter=1000)  # K

    # Define gas at the outlet of the current stage
    outlet_gas = RocketCycleFluid(species=inlet_gas.species, mass_fractions=inlet_gas.mass_fractions,
                                  temperature=Ts, type=inlet_gas.type, phase=inlet_gas.phase)

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

    # Calculate pressures at the outlet. This time use gamma at outlet, as static to
    # total properties are isentropic process at that location (and not across the turbine).
    outlet_gas.Pt = inlet_gas.Pt / beta_tt  # bar
    outlet_gas.calculate_static_from_total_pressure()

    # After the turbine outlet, perform equilibrium
    equilibrium_gas, equilibrium_output = outlet_gas.equilibrate()

    # Return turbine calculations results
    return beta_tt, outlet_gas, equilibrium_gas, equilibrium_output


def calculate_state_after_cooling_channels(fluid, fluid_object, mdot_coolant, mdot_film, pressure_drop,
                                           temperature_rise):
    """A function to calculate the state of the coolant propellant after it passes through the cooling channels.

    :param pyfluids.Fluid or RocketCycleFluid fluid: An object representing cooling fluid
    :param str fluid_object: A string indicating whether PyFluids'Fluid or RocketCycleFluid is used
    :param float or int mdot_coolant: Inlet coolant massflow (in kg/s)
    :param float or int mdot_film: Film cooling massflow (in kg/s)
    :param float or int pressure_drop: Pressure drop (in bar) across cooling channels
    :param float or int temperature_rise: Temperature rise (in K) across cooling channels

    :return: Object representing outlet fluid, outlet massflow (in kg/s)
    """

    # First calculate outlet massflow
    mdot_outlet = mdot_coolant - mdot_film  # kg / s

    # If PyFluids' Fluid is used, object method can be used
    if fluid_object == "PyFluid":
        fluid.heating_to_temperature(temperature=fluid.temperature + temperature_rise,
                                     pressure_drop=pressure_drop * 1e5)
        return fluid, mdot_outlet

    # If RocketCycleFluid is used, create a new object with new temperature and pressure.
    elif fluid_object == "RocketCycleFluid":
        new_Ps = fluid.Ps - pressure_drop
        fluid = RocketCycleFluid(species=fluid.species, mass_fractions=fluid.mass_fractions,
                                 temperature=fluid.Ts + temperature_rise, type=fluid.type, phase="liquid",
                                 volumetric_expansion_coefficient=fluid.volumetric_expansion_coefficient,
                                 liquid_elasticity=fluid.liquid_elasticity)
        fluid.Ps = new_Ps  # bar
        return fluid, mdot_outlet


def calculate_combustion_chamber_performance(mdot_oxidizer, mdot_fuel, oxidizer, fuel, CC_pressure_at_injector, CR,
                                             eps):
    """A function to calculate the combustion chamber performance.

    :param float or int mdot_oxidizer: Oxidizer massflow (kg/s)
    :param float or int mdot_fuel: Fuel massflow (kg/s)
    :param RocketCycleFluid oxidizer: RocketCycleFluid representing oxidizer
    :param RocketCycleFluid fuel: RocketCycleFluid representing fuel
    :param float or int CC_pressure_at_injector: CC pressure (in bar) at the injector plate
    :param float or int CR: CC contraction ratio
    :param float or int eps: CC expansion ratio

    :return: Full CEA output, CC plenum pressure, vacuum and sea level specific impulse, combustion temperature,
        vacuum and sea level thrust, throat and exit areas
    """

    # Get total massflow and OF
    mdot_total = mdot_oxidizer + mdot_fuel  # kg / s
    OF = mdot_oxidizer / mdot_fuel

    # Retrieve cards and add them to CEA
    rcea.add_new_fuel(name="fuel card", card_str=fuel.CEA_card)
    rcea.add_new_oxidizer(name="oxidizer card", card_str=oxidizer.CEA_card)

    # Create CEA object with Imperial units to be able to get full output
    CC = rcea.CEA_Obj(oxName="oxidizer card", fuelName="fuel card", fac_CR=CR)
    CC_output = CC.get_full_cea_output(Pc=CC_pressure_at_injector, MR=OF, eps=eps, pc_units="bar", output="si",
                                       short_output=1)

    # Create CEA object with SI units for other variables
    CC = CEA_Obj(oxName="oxidizer card", fuelName="fuel card", isp_units='sec', cstar_units='m/s',
                 pressure_units='bar', temperature_units='K', sonic_velocity_units='m/s', enthalpy_units='kJ/kg',
                 density_units='kg/m^3', specific_heat_units='kJ/kg-K', viscosity_units='millipoise',
                 thermal_cond_units='W/cm-degC', fac_CR=CR)
    (IspVac, Cstar, Tcomb) = CC.get_IvacCstrTc(Pc=CC_pressure_at_injector, MR=OF, eps=eps)
    CC_plenum_pressure = CC_pressure_at_injector / CC.get_Pinj_over_Pcomb(Pc=CC_pressure_at_injector, MR=OF)  # bar

    # Get throat and exit area
    A_t = Cstar * mdot_total / (CC_plenum_pressure * 1e5)  # m^2
    A_e = A_t * eps  # m^2

    # Get vacuum thrust, sea level thrust and Isp
    ThrustVac = IspVac * 9.80665 * mdot_total  # N
    ThrustSea = ThrustVac - 1.01325 * 1e5 * A_e  # N
    IspSea = ThrustSea / (mdot_total * 9.80665)  # s

    return CC_output, CC_plenum_pressure, IspVac, IspSea, Tcomb, ThrustVac / 1e3, ThrustSea / 1e3, A_t, A_e
