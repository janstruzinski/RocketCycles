import RocketCycleElements
from RocketCycleFluid import PyFluid_to_RocketCycleFluid
import pyfluids
import scipy.optimize as opt
import numpy as np


class CycleParameters:
    def __init__(self):
        """A class to store some of the data about the cycle"""

        # Massflows
        self.mdot_fuel = None
        self.mdot_oxidizer = None
        self.mdot_film = None
        self.mdot_crossflow_oxidizer = None
        self.mdot_crossflow_fuel = None
        self.mdot_cooling_channels_outlet = None
        self.mdot_f_FPB = None
        self.mdot_FT = None
        self.mdot_ox_OPB = None
        self.mdot_OT = None

        # OF ratios
        self.OF_FPB = None
        self.OF_OPB = None

        # Fluids
        self.fuel = None
        self.oxidizer = None
        self.pumped_fuel = None
        self.pumped_oxidizer = None
        self.heated_fuel = None
        self.FPB_products = None
        self.FT_outlet_gas = None
        self.FT_equilibrium_gas = None
        self.OPB_products = None
        self.OT_outlet_gas = None
        self.OT_equilibrium_gas = None

        # Powers and specific powers
        self.w_pumped_fuel = None
        self.Power_FP = None
        self.w_pumped_oxidizer = None
        self.Power_OP = None

        # CEA output strings
        self.FPB_CEA_output = None
        self.FT_equilibrium_gas_CEA_output = None
        self.OPB_CEA_output = None
        self.OT_equilibrium_gas_CEA_output = None
        self.CC_CEA_output = None

        # Pressures
        self.P_inj_FPB = None
        self.P_inj_OPB = None
        self.P_inj_CC = None
        self.P_plenum_CC = None

        # Pressure ratios
        self.FT_beta_tt = None
        self.OT_beta_tt = None

        # CC parameters
        self.IspVac_real = None
        self.IspSea_real = None
        self.ThrustVac = None
        self.ThrustSea = None
        self.A_t_CC = None
        self.A_e_CC = None

        # Temperatures
        self.CC_Tcomb = None


class FFSC_LRE:
    def __init__(self, OF, ThrustSea, oxidizer, fuel, fuel_CEA_name, oxidizer_CEA_name, T_oxidizer, T_fuel,
                 P_oxidizer, P_fuel, P_plenum_CC, T_FPB, T_OPB, eta_isotropic_OP, eta_isotropic_FP, eta_polytropic_OT,
                 eta_polytropic_FT, eta_FPB, eta_OPB, eta_cstar, eta_isp, dP_over_Pinj_CC, dP_over_Pinj_OPB,
                 dP_over_Pinj_FPB, CR_CC, eps_CC, mdot_film_over_mdot_fuel, cooling_channels_pressure_drop,
                 cooling_channels_temperature_rise, axial_velocity_OT, axial_velocity_FT, mdot_total_0,
                 mdot_crossflow_ox_over_mdot_ox_0, mdot_crossflow_f_over_mdot_f_0, dP_FP_0, dP_OP_0, lb, ub):
        """A class to analyse full flow staged combustion cycle.



        """

        # Assign top level parameters
        self.OF = OF
        self.ThrustSea = ThrustSea
        self.P_plenum_CC = P_plenum_CC

        # Define propellants
        self.fuel = fuel
        self.fuel_CEA_name = fuel_CEA_name
        self.oxidizer = oxidizer
        self.oxidizer_CEA_name = oxidizer_CEA_name

        # Define propellants temperatures and pressure
        self.T_oxidizer = T_oxidizer
        self.T_fuel = T_fuel
        self.P_oxidizer = P_oxidizer
        self.P_fuel = P_fuel

        # Assign temperature constraints
        self.T_FPB = T_FPB
        self.T_OPB = T_OPB

        # Assign efficiencies
        self.eta_isotropic_OP = eta_isotropic_OP
        self.eta_isotropic_FP = eta_isotropic_FP
        self.eta_polytropic_OT = eta_polytropic_OT
        self.eta_polytropic_FT = eta_polytropic_FT
        self.eta_FPB = eta_FPB
        self.eta_OPB = eta_OPB
        self.eta_cstar = eta_cstar
        self.eta_isp = eta_isp

        # Assign pressure drops
        self.dP_over_Pinj_CC = dP_over_Pinj_CC
        self.dP_over_Pinj_OPB = dP_over_Pinj_OPB
        self.dP_over_Pinj_FPB = dP_over_Pinj_FPB

        # Assign combustion chamber parameters
        self.CR_CC = CR_CC
        self.eps_CC = eps_CC
        self.mdot_film_over_mdot_fuel = mdot_film_over_mdot_fuel
        self.cooling_channels_pressure_drop = cooling_channels_pressure_drop
        self.cooling_channels_temperature_rise = cooling_channels_temperature_rise

        # Assign other parameters
        self.axial_velocity_OT = axial_velocity_OT
        self.axial_velocity_FT = axial_velocity_FT

        # Solve for parameters of the cycle that satisfy all requirements. First get the non-normalized first solution
        # estimate (which is also used as reference values). Inputs and outputs are normalized (wrt reference values)
        # for better convergence, so the first solution estimate needs to be normalized too (it is then just ones).
        # Also get normalized bounds.
        self.x_ref = np.array([mdot_total_0, mdot_crossflow_ox_over_mdot_ox_0, mdot_crossflow_f_over_mdot_f_0, dP_FP_0,
                               dP_OP_0])
        bounds = (lb / self.x_ref, ub / self.x_ref)
        x0 = np.ones(5)

        # Get the solution. Least-squares is used, because it is the only method that would converge.
        result = opt.least_squares(fun=self.get_residuals, x0=x0, jac="3-point", bounds=bounds,
                                   method="dogbox", loss="soft_l1", tr_solver="exact", verbose=2, xtol=1e-10)
        [self.mdot_total, self.mdot_crossflow_ox_over_mdot_ox, self.mdot_crossflow_fuel_over_mdot_fuel, self.dP_FP,
         self.dP_OP] = result.x * self.x_ref

        # Get the remianing parameters stored in CycleParameters object
        (residual_propellants_dP, residual_thrust, residual_CC_pressure, residual_T_FPB, residual_T_OPB, self.CP) = (
            self.analyze_cycle(mdot_total=self.mdot_total,
                               mdot_crossflow_ox_over_mdot_ox=self.mdot_crossflow_ox_over_mdot_ox,
                               mdot_crossflow_fuel_over_mdot_fuel=self.mdot_crossflow_fuel_over_mdot_fuel,
                               dP_FP=self.dP_FP, dP_OP=self.dP_OP))

    def analyze_cycle(self, mdot_total, mdot_crossflow_ox_over_mdot_ox, mdot_crossflow_fuel_over_mdot_fuel, dP_FP,
                      dP_OP):
        """A function to analyze the cycle."""

        # Create an object to store data. Cycle Parameters is used instead of self, such that inner loops in the solver
        # cannot change any global parameters
        CP = CycleParameters()

        # First calculate oxidizer, fuel, film crossflow massflows
        CP.mdot_fuel = mdot_total / (1 + self.OF)
        CP.mdot_oxidizer = self.OF * CP.mdot_fuel
        CP.mdot_film = CP.mdot_fuel * self.mdot_film_over_mdot_fuel
        CP.mdot_crossflow_oxidizer = mdot_crossflow_ox_over_mdot_ox * CP.mdot_oxidizer
        CP.mdot_crossflow_fuel = mdot_crossflow_fuel_over_mdot_fuel * CP.mdot_fuel

        # Create Pyfluids objects for propellants
        CP.fuel = pyfluids.Fluid(self.fuel).with_state(
            pyfluids.Input.pressure(self.P_fuel * 1e5), pyfluids.Input.temperature(self.T_fuel - 273.15))
        CP.oxidizer = pyfluids.Fluid(self.oxidizer).with_state(
            pyfluids.Input.pressure(self.P_oxidizer * 1e5), pyfluids.Input.temperature(self.T_oxidizer - 273.15))

        # First calculate states after pumps. Change oxidizer into RocketCycleFluid object.
        CP.pumped_fuel, CP.w_pumped_fuel = RocketCycleElements.calculate_state_after_pump_for_PyFluids(
            fluid=CP.fuel, delta_P=dP_FP, efficiency=self.eta_isotropic_FP)
        CP.Power_FP = CP.w_pumped_fuel * CP.mdot_fuel

        CP.pumped_oxidizer, CP.w_pumped_oxidizer = RocketCycleElements.calculate_state_after_pump_for_PyFluids(
            fluid=CP.oxidizer, delta_P=dP_OP, efficiency=self.eta_isotropic_OP)
        CP.Power_OP = CP.w_pumped_oxidizer * CP.mdot_oxidizer
        CP.pumped_oxidizer = PyFluid_to_RocketCycleFluid(fluid=CP.pumped_oxidizer, CEA_name=self.oxidizer_CEA_name,
                                                         type="oxidizer", phase="liquid")

        # Go over fuel side of the system. Calculate state after cooling channels and change fuel into RocketCycleFluid
        # object.
        CP.heated_fuel, CP.mdot_cooling_channels_outlet = (
            RocketCycleElements.calculate_state_after_cooling_channels_for_Pyfluids(
                fluid=CP.pumped_fuel, mdot_coolant=CP.mdot_fuel, mdot_film=CP.mdot_film,
                pressure_drop=self.cooling_channels_pressure_drop,
                temperature_rise=self.cooling_channels_temperature_rise))
        CP.heated_fuel = PyFluid_to_RocketCycleFluid(fluid=CP.heated_fuel, CEA_name=self.fuel_CEA_name, type="fuel",
                                                     phase="liquid")

        # Calculate state after fuel preburner
        CP.mdot_f_FPB = CP.mdot_cooling_channels_outlet - CP.mdot_crossflow_fuel
        CP.OF_FPB = CP.mdot_crossflow_oxidizer / CP.mdot_f_FPB
        # For determining preburner pressure, use minimum propellant pressure
        CP.P_inj_FPB = min(CP.heated_fuel.Pt, CP.pumped_oxidizer.Pt) / (1 + self.dP_over_Pinj_FPB)
        CP.FPB_CEA_output, CP.FPB_products = RocketCycleElements.calculate_state_after_preburner(
            OF=CP.OF_FPB, preburner_inj_pressure=CP.P_inj_FPB, products_velocity=self.axial_velocity_FT,
            preburner_eta=self.eta_FPB, fuel=CP.heated_fuel, oxidizer=CP.pumped_oxidizer)

        # Calculate state after fuel turbine
        CP.mdot_FT = CP.mdot_f_FPB + CP.mdot_crossflow_oxidizer
        CP.FT_beta_tt, CP.FT_outlet_gas, CP.FT_equilibrium_gas, CP.FT_equilibrium_gas_CEA_output = (
            RocketCycleElements.calculate_state_after_turbine(
                massflow=CP.mdot_FT, turbine_power=CP.Power_FP, turbine_polytropic_efficiency=self.eta_polytropic_FT,
                inlet_gas=CP.FPB_products, turbine_axial_velocity=self.axial_velocity_FT))

        # Now go over oxidizer side of the system. Calculate state after oxygen preburner.
        CP.mdot_ox_OPB = CP.mdot_oxidizer - CP.mdot_crossflow_oxidizer
        CP.OF_OPB = CP.mdot_ox_OPB / CP.mdot_crossflow_fuel
        # For determining preburner pressure, use minimum propellant pressure
        CP.P_inj_OPB = min(CP.pumped_oxidizer.Pt, CP.heated_fuel.Pt) / (1 + self.dP_over_Pinj_OPB)
        CP.OPB_CEA_output, CP.OPB_products = RocketCycleElements.calculate_state_after_preburner(
            OF=CP.OF_OPB, preburner_inj_pressure=CP.P_inj_OPB, products_velocity=self.axial_velocity_OT,
            preburner_eta=self.eta_OPB, fuel=CP.heated_fuel, oxidizer=CP.pumped_oxidizer)

        # Calculate state after oxygen turbine.
        CP.mdot_OT = CP.mdot_ox_OPB + CP.mdot_crossflow_fuel
        CP.OT_beta_tt, CP.OT_outlet_gas, CP.OT_equilibrium_gas, CP.OT_equilibrium_gas_CEA_output = (
            RocketCycleElements.calculate_state_after_turbine(
                massflow=CP.mdot_OT, turbine_power=CP.Power_OP, turbine_polytropic_efficiency=self.eta_polytropic_OT,
                inlet_gas=CP.OPB_products, turbine_axial_velocity=self.axial_velocity_OT))

        # Calculate combustion chamber performance. It does not matter with respect to which propellant pressure we
        # calculate its CC pressure, as it is imposed that these are the same below. Total pressure is used because
        # the gas should slow down in the turbine outlet manifold
        CP.P_inj_CC = CP.FT_equilibrium_gas.Pt / (1 + self.dP_over_Pinj_CC)
        (CP.CC_CEA_output, CP.P_plenum_CC, CP.IspVac_real, CP.IspSea_real, CP.CC_Tcomb, CP.ThrustVac, CP.ThrustSea,
         CP.A_t_CC, CP.A_e_CC) = (RocketCycleElements.calculate_combustion_chamber_performance(
            mdot_oxidizer=CP.mdot_OT, mdot_fuel=CP.mdot_FT, oxidizer=CP.OT_equilibrium_gas,
            fuel=CP.FT_equilibrium_gas, CC_pressure_at_injector=CP.P_inj_CC, CR=self.CR_CC, eps=self.eps_CC,
            eta_cstar=self.eta_cstar, eta_isp=self.eta_isp))

        # Get residuals. These will allow to find input parameters that allow to get feasible cycle.
        # All residuals are normalized.
        # Calculate pressure difference residual - fuel and oxidizer should be at the same pressure before injection
        # into CC.
        residual_propellants_dP = (CP.FT_equilibrium_gas.Pt - CP.OT_equilibrium_gas.Pt) / self.P_plenum_CC

        # Get thrust residual
        residual_thrust = (CP.ThrustSea - self.ThrustSea) / self.ThrustSea

        # Get CC pressure residual
        residual_CC_pressure = (CP.P_plenum_CC - self.P_plenum_CC) / self.P_plenum_CC

        # Get preburner temperature residuals
        residual_T_FPB = (CP.FPB_products.Ts - self.T_FPB) / self.T_FPB
        residual_T_OPB = (CP.OPB_products.Ts - self.T_OPB) / self.T_OPB

        # Return everything
        return (residual_propellants_dP, residual_thrust, residual_CC_pressure, residual_T_FPB, residual_T_OPB, CP)

    def get_residuals(self, x):
        """A function to get the residuals, which need to be zero to satisfy all constraints"""
        # Retrieve arguments
        [mdot_total, mdot_crossflow_ox_over_mdot_ox, mdot_crossflow_fuel_over_mdot_fuel, dP_FP, dP_OP] = x * self.x_ref

        # Analyze the cycle
        (residual_propellants_dP, residual_thrust, residual_CC_pressure, residual_T_FPB, residual_T_OPB, CP) = (
            self.analyze_cycle(mdot_total=mdot_total, mdot_crossflow_ox_over_mdot_ox=mdot_crossflow_ox_over_mdot_ox,
                               mdot_crossflow_fuel_over_mdot_fuel=mdot_crossflow_fuel_over_mdot_fuel,
                               dP_FP=dP_FP, dP_OP=dP_OP))

        # Return only the residuals
        return [residual_propellants_dP, residual_thrust, residual_CC_pressure, residual_T_FPB, residual_T_OPB]

    def get_full_output(self):
        """A function to return the string with the results."""
        string = \
            (f"--- INPUT PARAMETERS ---\n"
             f"---Top level parameters---\n"
             f"O/F: {self.OF}   Thrust: {self.ThrustSea} kN    CC plenum pressure: {self.P_plenum_CC} bar\n"
             f"---Propellants---\n"
             f"Fuel:"
             f"{self.fuel_CEA_name}   Temperature: {self.T_fuel} K   Pressure: {self.P_fuel}\n"
             f"Oxidizer:"
             f"{self.oxidizer_CEA_name}   Temperature: {self.T_oxidizer} K   Pressure: {self.P_oxidizer}\n"
             f"---Efficiencies---\n"
             f" - OP isotropic efficiency: {self.eta_isotropic_OP}   "
             f" - FP isotropic efficiency: {self.eta_isotropic_FP}\n"
             f" - OT polytropic efficiency: {self.eta_polytropic_OT}  "
             f" - FT polytropic efficiency: {self.eta_polytropic_FT}\n"
             f" - FPB efficiency: {self.eta_FPB}     "
             f" - OPB efficiency: {self.eta_OPB}\n"
             f" - C* efficiency: {self.eta_cstar}   "
             f" - Isp efficiency: {self.eta_isp}\n"
             f"---Pressure drop ratios---\n"
             f"Over CC injector: {self.dP_over_Pinj_CC}     Over OPB injector:{self.dP_over_Pinj_OPB}       "
             f"Over FPB injector:{self.dP_over_Pinj_FPB}\n"
             f"---Other parameters---\n"
             f"CC contraction ratio: {self.CR_CC}   CC expansion ratio: {self.eps_CC}\n"
             f"Film cooling massflow to fuel massflow: {self.mdot_film_over_mdot_fuel}\n"
             f"Cooling channels pressure drop: {self.cooling_channels_pressure_drop} bar    "
             f"Cooling channels temperature rise: {self.cooling_channels_temperature_rise} K\n"
             f"OT axial velocity: {self.axial_velocity_OT} m/s      FT axial velocity: {self.axial_velocity_OT} m/s\n\n"
             f"---MASSFLOWS---\n"
             f"Total massflow: {self.mdot_total} kg/s   Oxidizer massflow: {self.CP.mdot_oxidizer} kg/s     "
             f"Fuel massflow: {self.CP.mdot_fuel} kg/s\n"
             f"Oxidizer crossflow massflow: {self.CP.mdot_crossflow_oxidizer} kg/s  "
             f"Fuel crossflow massflow: {self.CP.mdot_crossflow_fuel} kg/s  "
             f"Film cooling massflow: {self.CP.mdot_film}\n\n"
             f"---FUEL SIDE----\n"
             f"---Fuel Pump---\n"
             f"FP pressure rise: {self.dP_FP} bar   "
             f"FP temperature rise: {self.CP.pumped_fuel.temperature + 273.15 - self.T_fuel} K   "
             f"Pump power: {self.CP.Power_FP * 1e-3} kW\n"
             f"---Cooling channels---\n"
             f"Fuel temperature: {self.CP.heated_fuel.Ts} K     Fuel pressure: {self.CP.heated_fuel.Ps} bar\n"
             f"---Fuel preburner---\n"
             f"Fuel massflow: {self.CP.mdot_f_FPB} kg/s     Preburner OF: {self.CP.OF_FPB}\n"
             f"Products static temperature: {self.CP.FPB_products.Ts} K     "
             f"Products total temperature: {self.CP.FPB_products.Tt} K\n"
             f"Pressure at injector: {self.CP.P_inj_FPB} bar  "
             f"Plenum static pressure: {self.CP.FPB_products.Ps} bar    "
             f"Plenum total pressure: {self.CP.FPB_products.Pt} bar\n"
             f"---Fuel turbine---\n"
             f"Massflow: {self.CP.mdot_FT} kg/s     Turbine beta_tt: {self.CP.FT_beta_tt}\n"
             f"Outlet gas static tempetature: {self.CP.FT_outlet_gas.Ts} K  "
             f"Outlet gas total tempetature: {self.CP.FT_outlet_gas.Tt} K\n"
             f"Outlet gas static pressure: {self.CP.FT_outlet_gas.Ps} bar  "
             f"Outlet gas total pressure: {self.CP.FT_outlet_gas.Pt} bar\n\n"
             f"---OXIDIZER SIDE---\n"
             f"---Oxidizer pump---\n"
             f"OP pressure rise: {self.dP_OP} bar   "
             f"OP temperature rise: {self.CP.pumped_oxidizer.Ts - self.T_oxidizer} K   "
             f"Pump power: {self.CP.Power_OP * 1e-3} kW\n"
             f"---Oxidizer preburner---\n"
             f"Oxidizer massflow: {self.CP.mdot_ox_OPB} kg/s     Preburner OF: {self.CP.OF_OPB}\n"
             f"Products static temperature: {self.CP.OPB_products.Ts} K     "
             f"Products total temperature: {self.CP.OPB_products.Tt} K\n"
             f"Pressure at injector: {self.CP.P_inj_OPB} bar  "
             f"Plenum static pressure: {self.CP.OPB_products.Ps} bar    "
             f"Plenum total pressure: {self.CP.OPB_products.Pt} bar\n"
             f"---Oxidizer turbine---\n"
             f"Massflow: {self.CP.mdot_OT} kg/s     Turbine beta_tt: {self.CP.OT_beta_tt}\n"
             f"Outlet gas static tempetature: {self.CP.OT_outlet_gas.Ts} K  "
             f"Outlet gas total tempetature: {self.CP.OT_outlet_gas.Tt} K\n"
             f"Outlet gas static pressure: {self.CP.OT_outlet_gas.Ps} bar  "
             f"Outlet gas total pressure: {self.CP.OT_outlet_gas.Pt} bar\n\n"
             f"---COMBUSTION CHAMBER---\n"
             f"Pressure at injector: {self.CP.P_inj_CC} bar   Plenum pressure: {self.CP.P_plenum_CC} bar "
             f"Combustion temperature: {self.CP.CC_Tcomb} K\n"
             f"Vacuum ISP: {self.CP.IspVac_real} s   Sea ISP: {self.CP.IspSea_real} s\n"
             f"Vacuum thrust: {self.CP.ThrustVac} kN    Sea thrust: {self.CP.ThrustSea} kN\n"
             f"Throat area: {self.CP.A_t_CC} m2    Nozzle exit area:  {self.CP.A_e_CC} m2\n\n"
             f"---CEA OUTPUTS---\n\n"
             f"---FPB CEA output---\n"
             f"{self.CP.FPB_CEA_output}\n\n"
             f"---FT equilibrium gas CEA output---\n"
             f"{self.CP.FT_equilibrium_gas_CEA_output}\n\n"
             f"---OPB CEA output---\n"
             f"{self.CP.OPB_CEA_output}\n\n"
             f"---OT equilibrium gas CEA output---\n"
             f"{self.CP.OT_equilibrium_gas_CEA_output}\n\n"
             f"---CC CEA Output---\n"
             f"{self.CP.CC_CEA_output}"
             )

        return string
