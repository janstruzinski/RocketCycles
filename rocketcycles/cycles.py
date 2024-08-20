from rocketcycles import elements
from rocketcycles.fluid import pyfluid_to_rocket_cycle_fluid, RocketCycleFluid
import pyfluids
import scipy.optimize as opt
import numpy as np
import warnings


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
        self.mdot_cooling_channels_inlet = None
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
        self.heated_oxidizer = None
        self.FPB_products = None
        self.FT_outlet_gas = None
        self.FT_equilibrium_gas = None
        self.OPB_products = None
        self.catalyst_products = None
        self.OT_outlet_gas = None
        self.OT_equilibrium_gas = None
        self.boosted_fuel = None

        # Powers and specific powers
        self.w_pumped_fuel = None
        self.Power_FP = None
        self.w_pumped_oxidizer = None
        self.Power_OP = None
        self.w_boosted_fuel = None
        self.Power_BFP = None
        self.OT_shaft_power = None

        # CEA output strings
        self.FPB_CEA_output = None
        self.FT_equilibrium_gas_CEA_output = None
        self.OPB_CEA_output = None
        self.catalyst_CEA_output = None
        self.OT_equilibrium_gas_CEA_output = None
        self.CC_CEA_output = None

        # Pressures
        self.P_inj_FPB = None
        self.P_inj_OPB = None
        self.P_inj_catalyst = None
        self.P_inj_CC = None
        self.P_plenum_CC = None
        self.dP_OP = None
        self.dP_BFP = None

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

        # Other
        self.OT_molar_Cp_average = None
        self.OT_molar_Cp_average = None
        self.FT_gamma_average = None
        self.FT_gamma_average = None


class FFSC_LRE:
    def __init__(self, OF, oxidizer, fuel, fuel_CEA_name, oxidizer_CEA_name, T_oxidizer, T_fuel,
                 P_oxidizer, P_fuel, eta_isotropic_OP, eta_isotropic_FP, eta_polytropic_OT,
                 eta_polytropic_FT, eta_FPB, eta_OPB, eta_cstar, eta_isp, Ps_Pt_OT, Ps_Pt_FT, dP_over_Pinj_CC,
                 dP_over_Pinj_OPB, dP_over_Pinj_FPB, CR_FPB, CR_OPB, CR_CC, eps_CC, mdot_film_over_mdot_fuel,
                 cooling_channels_pressure_drop, cooling_channels_temperature_rise, axial_velocity_OT,
                 axial_velocity_FT, mdot_total_0, mdot_crossflow_f_over_mdot_f_0,
                 dP_FP_0, mode, mdot_crossflow_ox_over_mdot_ox_0=None, dP_OP_0=None, ThrustSea=None, P_plenum_CC=None,
                 T_FPB=None, T_OPB=None, lb=None, ub=None, jac=None, method=None, loss=None, tr_solver=None, xtol=None,
                 ftol=None, fscale=1, diff_step=None):
        """A class to analyse and size full flow staged combustion cycle.

        :param string mode: "analysis" or "sizing". "Analysis" will do cycle calculations for given mdot_total_0,
         mdot_crossflow_ox_over_mdot_ox_0, mdot_crossflow_f_over_mdot_f_0, dP_FP_0, dP_OP_0. "Sizing" will find values
          for these that give desired ThrustSea, P_plenum_CC, T_FPB, T_OPB using scipy.optimize.least_squares.
           In that case, only mdot_total_0, mdot_crossflow_f_over_mdot_f_0, dP_FP_0 need to be provided for the first
            estimate of the solution.
        :param float or int OF: Oxidizer-to-Fuel ratio of the cycle
        :param pyfluids.FluidsList oxidizer: PyFluids object representing oxidizer
        :param pyfluids.FluidsList fuel: PyFluids object representing fuel
        :param string fuel_CEA_name: CEA name of fuel
        :param string oxidizer_CEA_name: CEA name of oxidizer
        :param float or int T_oxidizer: Temperature of inlet oxidizer (K)
        :param float or int T_fuel: Temperature of inlet fuel (K)
        :param float or int P_oxidizer: Pressure of inlet oxidizer (bar)
        :param float or int P_fuel: Pressure of inlet fuel (bar)
        :param float or int eta_isotropic_OP: Isotropic efficiency of the oxidizer pump (-)
        :param float or int eta_isotropic_FP: Isotropic efficiency of the fuel pump (-)
        :param float or int eta_polytropic_OT: Polytropic efficiency of the oxidizer turbine (-)
        :param float or int eta_polytropic_FT: Polytropic efficiency of the fuel turbine (-)
        :param float or int eta_FPB: Efficiency of the fuel preburner (-) defined as ratio of actual to ideal
         temperature
        :param float or int eta_OPB: Efficiency of the oxidizer preburner (-) defined as ratio of actual to ideal
         temperature
        :param float or int eta_cstar: C* efficiency of the CC (-)
        :param float or int eta_isp: Isp efficiency of the CC (-)
        :param float Ps_Pt_OT: Pressure recovery factor for oxidizer turbine. It is a ratio of static to total pressure
         recovered in diffuser and manifold after turbine (-)
        :param float Ps_Pt_FT: Pressure recovery factor for oxidizer turbine. It is a ratio of static to total pressure
         recovered in diffuser and manifold after turbine (-)
        :param float or int dP_over_Pinj_FPB: Pressure drop ratio for the fuel preburner (-)
        :param float or int dP_over_Pinj_OPB: Pressure drop ratio for the oxidizer preburner (-)
        :param float or int dP_over_Pinj_CC: Pressure drop ratio for the combustion chamber (-)
        :param float or int CR_FPB: Contraction ratio of the fuel preburner (-), used as a measure of its size
        (as preburner does not usually have a sonic throat)
        :param float or int CR_OPB: Contraction ratio of the oxidizer preburner (-), used as a measure of its size
        (as preburner does not usually have a sonic throat)
        :param float or int CR_CC: Contraction ratio of the combustion chamber (-)
        :param float or int eps_CC: Expansion ratio of the combustion chamber (-)
        :param float or int mdot_film_over_mdot_fuel: Ratio of the fuel film cooling massflow to total fuel massflow (-)
        :param float or int cooling_channels_pressure_drop: Pressure drop in the cooling channels (bar)
        :param float or int cooling_channels_temperature_rise: Temperature rise in the cooling channels (K)
        :param float or int axial_velocity_OT: Axial velocity across oxidizer turbine (m/s)
        :param float or int axial_velocity_FT: Axial velocity across fuel turbine (m/s)
        :param float or int mdot_total_0: Total massflow (kg/s). In "sizing" mode, first estimate of it.
        :param float or int mdot_crossflow_f_over_mdot_f_0: Fuel crossflow massflow to fuel massflow (-).
         In "sizing" mode, first estimate of it.
        :param float or int mdot_crossflow_ox_over_mdot_ox_0: Oxidizer crossflow massflow to oxidizer massflow (-).
         Only for "analysis" mode. In "sizing" mode, it is calculated.
        :param float or int dP_FP_0: Pressure rise (bar) in the fuel pump. In "sizing" mode, first estimate of it.
        :param float or int dP_OP_0: Pressure rise (bar) in the oxidizer pump. Only for "analysis" mode.
         In "sizing" mode, it is calculated.
        :param float or int ThrustSea: Desired thrust (kN) at the sea level for "sizing" mode.
        :param float or int P_plenum_CC: Desired CC plenum pressure (bar) for "sizing" mode.
        :param float or int T_FPB: Desired fuel preburner temperature (K) for "sizing" mode.
        :param float or int T_OPB: Desired oxidizer preburner temperature (K) for "sizing" mode.
        :param list lb: A list of lower bounds for variables mdot_total,
        mdot_crossflow_f_over_mdot_f, dP_FP, when "sizing" mode is used.
        :param list ub: A list of upper bounds for variables mdot_total,
        mdot_crossflow_f_over_mdot_f, dP_FP, when "sizing" mode is used.
        :param string jac: jac argument for scipy.optimize.least_squares
        :param string method: method argument for scipy.optimize.least_squares
        :param string loss: loss argument for scipy.optimize.least_squares
        :param string tr_solver: tr_solver argument for scipy.optimize.least_squares
        :param float or int xtol: xtol argument for scipy.optimize.least_squares
        :param float or int ftol: ftol argument for scipy.optimize.least_squares
        :param float or int fscale: fscale argument for scipy.optimize.least_squares
        :param float or int diff_step: diff_step argument for scipy.optimize.least_squares
        """

        # First do user warnings if wrong combinations of arguments are called
        if mode == "analysis" and (mdot_crossflow_ox_over_mdot_ox_0 is None or dP_OP_0 is None):
            warnings.simplefilter("error", UserWarning)
            warnings.warn("mdot_crossflow_ox_over_mdot_ox_0 and dP_OP_0 arguments need to be defined to analysis mode")

        elif mode == "sizing" and (mdot_crossflow_ox_over_mdot_ox_0 is not None or dP_OP_0 is not None):
            warnings.simplefilter("error", UserWarning)
            warnings.warn("mdot_crossflow_ox_over_mdot_ox_0 and dP_OP_0 arguments cannot be be defined for sizing mode")

        elif (mode == "sizing" and any([ThrustSea, P_plenum_CC, T_FPB, T_OPB, lb, ub, jac, method, loss, tr_solver]) is
              None):
            warnings.simplefilter("error", UserWarning)
            warnings.warn("Arguments missing for sizing mode")

        # Assign in which mode the object is used
        self.mode = mode

        # Assign top level parameters
        self.OF = OF

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

        # Assign efficiencies
        self.eta_isotropic_OP = eta_isotropic_OP
        self.eta_isotropic_FP = eta_isotropic_FP
        self.eta_polytropic_OT = eta_polytropic_OT
        self.eta_polytropic_FT = eta_polytropic_FT
        self.eta_FPB = eta_FPB
        self.eta_OPB = eta_OPB
        self.eta_cstar = eta_cstar
        self.eta_isp = eta_isp
        self.Ps_Pt_OT = Ps_Pt_OT
        self.Ps_Pt_FT = Ps_Pt_FT

        # Assign pressure drops
        self.dP_over_Pinj_CC = dP_over_Pinj_CC
        self.dP_over_Pinj_OPB = dP_over_Pinj_OPB
        self.dP_over_Pinj_FPB = dP_over_Pinj_FPB

        # Assign combustion chamber and preburner parameters
        self.CR_FPB = CR_FPB
        self.CR_OPB = CR_OPB
        self.CR_CC = CR_CC
        self.eps_CC = eps_CC
        self.mdot_film_over_mdot_fuel = mdot_film_over_mdot_fuel
        self.cooling_channels_pressure_drop = cooling_channels_pressure_drop
        self.cooling_channels_temperature_rise = cooling_channels_temperature_rise

        # Assign other parameters
        self.axial_velocity_OT = axial_velocity_OT
        self.axial_velocity_FT = axial_velocity_FT

        # Assign required parameters/constraints. (For "analysis" mode these will be just None)
        self.ThrustSea = ThrustSea
        self.P_plenum_CC = P_plenum_CC
        self.T_FPB = T_FPB
        self.T_OPB = T_OPB

        # If mode is analysis, analyse the cycle.
        if self.mode == "analysis":
            # CP stores parameters of the cycle
            self.CP = (self.analyze_cycle(mdot_total=mdot_total_0,
                                          mdot_crossflow_ox_over_mdot_ox=mdot_crossflow_ox_over_mdot_ox_0,
                                          mdot_crossflow_fuel_over_mdot_fuel=mdot_crossflow_f_over_mdot_f_0,
                                          dP_FP=dP_FP_0, dP_OP=dP_OP_0))
            # Assign these variables, so that the same get_full_output function can be used for both sizing and
            # analysis modes. It just means that variables that would be first estimate for sizing mode, are the
            # final parameters in analysis mode.
            self.mdot_total = mdot_total_0
            self.dP_FP = dP_FP_0

        # If mode is to size the cycle, find solution that satisfies requirements.
        elif self.mode == "sizing":
            # Solve for parameters of the cycle that satisfy all requirements. First get the non-normalized first
            # solution estimate (which is also used as reference values). Inputs are normalized (wrt
            # reference values) for better convergence, so the first solution estimate needs to be normalized too (it
            # is then just ones). Also get normalized upper and lower bounds.
            self.x_ref = np.array([mdot_total_0, mdot_crossflow_f_over_mdot_f_0, dP_FP_0])
            bounds = (lb / self.x_ref, ub / self.x_ref)
            x0 = np.ones(3)

            # Get initial residuals, both non- and non-normalized. Residuals are also normalized wrt to some reference
            # scales.
            self.norm_residuals_0 = self.calculate_residuals(x0)
            self.residuals_0 = self.norm_residuals_0 * np.array([self.ThrustSea, self.P_plenum_CC, self.T_OPB])

            # Get the solution. Least-squares is used, because it is the only method that would converge.
            # Solution is normalized, so it needs to be scaled again.
            result = opt.least_squares(fun=self.calculate_residuals, x0=x0, jac=jac, bounds=bounds,
                                       method=method, loss=loss, tr_solver=tr_solver, xtol=xtol, ftol=ftol,
                                       f_scale=fscale, verbose=2, diff_step=diff_step, x_scale="jac")
            [self.mdot_total, self.mdot_crossflow_fuel_over_mdot_fuel, self.dP_FP] = result.x * self.x_ref

            # Get residuals for the solution, both normalized and non-normalized.
            self.norm_residuals = self.calculate_residuals(result.x)
            self.residuals = self.norm_residuals * np.array([self.ThrustSea, self.P_plenum_CC, self.T_OPB])

            # Get the remaining parameters stored in CycleParameters object
            self.CP = (self.analyze_cycle(mdot_total=self.mdot_total,
                                          mdot_crossflow_fuel_over_mdot_fuel=self.mdot_crossflow_fuel_over_mdot_fuel,
                                          dP_FP=self.dP_FP))

    def analyze_cycle(self, mdot_total, mdot_crossflow_fuel_over_mdot_fuel, dP_FP, mdot_crossflow_ox_over_mdot_ox=None,
                      dP_OP=None):
        """A function to analyze the cycle. If mode is "analysis", the algorithm will simply perform it for given
         arguments. If mode is "sizing", only the first three arguments are found numerically by the top level solver.
          The other two parameters will be calculated inside the function itself. Oxidizer crossflow massflow will be
           calculated based on required FPB temperature, while oxidizer pressure pump will be determined from minimum
            pressure required to pump oxidizer into FPB. This helps with coupling between variables and convergence.

        :param int or float mdot_total: Total massflow (kg/s)
        :param int or float mdot_crossflow_ox_over_mdot_ox: Oxidizer crossflow massflow to oxidizer massflow (-)
        :param int or float mdot_crossflow_fuel_over_mdot_fuel: Fuel crossflow massflow to fuel massflow (-)
        :param int or float dP_FP: Pressure rise (bar) in the fuel pump.
        :param int or float dP_OP: Pressure rise (bar) in the oxidizer pump.

        :return: ClassParameters object storing data about the cycle.
        """

        # Create an object to store data. Cycle Parameters is used instead of self, such that inner loops in the solver
        # cannot change any global parameters
        CP = CycleParameters()

        # First calculate oxidizer, fuel, film crossflow massflows
        CP.mdot_fuel = mdot_total / (1 + self.OF)
        CP.mdot_oxidizer = self.OF * CP.mdot_fuel
        CP.mdot_film = CP.mdot_fuel * self.mdot_film_over_mdot_fuel
        CP.mdot_crossflow_fuel = mdot_crossflow_fuel_over_mdot_fuel * CP.mdot_fuel
        # If mode is analysis, oxidizer crossflow is already known. Otherwise, it will be calculated based on FPB
        # temperature later on.
        if self.mode == "analysis":
            CP.mdot_crossflow_oxidizer = mdot_crossflow_ox_over_mdot_ox * CP.mdot_oxidizer

        # Create Pyfluids Fluid objects for propellants with correct units (Pa and deg C)
        CP.fuel = pyfluids.Fluid(self.fuel).with_state(
            pyfluids.Input.pressure(self.P_fuel * 1e5), pyfluids.Input.temperature(self.T_fuel - 273.15))
        CP.oxidizer = pyfluids.Fluid(self.oxidizer).with_state(
            pyfluids.Input.pressure(self.P_oxidizer * 1e5), pyfluids.Input.temperature(self.T_oxidizer - 273.15))

        # Go over fuel side of the system. First calculate states after fuel pump and power required to drive it.
        CP.pumped_fuel, CP.w_pumped_fuel = elements.calculate_state_after_pump_for_pyfluids(
            fluid=CP.fuel, delta_P=dP_FP, efficiency=self.eta_isotropic_FP)
        CP.Power_FP = CP.w_pumped_fuel * CP.mdot_fuel

        # Calculate state after cooling channels and change both heated and pumped
        # fuel into RocketCycleFluid object. Pumped fuel will be still used later on for oxygen preburner,
        # hence it is changed to RocketCycleFluid.
        CP.heated_fuel, CP.mdot_cooling_channels_outlet = (
            elements.calculate_state_after_cooling_channels_for_Pyfluids(
                fluid=CP.pumped_fuel, mdot_coolant=CP.mdot_fuel, mdot_film=CP.mdot_film,
                pressure_drop=self.cooling_channels_pressure_drop,
                temperature_rise=self.cooling_channels_temperature_rise))
        CP.heated_fuel = pyfluid_to_rocket_cycle_fluid(fluid=CP.heated_fuel, CEA_name=self.fuel_CEA_name, type="fuel",
                                                       phase="liquid")
        CP.pumped_fuel = pyfluid_to_rocket_cycle_fluid(fluid=CP.pumped_fuel, CEA_name=self.fuel_CEA_name, type="fuel",
                                                       phase="liquid")

        # Now perform calculations related to fuel preburner. First calculate its pressure based on fuel total pressure.
        CP.P_inj_FPB = CP.heated_fuel.Pt / (1 + self.dP_over_Pinj_FPB)

        # Before we calculate state after fuel preburner, we need to get our pumped oxidizer. For sizing mode, oxidizer
        # pump outlet pressure must simply match fuel pressure after cooling channels, so that both can inject
        # propellants into FPB at the same pressure ratio. Alternatively, if mode is analysis, dP_OP is already known.
        # We can already calculate power required to drive oxidizer pump as well.
        # Get pressure rise if mode is sizing:
        if self.mode == "sizing":
            dP_OP = CP.heated_fuel.Pt - self.P_oxidizer
        # dP_OP should be parameter of CP regardless of mode, so that the same get_full_output function can be used for
        # both modes.
        CP.dP_OP = dP_OP
        # Get state after oxidizer pump and power required to drive it
        CP.pumped_oxidizer, CP.w_pumped_oxidizer = elements.calculate_state_after_pump_for_pyfluids(
            fluid=CP.oxidizer, delta_P=CP.dP_OP, efficiency=self.eta_isotropic_OP)
        CP.Power_OP = CP.w_pumped_oxidizer * CP.mdot_oxidizer
        # Change oxidizer into RocketCycleFluid object.
        CP.pumped_oxidizer = pyfluid_to_rocket_cycle_fluid(fluid=CP.pumped_oxidizer, CEA_name=self.oxidizer_CEA_name,
                                                           type="oxidizer", phase="liquid")

        # Fuel massflow into FPB needs to be calculated too:
        CP.mdot_f_FPB = CP.mdot_cooling_channels_outlet - CP.mdot_crossflow_fuel

        # For sizing mode, oxidizer massflow that will allow to achieve certain FPB temperature needs to be found.
        # For analysis, it is already known.
        # Find oxidizer massflow if sizing mode is used:
        if self.mode == "sizing":
            # Oxidizer massflow that will allow to achieve certain FPB temperature can be found using bisection
            # algorithm. First define a function for it to solve.
            def calculate_FPB_temperature_residual(mdot_ox):
                # Get its OF ratio.
                OF_FPB = mdot_ox / CP.mdot_f_FPB
                # Get results for FPB.
                FPB_CEA_output, FPB_products = elements.calculate_state_after_preburner(
                    OF=OF_FPB, preburner_inj_pressure=CP.P_inj_FPB, CR=self.CR_FPB,
                    preburner_eta=self.eta_FPB, fuel=CP.heated_fuel, oxidizer=CP.pumped_oxidizer)
                # Return residual. Total temperature is used, since this is what will be next to the walls due to gas
                # slowing down due to boundary layer.
                return FPB_products.Tt - self.T_FPB

            # Now solve the function to find the massflow that satisfies requirements. 0.04 and 0.11 are good lower
            # and upper bounds for any reasonable FPB temperature.
            CP.mdot_crossflow_oxidizer = opt.toms748(calculate_FPB_temperature_residual, a=0.04 * CP.mdot_oxidizer,
                                                     b=0.11 * CP.mdot_oxidizer, maxiter=1000, xtol=1e-1)
        # Now get FPB OF ratio.
        CP.OF_FPB = CP.mdot_crossflow_oxidizer / CP.mdot_f_FPB
        # And get results for FPB.
        CP.FPB_CEA_output, CP.FPB_products = elements.calculate_state_after_preburner(
            OF=CP.OF_FPB, preburner_inj_pressure=CP.P_inj_FPB, CR=self.CR_FPB,
            preburner_eta=self.eta_FPB, fuel=CP.heated_fuel, oxidizer=CP.pumped_oxidizer)

        # Calculate state after fuel turbine
        CP.mdot_FT = CP.mdot_f_FPB + CP.mdot_crossflow_oxidizer
        (CP.FT_beta_tt, CP.FT_outlet_gas, CP.FT_equilibrium_gas, CP.FT_equilibrium_gas_CEA_output,
         CP.FT_molar_Cp_average, CP.FT_gamma_average) = (
            elements.calculate_state_after_turbine(
                massflow=CP.mdot_FT, turbine_power=CP.Power_FP, turbine_polytropic_efficiency=self.eta_polytropic_FT,
                preburner_products=CP.FPB_products, turbine_axial_velocity=self.axial_velocity_FT,
                pressure_recovery_factor=self.Ps_Pt_FT))

        # Now go over oxidizer side of the system. Calculate state after oxygen preburner. Again first calculate
        # oxidizer massflow through it and preburner OF.
        CP.mdot_ox_OPB = CP.mdot_oxidizer - CP.mdot_crossflow_oxidizer
        CP.OF_OPB = CP.mdot_ox_OPB / CP.mdot_crossflow_fuel
        # For determining preburner pressure, use minimum propellant pressure
        CP.P_inj_OPB = min(CP.pumped_oxidizer.Pt, CP.pumped_fuel.Pt) / (1 + self.dP_over_Pinj_OPB)
        # Get preburner results
        CP.OPB_CEA_output, CP.OPB_products = elements.calculate_state_after_preburner(
            OF=CP.OF_OPB, preburner_inj_pressure=CP.P_inj_OPB, CR=self.CR_OPB,
            preburner_eta=self.eta_OPB, fuel=CP.pumped_fuel, oxidizer=CP.pumped_oxidizer)

        # Calculate state after oxygen turbine.
        CP.mdot_OT = CP.mdot_ox_OPB + CP.mdot_crossflow_fuel
        (CP.OT_beta_tt, CP.OT_outlet_gas, CP.OT_equilibrium_gas, CP.OT_equilibrium_gas_CEA_output,
         CP.OT_molar_Cp_average, CP.OT_gamma_average) = (
            elements.calculate_state_after_turbine(
                massflow=CP.mdot_OT, turbine_power=CP.Power_OP, turbine_polytropic_efficiency=self.eta_polytropic_OT,
                preburner_products=CP.OPB_products, turbine_axial_velocity=self.axial_velocity_OT,
                pressure_recovery_factor=self.Ps_Pt_OT))

        # Calculate combustion chamber performance. CC pressure at injector is determined wrt to minimum propellant
        # pressure. Total pressure is used because the gas should slow down in the turbine outlet manifold.
        CP.P_inj_CC = min(CP.FT_equilibrium_gas.Ps, CP.OT_equilibrium_gas.Ps) / (1 + self.dP_over_Pinj_CC)
        (CP.CC_CEA_output, CP.P_plenum_CC, CP.IspVac_real, CP.IspSea_real, CP.CC_Tcomb, CP.ThrustVac, CP.ThrustSea,
         CP.A_t_CC, CP.A_e_CC) = (elements.calculate_combustion_chamber_performance(
            mdot_oxidizer=CP.mdot_OT, mdot_fuel=CP.mdot_FT, oxidizer=CP.OT_equilibrium_gas,
            fuel=CP.FT_equilibrium_gas, CC_pressure_at_injector=CP.P_inj_CC, CR=self.CR_CC, eps=self.eps_CC,
            eta_cstar=self.eta_cstar, eta_isp=self.eta_isp))

        # Return CycleParameters object storing data about cycle
        return CP

    def calculate_residuals(self, x):
        """A function to get the residuals, which need to be zero to satisfy all constraints.

        :param np.ndarray x: Normalized estimate of the solution.

        """
        # Retrieve non-normalized arguments
        [mdot_total, mdot_crossflow_fuel_over_mdot_fuel, dP_FP] = x * self.x_ref

        # Analyze the cycle
        CP = (self.analyze_cycle(mdot_total=mdot_total,
                                 mdot_crossflow_fuel_over_mdot_fuel=mdot_crossflow_fuel_over_mdot_fuel,
                                 dP_FP=dP_FP))

        # Get residuals. These will allow to find input parameters that allow to get feasible cycle.
        # All residuals are normalized wrt some reference scales.
        # Get thrust residual
        residual_thrust = (CP.ThrustSea - self.ThrustSea) / self.ThrustSea

        # Get CC pressure residual
        residual_CC_pressure = (CP.P_plenum_CC - self.P_plenum_CC) / self.P_plenum_CC

        # Get OPB temperature residuals. Again total temperature is used, bcs it will be closer to adiabatic temperature
        # near the walls.
        residual_T_OPB = (CP.OPB_products.Tt - self.T_OPB) / self.T_OPB

        # Return the residuals
        return [residual_thrust, residual_CC_pressure, residual_T_OPB]

    def get_full_output(self):
        """A function to return the string with data about the cycle."""
        string = \
            (f"\n\n--- INPUT PARAMETERS ---\n"
             f"---Top level parameters---\n"
             f"O/F: {self.OF}   Required thrust: {self.ThrustSea} kN    "
             f"Required CC plenum pressure: {self.P_plenum_CC} bar\n"
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
             f" - OT pressure recovery factor: {self.Ps_Pt_OT}"
             f" - FT pressure recovery factor: {self.Ps_Pt_FT}\n"
             f"---Pressure drop ratios---\n"
             f"Over CC injector: {self.dP_over_Pinj_CC}     Over OPB injector:{self.dP_over_Pinj_OPB}       "
             f"Over FPB injector:{self.dP_over_Pinj_FPB}\n"
             f"---Other parameters---\n"
             f"CC contraction ratio: {self.CR_CC}   CC expansion ratio: {self.eps_CC}\n"
             f"Film cooling massflow to fuel massflow: {self.mdot_film_over_mdot_fuel}\n"
             f"Cooling channels pressure drop: {self.cooling_channels_pressure_drop} bar    "
             f"Cooling channels temperature rise: {self.cooling_channels_temperature_rise} K\n"
             f"Required OPB temperature: {self.T_OPB} K    Required FPB temperature: {self.T_FPB} K\n"
             f"OPB CR: {self.CR_OPB}    FPB CR: {self.CR_FPB} \n"
             f"OT axial velocity: {self.axial_velocity_OT} m/s      FT axial velocity: {self.axial_velocity_FT} m/s\n\n"
             f"---MASSFLOWS---\n"
             f"Total massflow: {self.mdot_total} kg/s   Oxidizer massflow: {self.CP.mdot_oxidizer} kg/s     "
             f"Fuel massflow: {self.CP.mdot_fuel} kg/s\n"
             f"Oxidizer crossflow massflow: {self.CP.mdot_crossflow_oxidizer} kg/s  "
             f"Fuel crossflow massflow: {self.CP.mdot_crossflow_fuel} kg/s  "
             f"Film cooling massflow: {self.CP.mdot_film}\n\n"
             f"---FUEL SIDE----\n"
             f"---Fuel Pump---\n"
             f"FP pressure rise: {self.dP_FP} bar   "
             f"FP temperature rise: {self.CP.pumped_fuel.Ts - self.T_fuel} K   "
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
             f"Outlet gas total pressure: {self.CP.FT_outlet_gas.Pt} bar\n"
             f"Gas static temperature in CC manifold: {self.CP.FT_equilibrium_gas.Ts} K  "
             f"Gas static pressure in CC manifold: {self.CP.FT_equilibrium_gas.Ps} bar\n\n"
             f"---OXIDIZER SIDE---\n"
             f"---Oxidizer pump---\n"
             f"OP pressure rise: {self.CP.dP_OP} bar   "
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
             f"Outlet gas total pressure: {self.CP.OT_outlet_gas.Pt} bar\n"
             f"Gas static temperature in CC manifold: {self.CP.OT_equilibrium_gas.Ts} K  "
             f"Gas static pressure in CC manifold: {self.CP.OT_equilibrium_gas.Ps} bar\n\n"
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

    def get_residuals(self):
        """A function to return string about initial and final residuals."""

        string = \
            (f"\n---INITIAL RESIDUALS---\n"
             f"- Difference between actual and desired sea thrust: {self.residuals_0[0]} kN"
             f" (normalized: {self.norm_residuals_0[0]})\n"
             f"- Pressure difference between actual and desired CC plenum pressure: {self.residuals_0[1]} bar"
             f" (normalized: {self.norm_residuals_0[1]})\n"
             f"- Temperature difference between actual and desired temperature in OPB: {self.residuals_0[2]} K"
             f" (normalized: {self.norm_residuals_0[2]})\n\n"
             f"\n---REMAINING RESIDUALS---\n"
             f"- Difference between actual and desired sea thrust: {self.residuals[0]} kN"
             f" (normalized: {self.norm_residuals[0]})\n"
             f"- Pressure difference between actual and desired CC plenum pressure: {self.residuals[1]} bar"
             f" (normalized: {self.norm_residuals[1]})\n"
             f"- Temperature difference between actual and desired temperature in OPB: {self.residuals[2]} K"
             f" (normalized: {self.norm_residuals[2]})\n\n"
             )
        return string


class ORSC_LRE:
    def __init__(self, OF, oxidizer, fuel, fuel_CEA_name, oxidizer_CEA_name, T_oxidizer, T_fuel,
                 P_oxidizer, P_fuel, eta_isotropic_OP, eta_isotropic_FP, eta_isotropic_BFP, eta_polytropic_OT,
                 eta_OPB, eta_cstar, eta_isp, Ps_Pt_OT, dP_over_Pinj_CC, dP_over_Pinj_OPB, CR_OPB, CR_CC, eps_CC,
                 mdot_film_over_mdot_fuel, cooling_channels_pressure_drop, cooling_channels_temperature_rise,
                 axial_velocity_OT, mdot_total_0, dP_FP_0, dP_OP_0, mode, mdot_crossflow_f_over_mdot_f_0=None,
                 ThrustSea=None, P_plenum_CC=None, T_OPB=None, lb=None, ub=None, jac=None, method=None, loss=None,
                 tr_solver=None, xtol=None, ftol=None, fscale=1, diff_step=None):
        """A class to analyse and size oxygen rich staged combustion cycle.

        :param string mode: "analysis" or "sizing". "Analysis" will do cycle calculations for given mdot_total_0,
         mdot_crossflow_f_over_mdot_f_0, dP_FP_0, dP_OP_0. "Sizing" will find values
          for these that give desired ThrustSea, P_plenum_CC, T_OPB using scipy.optimize.least_squares.
           In that case, only mdot_total_0, dP_OP_0, dP_FP_0 need to be provided for the first
            estimate of the solution.
        :param float or int OF: Oxidizer-to-Fuel ratio of the cycle
        :param pyfluids.FluidsList oxidizer: PyFluids object representing oxidizer
        :param pyfluids.FluidsList fuel: PyFluids object representing fuel
        :param string fuel_CEA_name: CEA name of fuel
        :param string oxidizer_CEA_name: CEA name of oxidizer
        :param float or int T_oxidizer: Temperature of inlet oxidizer (K)
        :param float or int T_fuel: Temperature of inlet fuel (K)
        :param float or int P_oxidizer: Pressure of inlet oxidizer (bar)
        :param float or int P_fuel: Pressure of inlet fuel (bar)
        :param float or int eta_isotropic_OP: Isotropic efficiency of the oxidizer pump (-)
        :param float or int eta_isotropic_FP: Isotropic efficiency of the fuel pump (-)
        :param float or int eta_isotropic_BFP: Isotropic efficiency of the booster fuel pump (-)
        :param float or int eta_polytropic_OT: Polytropic efficiency of the oxidizer turbine (-)
        :param float or int eta_OPB: Efficiency of the oxidizer preburner (-) defined as ratio of actual to ideal
         temperature
        :param float or int eta_cstar: C* efficiency of the CC (-)
        :param float or int eta_isp: Isp efficiency of the CC (-)
        :param float Ps_Pt_OT: Pressure recovery factor for oxidizer turbine. It is a ratio of static to total pressure
         recovered in diffuser and manifold after turbine (-)
        :param float or int dP_over_Pinj_OPB: Pressure drop ratio for the oxidizer preburner (-)
        :param float or int dP_over_Pinj_CC: Pressure drop ratio for the combustion chamber (-)
        :param float or int CR_OPB: Contraction ratio of the oxidizer preburner (-), used as a measure of its size
        (as preburner does not usually have a sonic throat)
        :param float or int CR_CC: Contraction ratio of the combustion chamber (-)
        :param float or int eps_CC: Expansion ratio of the combustion chamber (-)
        :param float or int mdot_film_over_mdot_fuel: Ratio of the fuel film cooling massflow to total fuel massflow (-)
        :param float or int cooling_channels_pressure_drop: Pressure drop in the cooling channels (bar)
        :param float or int cooling_channels_temperature_rise: Temperature rise in the cooling channels (K)
        :param float or int axial_velocity_OT: Axial velocity across oxidizer turbine (m/s)
        :param float or int mdot_total_0: Total massflow (kg/s). In "sizing" mode, first estimate of it.
        :param float or int mdot_crossflow_f_over_mdot_f_0: Fuel crossflow massflow to fuel massflow (-).
         Only for "analysis" mode. In "sizing" mode, it is calculated.
        :param float or int dP_FP_0: Pressure rise (bar) in the fuel pump. In "sizing" mode, first estimate of it.
        :param float or int dP_OP_0: Pressure rise (bar) in the oxidizer pump. In "sizing" mode, first estimate of it.
        :param float or int ThrustSea: Desired thrust (kN) at the sea level for "sizing" mode.
        :param float or int P_plenum_CC: Desired CC plenum pressure (bar) for "sizing" mode.
        :param float or int T_OPB: Desired oxidizer preburner temperature (K) for "sizing" mode.
        :param list lb: A list of lower bounds for variables mdot_total,
        dP_OP, dP_FP, when "sizing" mode is used.
        :param list ub: A list of upper bounds for variables mdot_total,
        dP_OP, dP_FP, when "sizing" mode is used.
        :param string jac: jac argument for scipy.optimize.least_squares
        :param string method: method argument for scipy.optimize.least_squares
        :param string loss: loss argument for scipy.optimize.least_squares
        :param string tr_solver: tr_solver argument for scipy.optimize.least_squares
        :param float or int xtol: xtol argument for scipy.optimize.least_squares
        :param float or int ftol: ftol argument for scipy.optimize.least_squares
        :param float or int fscale: fscale argument for scipy.optimize.least_squares
        :param float or int diff_step: diff_step argument for scipy.optimize.least_squares
        """

        # First do user warnings if wrong combinations of arguments are called
        if mode == "analysis" and mdot_crossflow_f_over_mdot_f_0 is None:
            warnings.simplefilter("error", UserWarning)
            warnings.warn("mdot_crossflow_f_over_mdot_f_0 needs to be defined to analysis mode")

        elif mode == "sizing" and mdot_crossflow_f_over_mdot_f_0 is not None:
            warnings.simplefilter("error", UserWarning)
            warnings.warn("mdot_crossflow_f_over_mdot_f_0 argument cannot be be defined for sizing mode")

        elif (mode == "sizing" and any([ThrustSea, P_plenum_CC, T_OPB, lb, ub, jac, method, loss, tr_solver]) is
              None):
            warnings.simplefilter("error", UserWarning)
            warnings.warn("Arguments missing for sizing mode")

        # Assign in which mode the object is used
        self.mode = mode

        # Assign top level parameters
        self.OF = OF

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

        # Assign efficiencies
        self.eta_isotropic_OP = eta_isotropic_OP
        self.eta_isotropic_FP = eta_isotropic_FP
        self.eta_isotropic_BFP = eta_isotropic_BFP
        self.eta_polytropic_OT = eta_polytropic_OT
        self.eta_OPB = eta_OPB
        self.eta_cstar = eta_cstar
        self.eta_isp = eta_isp
        self.Ps_Pt_OT = Ps_Pt_OT

        # Assign pressure drops
        self.dP_over_Pinj_CC = dP_over_Pinj_CC
        self.dP_over_Pinj_OPB = dP_over_Pinj_OPB

        # Assign combustion chamber and preburner parameters
        self.CR_OPB = CR_OPB
        self.CR_CC = CR_CC
        self.eps_CC = eps_CC
        self.mdot_film_over_mdot_fuel = mdot_film_over_mdot_fuel
        self.cooling_channels_pressure_drop = cooling_channels_pressure_drop
        self.cooling_channels_temperature_rise = cooling_channels_temperature_rise

        # Assign other parameters
        self.axial_velocity_OT = axial_velocity_OT

        # Assign required parameters/constraints. (For "analysis" mode these will be just None)
        self.ThrustSea = ThrustSea
        self.P_plenum_CC = P_plenum_CC
        self.T_OPB = T_OPB

        # If mode is analysis, analyse the cycle.
        if self.mode == "analysis":
            # CP stores parameters of the cycle
            self.CP = (self.analyze_cycle(mdot_total=mdot_total_0,
                                          mdot_crossflow_fuel_over_mdot_fuel=mdot_crossflow_f_over_mdot_f_0,
                                          dP_FP=dP_FP_0, dP_OP=dP_OP_0))
            # Assign these variables, so that the same get_full_output function can be used for both sizing and
            # analysis modes. It just means that variables that would be first estimate for sizing mode, are the
            # final parameters in analysis mode.
            self.mdot_total = mdot_total_0
            self.dP_FP = dP_FP_0
            self.dP_OP = dP_OP_0

        # If mode is to size the cycle, find solution that satisfies requirements.
        elif self.mode == "sizing":
            # Solve for parameters of the cycle that satisfy all requirements. First get the non-normalized first
            # solution estimate (which is also used as reference values). Inputs are normalized (wrt
            # reference values) for better convergence, so the first solution estimate needs to be normalized too (it
            # is then just ones). Also get normalized upper and lower bounds.
            self.x_ref = np.array([mdot_total_0, dP_OP_0, dP_FP_0])
            bounds = (lb / self.x_ref, ub / self.x_ref)
            x0 = np.ones(3)

            # Get initial residuals, both non- and non-normalized. Residuals are also normalized wrt to some reference
            # scales.
            self.norm_residuals_0 = self.calculate_residuals(x0)
            self.residuals_0 = self.norm_residuals_0 * np.array([self.ThrustSea, self.P_plenum_CC, self.P_plenum_CC])

            # Get the solution. Least-squares is used, because it is the only method that would converge.
            # Solution is normalized, so it needs to be scaled again.
            result = opt.least_squares(fun=self.calculate_residuals, x0=x0, jac=jac, bounds=bounds,
                                       method=method, loss=loss, tr_solver=tr_solver, xtol=xtol, ftol=ftol,
                                       f_scale=fscale, verbose=2, diff_step=diff_step, x_scale="jac")
            [self.mdot_total, self.dP_OP, self.dP_FP] = result.x * self.x_ref

            # Get residuals for the solution, both normalized and non-normalized.
            self.norm_residuals = self.calculate_residuals(result.x)
            self.residuals = self.norm_residuals * np.array([self.ThrustSea, self.P_plenum_CC, self.P_plenum_CC])

            # Get the remaining parameters stored in CycleParameters object
            self.CP = self.analyze_cycle(mdot_total=self.mdot_total, dP_FP=self.dP_FP, dP_OP=self.dP_OP)

    def analyze_cycle(self, mdot_total, dP_OP, dP_FP, mdot_crossflow_fuel_over_mdot_fuel=None):
        """A function to analyze the cycle. If mode is "analysis", the algorithm will simply perform it for given
         arguments. If mode is "sizing", only the first three arguments are found numerically by the top level solver.
          Fuel crossflow massflow will be calculated based on required OPB temperature. This helps with coupling between
           variables and convergence.

        :param int or float mdot_total: Total massflow (kg/s)
        :param int or float dP_FP: Pressure rise (bar) in the fuel pump.
        :param int or float dP_OP: Pressure rise (bar) in the oxidizer pump.
        :param int or float mdot_crossflow_fuel_over_mdot_fuel: Fuel crossflow massflow to fuel massflow (-)


        :return: ClassParameters object storing data about the cycle.
        """

        # Create an object to store data. Cycle Parameters is used instead of self, such that inner loops in the solver
        # cannot change any global parameters
        CP = CycleParameters()

        # First calculate oxidizer, fuel, film crossflow massflows
        CP.mdot_fuel = mdot_total / (1 + self.OF)
        CP.mdot_oxidizer = self.OF * CP.mdot_fuel
        CP.mdot_film = CP.mdot_fuel * self.mdot_film_over_mdot_fuel
        # If mode is analysis, oxidizer crossflow is already known. Otherwise, it will be calculated based on FPB
        # temperature later on.
        if self.mode == "analysis":
            CP.mdot_crossflow_fuel = mdot_crossflow_fuel_over_mdot_fuel * CP.mdot_fuel

        # Create Pyfluids Fluid objects for propellants with correct units (Pa and deg C)
        CP.fuel = pyfluids.Fluid(self.fuel).with_state(
            pyfluids.Input.pressure(self.P_fuel * 1e5), pyfluids.Input.temperature(self.T_fuel - 273.15))
        CP.oxidizer = pyfluids.Fluid(self.oxidizer).with_state(
            pyfluids.Input.pressure(self.P_oxidizer * 1e5), pyfluids.Input.temperature(self.T_oxidizer - 273.15))

        # First go over the oxidizer side of the system. Get state after oxidizer pump and power required to drive it
        CP.pumped_oxidizer, CP.w_pumped_oxidizer = elements.calculate_state_after_pump_for_pyfluids(
            fluid=CP.oxidizer, delta_P=dP_OP, efficiency=self.eta_isotropic_OP)
        CP.Power_OP = CP.w_pumped_oxidizer * CP.mdot_oxidizer
        # Change oxidizer into RocketCycleFluid object.
        CP.pumped_oxidizer = pyfluid_to_rocket_cycle_fluid(fluid=CP.pumped_oxidizer, CEA_name=self.oxidizer_CEA_name,
                                                           type="oxidizer", phase="liquid")

        # Now perform calculations related to oxidizer preburner. First calculate its pressure based on oxidizer total
        # pressure.
        CP.P_inj_OPB = CP.pumped_oxidizer.Pt / (1 + self.dP_over_Pinj_OPB)

        # Before we calculate state after oxidizer preburner, we need to get our pumped amd boosted fuel. The pressure
        # rise across the main fuel pump is given, so the pressure rise in the boosted pump needs to be calculated.
        # First calculate states after the main fuel pump and power required to drive it.
        CP.pumped_fuel, CP.w_pumped_fuel = elements.calculate_state_after_pump_for_pyfluids(
            fluid=CP.fuel, delta_P=dP_FP, efficiency=self.eta_isotropic_FP)
        CP.Power_FP = CP.w_pumped_fuel * CP.mdot_fuel
        # Now calculate the booster pump pressure rise
        CP.dP_BFP = CP.pumped_oxidizer.Pt - (CP.pumped_fuel.pressure / 1e5)  # bar
        # Now get the state after booster pump
        CP.boosted_fuel, CP.w_boosted_fuel = elements.calculate_state_after_pump_for_pyfluids(
            fluid=CP.pumped_fuel, delta_P=CP.dP_BFP, efficiency=self.eta_isotropic_BFP)
        # Change boosted fuel into RocketCycleFluid object
        CP.boosted_fuel = pyfluid_to_rocket_cycle_fluid(fluid=CP.boosted_fuel, CEA_name=self.fuel_CEA_name,
                                                        type="fuel", phase="liquid")

        # For sizing mode, fuel massflow that will allow to achieve certain OPB temperature needs to be found.
        # For analysis, it is already known.
        # Find oxidizer massflow if sizing mode is used:
        if self.mode == "sizing":
            # Fuel massflow that will allow to achieve certain FPB temperature can be found using bisection
            # algorithm. First define a function for it to solve.
            def calculate_OPB_temperature_residual(mdot_f):
                # Get its OF ratio.
                OF_OPB = CP.mdot_oxidizer / mdot_f
                # Get results for FPB.
                OPB_CEA_output, OPB_products = elements.calculate_state_after_preburner(
                    OF=OF_OPB, preburner_inj_pressure=CP.P_inj_OPB, CR=self.CR_OPB,
                    preburner_eta=self.eta_OPB, fuel=CP.boosted_fuel, oxidizer=CP.pumped_oxidizer)
                # Return residual. Total temperature is used, since this is what will be next to the walls due to gas
                # slowing down due to boundary layer.
                return OPB_products.Tt - self.T_OPB

            # Now solve the function to find the massflow that satisfies requirements. 0.04 and 0.10 are good lower
            # and upper bounds for any reasonable OPB temperature.
            CP.mdot_crossflow_fuel = opt.toms748(calculate_OPB_temperature_residual, a=0.04 * CP.mdot_fuel,
                                                 b=0.10 * CP.mdot_fuel, maxiter=1000, xtol=1e-1)
        # Now get OPB OF ratio.
        CP.OF_OPB = CP.mdot_oxidizer / CP.mdot_crossflow_fuel
        # And get results for FPB.
        CP.OPB_CEA_output, CP.OPB_products = elements.calculate_state_after_preburner(
            OF=CP.OF_OPB, preburner_inj_pressure=CP.P_inj_OPB, CR=self.CR_OPB,
            preburner_eta=self.eta_OPB, fuel=CP.boosted_fuel, oxidizer=CP.pumped_oxidizer)
        # Also get the power required to drive the booster pump
        CP.Power_BFP = CP.w_boosted_fuel * CP.mdot_crossflow_fuel

        # Calculate state after oxygen turbine.
        CP.mdot_OT = CP.mdot_oxidizer + CP.mdot_crossflow_fuel
        CP.OT_shaft_power = CP.Power_BFP + CP.Power_FP + CP.Power_OP
        (CP.OT_beta_tt, CP.OT_outlet_gas, CP.OT_equilibrium_gas, CP.OT_equilibrium_gas_CEA_output,
         CP.OT_molar_Cp_average, CP.OT_gamma_average) = (
            elements.calculate_state_after_turbine(
                massflow=CP.mdot_OT, turbine_power=CP.OT_shaft_power,
                turbine_polytropic_efficiency=self.eta_polytropic_OT, preburner_products=CP.OPB_products,
                turbine_axial_velocity=self.axial_velocity_OT, pressure_recovery_factor=self.Ps_Pt_OT))

        # Calculate state after cooling channels and change heated fuel into RocketCycleFluid object.
        CP.mdot_cooling_channels_inlet = CP.mdot_fuel - CP.mdot_crossflow_fuel
        CP.heated_fuel, CP.mdot_cooling_channels_outlet = (
            elements.calculate_state_after_cooling_channels_for_Pyfluids(
                fluid=CP.pumped_fuel, mdot_coolant=CP.mdot_cooling_channels_inlet, mdot_film=CP.mdot_film,
                pressure_drop=self.cooling_channels_pressure_drop,
                temperature_rise=self.cooling_channels_temperature_rise))
        CP.heated_fuel = pyfluid_to_rocket_cycle_fluid(fluid=CP.heated_fuel, CEA_name=self.fuel_CEA_name, type="fuel",
                                                       phase="liquid")

        # Calculate combustion chamber performance. For sizing mode, it does not matter wrt which propellant CC pressure
        # at injector is established, as it will be imposed by the residuals that their total pressure is the same.
        # Using consitent pressure will help with potential seesaw effect in the solver.
        if self.mode == "sizing":
            CP.P_inj_CC = CP.heated_fuel.Pt / (1 + self.dP_over_Pinj_CC)
        # However, for analysis mode, the smaller pressure must be taken (so that CC pressure is always smaller than the
        # injected propellant pressure)
        elif self.mode == "analysis":
            CP.P_inj_CC = min(CP.OT_equilibrium_gas.Ps, CP.heated_fuel.Pt) / (1 + self.dP_over_Pinj_CC)
        # Now get CC results.
        (CP.CC_CEA_output, CP.P_plenum_CC, CP.IspVac_real, CP.IspSea_real, CP.CC_Tcomb, CP.ThrustVac, CP.ThrustSea,
         CP.A_t_CC, CP.A_e_CC) = (elements.calculate_combustion_chamber_performance(
            mdot_oxidizer=CP.mdot_OT, mdot_fuel=CP.mdot_cooling_channels_outlet, oxidizer=CP.OT_equilibrium_gas,
            fuel=CP.heated_fuel, CC_pressure_at_injector=CP.P_inj_CC, CR=self.CR_CC, eps=self.eps_CC,
            eta_cstar=self.eta_cstar, eta_isp=self.eta_isp))

        # Return CycleParameters object storing data about cycle
        return CP

    def calculate_residuals(self, x):
        """A function to get the residuals, which need to be zero to satisfy all constraints.

        :param np.ndarray x: Normalized estimate of the solution.

        """
        # Retrieve non-normalized arguments
        [mdot_total, dP_OP, dP_FP] = x * self.x_ref

        # Analyze the cycle
        CP = (self.analyze_cycle(mdot_total=mdot_total, dP_OP=dP_OP, dP_FP=dP_FP))

        # Get residuals. These will allow to find input parameters that allow to get feasible cycle.
        # All residuals are normalized wrt some reference scales.
        # Get thrust residual
        residual_thrust = (CP.ThrustSea - self.ThrustSea) / self.ThrustSea

        # Get CC pressure residual
        residual_CC_pressure = (CP.P_plenum_CC - self.P_plenum_CC) / self.P_plenum_CC

        # Get propellants pressure difference in the manifold residual. It is desirable that these are the same to
        # limit pressure drops
        residual_dP_propellants = (CP.heated_fuel.Pt - CP.OT_equilibrium_gas.Ps) / self.P_plenum_CC

        # Return the residuals
        return [residual_thrust, residual_CC_pressure, residual_dP_propellants]

    def get_full_output(self):
        """A function to return the string with data about the cycle."""
        string = \
            (f"\n\n--- INPUT PARAMETERS ---\n"
             f"---Top level parameters---\n"
             f"O/F: {self.OF}   Required thrust: {self.ThrustSea} kN    "
             f"Required CC plenum pressure: {self.P_plenum_CC} bar\n"
             f"---Propellants---\n"
             f"Fuel:"
             f"{self.fuel_CEA_name}   Temperature: {self.T_fuel} K   Pressure: {self.P_fuel}\n"
             f"Oxidizer:"
             f"{self.oxidizer_CEA_name}   Temperature: {self.T_oxidizer} K   Pressure: {self.P_oxidizer}\n"
             f"---Efficiencies---\n"
             f" - OP isotropic efficiency: {self.eta_isotropic_OP}   "
             f" - FP isotropic efficiency: {self.eta_isotropic_FP}\n"
             f" - BFP isotropic efficiency: {self.eta_isotropic_BFP}  "
             f" - OT polytropic efficiency: {self.eta_polytropic_OT}\n"
             f" - OPB efficiency: {self.eta_OPB}    "
             f" - C* efficiency: {self.eta_cstar}   "
             f" - Isp efficiency: {self.eta_isp}\n"
             f" - OT pressure recovery factor: {self.Ps_Pt_OT}\n"
             f"---Pressure drop ratios---\n"
             f"Over CC injector: {self.dP_over_Pinj_CC}     Over OPB injector:{self.dP_over_Pinj_OPB}\n"
             f"---Other parameters---\n"
             f"CC contraction ratio: {self.CR_CC}   CC expansion ratio: {self.eps_CC}\n"
             f"Film cooling massflow to fuel massflow: {self.mdot_film_over_mdot_fuel}\n"
             f"Cooling channels pressure drop: {self.cooling_channels_pressure_drop} bar    "
             f"Cooling channels temperature rise: {self.cooling_channels_temperature_rise} K\n"
             f"Required OPB temperature: {self.T_OPB} K    OPB CR: {self.CR_OPB}    "
             f"OT axial velocity: {self.axial_velocity_OT} m/s\n\n"
             f"---MASSFLOWS---\n"
             f"Total massflow: {self.mdot_total} kg/s   Oxidizer massflow: {self.CP.mdot_oxidizer} kg/s     "
             f"Fuel massflow: {self.CP.mdot_fuel} kg/s\n"
             f"Fuel crossflow massflow: {self.CP.mdot_crossflow_fuel} kg/s  "
             f"Film cooling massflow: {self.CP.mdot_film} kg/s\n\n"
             f"---FUEL SIDE----\n"
             f"---Fuel Pump---\n"
             f"FP pressure rise: {self.dP_FP} bar   "
             f"FP temperature rise: {self.CP.pumped_fuel.temperature + 273.15 - self.T_fuel} K   "
             f"Pump power: {self.CP.Power_FP * 1e-3} kW\n"
             f"---Booster Fuel Pump---\n"
             f"BFP pressure rise: {self.CP.dP_BFP} bar   "
             f"BFP temperature rise: {self.CP.boosted_fuel.Ts - (self.CP.pumped_fuel.temperature + 273.15)} K   "
             f"Booster pump power: {self.CP.Power_BFP * 1e-3} kW\n"
             f"---Cooling channels---\n"
             f"Fuel temperature: {self.CP.heated_fuel.Ts} K     Fuel pressure: {self.CP.heated_fuel.Ps} bar\n"
             f"---OXIDIZER SIDE---\n"
             f"---Oxidizer pump---\n"
             f"OP pressure rise: {self.dP_OP} bar   "
             f"OP temperature rise: {self.CP.pumped_oxidizer.Ts - self.T_oxidizer} K   "
             f"Pump power: {self.CP.Power_OP * 1e-3} kW\n"
             f"---Oxidizer preburner---\n"
             f"Preburner OF: {self.CP.OF_OPB}\n"
             f"Products static temperature: {self.CP.OPB_products.Ts} K     "
             f"Products total temperature: {self.CP.OPB_products.Tt} K\n"
             f"Pressure at injector: {self.CP.P_inj_OPB} bar  "
             f"Plenum static pressure: {self.CP.OPB_products.Ps} bar    "
             f"Plenum total pressure: {self.CP.OPB_products.Pt} bar\n"
             f"---Oxidizer turbine---\n"
             f"Massflow: {self.CP.mdot_OT} kg/s     Turbine beta_tt: {self.CP.OT_beta_tt}   "
             f"Turbine shaft power: {self.CP.OT_shaft_power * 1e-3} kW\n"
             f"Outlet gas static tempetature: {self.CP.OT_outlet_gas.Ts} K  "
             f"Outlet gas total tempetature: {self.CP.OT_outlet_gas.Tt} K\n"
             f"Outlet gas static pressure: {self.CP.OT_outlet_gas.Ps} bar  "
             f"Outlet gas total pressure: {self.CP.OT_outlet_gas.Pt} bar\n"
             f"Gas static temperature in CC manifold: {self.CP.OT_equilibrium_gas.Ts} K  "
             f"Gas static pressure in CC manifold: {self.CP.OT_equilibrium_gas.Ps} bar\n\n"
             f"---COMBUSTION CHAMBER---\n"
             f"Pressure at injector: {self.CP.P_inj_CC} bar   Plenum pressure: {self.CP.P_plenum_CC} bar "
             f"Combustion temperature: {self.CP.CC_Tcomb} K\n"
             f"Vacuum ISP: {self.CP.IspVac_real} s   Sea ISP: {self.CP.IspSea_real} s\n"
             f"Vacuum thrust: {self.CP.ThrustVac} kN    Sea thrust: {self.CP.ThrustSea} kN\n"
             f"Throat area: {self.CP.A_t_CC} m2    Nozzle exit area:  {self.CP.A_e_CC} m2\n\n"
             f"---CEA OUTPUTS---\n\n"
             f"---OPB CEA output---\n"
             f"{self.CP.OPB_CEA_output}\n\n"
             f"---OT equilibrium gas CEA output---\n"
             f"{self.CP.OT_equilibrium_gas_CEA_output}\n\n"
             f"---CC CEA Output---\n"
             f"{self.CP.CC_CEA_output}"
             )
        return string

    def get_residuals(self):
        """A function to return string about initial and final residuals."""

        string = \
            (f"\n---INITIAL RESIDUALS---\n"
             f"- Difference between actual and desired sea thrust: {self.residuals_0[0]} kN"
             f" (normalized: {self.norm_residuals_0[0]})\n"
             f"- Pressure difference between actual and desired CC plenum pressure: {self.residuals_0[1]} bar"
             f" (normalized: {self.norm_residuals_0[1]})\n"
             f"- Pressure difference between fuel and oxidizer in the CC manifold: {self.residuals_0[2]} bar"
             f" (normalized: {self.norm_residuals_0[2]})\n\n"
             f"\n---REMAINING RESIDUALS---\n"
             f"- Difference between actual and desired sea thrust: {self.residuals[0]} kN"
             f" (normalized: {self.norm_residuals[0]})\n"
             f"- Pressure difference between actual and desired CC plenum pressure: {self.residuals[1]} bar"
             f" (normalized: {self.norm_residuals[1]})\n"
             f"- Pressure difference between fuel and oxidizer in the CC manifold: {self.residuals[2]} bar"
             f" (normalized: {self.norm_residuals[2]})\n\n"
             )
        return string


class ClosedCatalyst_LRE:
    def __init__(self, OF, oxidizer, fuel, eta_isotropic_OP, eta_isotropic_FP, eta_polytropic_OT,
                 eta_catalyst, eta_cstar, eta_isp, Ps_Pt_OT, dP_over_Pinj_CC, dP_over_Pinj_catalyst, CR_catalyst,
                 CR_CC, eps_CC, mdot_film_over_mdot_oxid, cooling_channels_pressure_drop,
                 cooling_channels_temperature_rise, axial_velocity_OT, mdot_total_0, dP_FP_0, dP_OP_0, mode,
                 ThrustSea=None, P_plenum_CC=None, lb=None, ub=None, jac=None, method=None, loss=None,
                 tr_solver=None, xtol=None, ftol=None, fscale=1, diff_step=None):
        """A class to analyse and size oxygen rich staged combustion cycle.

        :param string mode: "analysis" or "sizing". "Analysis" will do cycle calculations for given mdot_total_0,
         mdot_crossflow_f_over_mdot_f_0, dP_FP_0, dP_OP_0. "Sizing" will find values
          for these that give desired ThrustSea, P_plenum_CC, T_OPB using scipy.optimize.least_squares.
           In that case, only mdot_total_0, dP_OP_0, dP_FP_0 need to be provided for the first
            estimate of the solution.
        :param float or int OF: Oxidizer-to-Fuel ratio of the cycle
        :param RocketCycleFluid oxidizer: PyFluids object representing oxidizer
        :param RocketCycleFluid fuel: PyFluids object representing fuel
        :param float or int eta_isotropic_OP: Isotropic efficiency of the oxidizer pump (-)
        :param float or int eta_isotropic_FP: Isotropic efficiency of the fuel pump (-)
        :param float or int eta_polytropic_OT: Polytropic efficiency of the oxidizer turbine (-)
        :param float or int eta_catalyst: Efficiency of the catalyst (-) defined as ratio of actual to ideal
         temperature (as not all of H2O2 will be decomposed)
        :param float or int eta_cstar: C* efficiency of the CC (-)
        :param float or int eta_isp: Isp efficiency of the CC (-)
        :param float Ps_Pt_OT: Pressure recovery factor for oxidizer turbine. It is a ratio of static to total pressure
         recovered in diffuser and manifold after turbine (-)
        :param float or int dP_over_Pinj_catalyst: Total pressure drop ratio for the catalyst, across the injector and
         the catalyst itself(-)
        :param float or int dP_over_Pinj_CC: Pressure drop ratio for the combustion chamber (-)
        :param float or int CR_catalyst: Contraction ratio of the catalyst (-), used as a measure of its size
        (as preburner does not usually have a sonic throat)
        :param float or int CR_CC: Contraction ratio of the combustion chamber (-)
        :param float or int eps_CC: Expansion ratio of the combustion chamber (-)
        :param float or int mdot_film_over_mdot_oxid: Ratio of the oxidizer film cooling massflow to total oxidizer
         massflow (-)
        :param float or int cooling_channels_pressure_drop: Pressure drop in the cooling channels (bar)
        :param float or int cooling_channels_temperature_rise: Temperature rise in the cooling channels (K)
        :param float or int axial_velocity_OT: Axial velocity across oxidizer turbine (m/s)
        :param float or int mdot_total_0: Total massflow (kg/s). In "sizing" mode, first estimate of it.
        :param float or int dP_FP_0: Pressure rise (bar) in the fuel pump. In "sizing" mode, first estimate of it.
        :param float or int dP_OP_0: Pressure rise (bar) in the oxidizer pump. In "sizing" mode, first estimate of it.
        :param float or int ThrustSea: Desired thrust (kN) at the sea level for "sizing" mode.
        :param float or int P_plenum_CC: Desired CC plenum pressure (bar) for "sizing" mode.
        :param list lb: A list of lower bounds for variables mdot_total,
        dP_OP, dP_FP, when "sizing" mode is used.
        :param list ub: A list of upper bounds for variables mdot_total,
        dP_OP, dP_FP, when "sizing" mode is used.
        :param string jac: jac argument for scipy.optimize.least_squares
        :param string method: method argument for scipy.optimize.least_squares
        :param string loss: loss argument for scipy.optimize.least_squares
        :param string tr_solver: tr_solver argument for scipy.optimize.least_squares
        :param float or int xtol: xtol argument for scipy.optimize.least_squares
        :param float or int ftol: ftol argument for scipy.optimize.least_squares
        :param float or int fscale: fscale argument for scipy.optimize.least_squares
        :param float or int diff_step: diff_step argument for scipy.optimize.least_squares
        """

        # First do user warnings if wrong combinations of arguments are called
        if (mode == "sizing" and any([ThrustSea, P_plenum_CC, lb, ub, jac, method, loss, tr_solver]) is
              None):
            warnings.simplefilter("error", UserWarning)
            warnings.warn("Arguments missing for sizing mode")

        # Assign in which mode the object is used
        self.mode = mode

        # Assign top level parameters
        self.OF = OF

        # Define propellants
        self.fuel = fuel
        self.oxidizer = oxidizer

        # Assign efficiencies
        self.eta_isotropic_OP = eta_isotropic_OP
        self.eta_isotropic_FP = eta_isotropic_FP
        self.eta_polytropic_OT = eta_polytropic_OT
        self.eta_catalyst = eta_catalyst
        self.eta_cstar = eta_cstar
        self.eta_isp = eta_isp
        self.Ps_Pt_OT = Ps_Pt_OT

        # Assign pressure drops
        self.dP_over_Pinj_CC = dP_over_Pinj_CC
        self.dP_over_Pinj_catalyst = dP_over_Pinj_catalyst

        # Assign combustion chamber and catalyst parameters
        self.CR_catalyst = CR_catalyst
        self.CR_CC = CR_CC
        self.eps_CC = eps_CC
        self.mdot_film_over_mdot_oxid = mdot_film_over_mdot_oxid
        self.cooling_channels_pressure_drop = cooling_channels_pressure_drop
        self.cooling_channels_temperature_rise = cooling_channels_temperature_rise

        # Assign other parameters
        self.axial_velocity_OT = axial_velocity_OT

        # Assign required parameters/constraints. (For "analysis" mode these will be just None)
        self.ThrustSea = ThrustSea
        self.P_plenum_CC = P_plenum_CC

        # If mode is analysis, analyse the cycle.
        if self.mode == "analysis":
            # CP stores parameters of the cycle
            self.CP = self.analyze_cycle(mdot_total=mdot_total_0, dP_FP=dP_FP_0, dP_OP=dP_OP_0)
            # Assign these variables, so that the same get_full_output function can be used for both sizing and
            # analysis modes. It just means that variables that would be first estimate for sizing mode, are the
            # final parameters in analysis mode.
            self.mdot_total = mdot_total_0
            self.dP_FP = dP_FP_0
            self.dP_OP = dP_OP_0

        # If mode is to size the cycle, find solution that satisfies requirements.
        elif self.mode == "sizing":
            # Solve for parameters of the cycle that satisfy all requirements. First get the non-normalized first
            # solution estimate (which is also used as reference values). Inputs are normalized (wrt
            # reference values) for better convergence, so the first solution estimate needs to be normalized too (it
            # is then just ones). Also get normalized upper and lower bounds.
            self.x_ref = np.array([mdot_total_0, dP_OP_0, dP_FP_0])
            bounds = (lb / self.x_ref, ub / self.x_ref)
            x0 = np.ones(3)

            # Get initial residuals, both non- and non-normalized. Residuals are also normalized wrt to some reference
            # scales.
            self.norm_residuals_0 = self.calculate_residuals(x0)
            self.residuals_0 = self.norm_residuals_0 * np.array([self.ThrustSea, self.P_plenum_CC, self.P_plenum_CC])

            # Get the solution. Least-squares is used, because it is the only method that would converge.
            # Solution is normalized, so it needs to be scaled again.
            result = opt.least_squares(fun=self.calculate_residuals, x0=x0, jac=jac, bounds=bounds,
                                       method=method, loss=loss, tr_solver=tr_solver, xtol=xtol, ftol=ftol,
                                       f_scale=fscale, verbose=2, diff_step=diff_step, x_scale="jac")
            [self.mdot_total, self.dP_OP, self.dP_FP] = result.x * self.x_ref

            # Get residuals for the solution, both normalized and non-normalized.
            self.norm_residuals = self.calculate_residuals(result.x)
            self.residuals = self.norm_residuals * np.array([self.ThrustSea, self.P_plenum_CC, self.P_plenum_CC])

            # Get the remaining parameters stored in CycleParameters object
            self.CP = self.analyze_cycle(mdot_total=self.mdot_total, dP_FP=self.dP_FP, dP_OP=self.dP_OP)

    def analyze_cycle(self, mdot_total, dP_OP, dP_FP):
        """A function to analyze the cycle. If mode is "analysis", the algorithm will simply perform it for given
         arguments. If mode is "sizing", these arguments are found numerically by the top level solver.

        :param int or float mdot_total: Total massflow (kg/s)
        :param int or float dP_FP: Pressure rise (bar) in the fuel pump.
        :param int or float dP_OP: Pressure rise (bar) in the oxidizer pump.

        :return: ClassParameters object storing data about the cycle.
        """

        # Create an object to store data. Cycle Parameters is used instead of self, such that inner loops in the solver
        # cannot change any global parameters
        CP = CycleParameters()

        # First calculate oxidizer, fuel, film crossflow massflows
        CP.mdot_fuel = mdot_total / (1 + self.OF)
        CP.mdot_oxidizer = self.OF * CP.mdot_fuel
        CP.mdot_film = CP.mdot_oxidizer * self.mdot_film_over_mdot_oxid

        # First go over the oxidizer side of the system. Get state after oxidizer pump and power required to drive it
        CP.pumped_oxidizer, CP.w_pumped_oxidizer = elements.calculate_state_after_pump(
            fluid=self.oxidizer, delta_P=dP_OP, efficiency=self.eta_isotropic_OP)
        CP.Power_OP = CP.w_pumped_oxidizer * CP.mdot_oxidizer

        # Calculate state after cooling channels.
        CP.heated_oxidizer, CP.mdot_cooling_channels_outlet = (
            elements.calculate_state_after_cooling_channels(
                fluid=CP.pumped_oxidizer, mdot_coolant=CP.mdot_oxidizer, mdot_film=CP.mdot_film,
                pressure_drop=self.cooling_channels_pressure_drop,
                temperature_rise=self.cooling_channels_temperature_rise))

        # Now perform calculations related to catalyst bed. First calculate its (outlet) pressure based on oxidizer
        # injection pressure.
        CP.P_inj_catalyst = CP.heated_oxidizer.Pt / (1 + self.dP_over_Pinj_catalyst)

        # And get results for catalyst. Here OPB related parameters in CP are used as placeholder for catalyst
        # parameters.
        CP.catalyst_CEA_output, CP.catalyst_products = elements.calculate_state_after_preburner(
            preburner_inj_pressure=CP.P_inj_catalyst, CR=self.CR_catalyst,
            preburner_eta=self.eta_catalyst, monopropellant=CP.heated_oxidizer)

        # Now calculate state after fuel pump and power required to drive it.
        CP.pumped_fuel, CP.w_pumped_fuel = elements.calculate_state_after_pump(
            fluid=self.fuel, delta_P=dP_FP, efficiency=self.eta_isotropic_FP)
        CP.Power_FP = CP.w_pumped_fuel * CP.mdot_fuel

        # Calculate state after oxygen turbine.
        CP.OT_shaft_power = CP.Power_FP + CP.Power_OP
        CP.mdot_OT = CP.mdot_cooling_channels_outlet
        (CP.OT_beta_tt, CP.OT_outlet_gas, CP.OT_equilibrium_gas, CP.OT_equilibrium_gas_CEA_output,
         CP.OT_molar_Cp_average, CP.OT_gamma_average) = (
            elements.calculate_state_after_turbine(
                massflow=CP.mdot_OT, turbine_power=CP.OT_shaft_power,
                turbine_polytropic_efficiency=self.eta_polytropic_OT, preburner_products=CP.catalyst_products,
                turbine_axial_velocity=self.axial_velocity_OT, pressure_recovery_factor=self.Ps_Pt_OT))

        # Calculate combustion chamber performance. For sizing mode, it does not matter wrt which propellant CC pressure
        # at injector is established, as it will be imposed by the residuals that their total pressure is the same.
        # Using consistent pressure will help with potential seesaw effect in the solver.
        if self.mode == "sizing":
            CP.P_inj_CC = CP.OT_equilibrium_gas.Ps / (1 + self.dP_over_Pinj_CC)
        # However, for analysis mode, the smaller pressure must be taken (so that CC pressure is always smaller than the
        # injected propellant pressure)
        elif self.mode == "analysis":
            CP.P_inj_CC = min(CP.OT_equilibrium_gas.Ps, CP.pumped_fuel.Pt) / (1 + self.dP_over_Pinj_CC)
        # Now get CC results.
        (CP.CC_CEA_output, CP.P_plenum_CC, CP.IspVac_real, CP.IspSea_real, CP.CC_Tcomb, CP.ThrustVac, CP.ThrustSea,
         CP.A_t_CC, CP.A_e_CC) = (elements.calculate_combustion_chamber_performance(
            mdot_oxidizer=CP.mdot_OT, mdot_fuel=CP.mdot_fuel, oxidizer=CP.OT_equilibrium_gas,
            fuel=CP.pumped_fuel, CC_pressure_at_injector=CP.P_inj_CC, CR=self.CR_CC, eps=self.eps_CC,
            eta_cstar=self.eta_cstar, eta_isp=self.eta_isp))

        # Return CycleParameters object storing data about cycle
        return CP

    def calculate_residuals(self, x):
        """A function to get the residuals, which need to be zero to satisfy all constraints.

        :param np.ndarray x: Normalized estimate of the solution.

        """
        # Retrieve non-normalized arguments
        [mdot_total, dP_OP, dP_FP] = x * self.x_ref

        # Analyze the cycle
        CP = (self.analyze_cycle(mdot_total=mdot_total, dP_OP=dP_OP, dP_FP=dP_FP))

        # Get residuals. These will allow to find input parameters that allow to get feasible cycle.
        # All residuals are normalized wrt some reference scales.
        # Get thrust residual
        residual_thrust = (CP.ThrustSea - self.ThrustSea) / self.ThrustSea

        # Get CC pressure residual
        residual_CC_pressure = (CP.P_plenum_CC - self.P_plenum_CC) / self.P_plenum_CC

        # Get propellants pressure difference in the manifold residual. It is desirable that these are the same to
        # limit pressure drops
        residual_dP_propellants = (CP.pumped_fuel.Pt - CP.OT_equilibrium_gas.Ps) / self.P_plenum_CC

        # Return the residuals
        return [residual_thrust, residual_CC_pressure, residual_dP_propellants]

    def get_full_output(self):
        """A function to return the string with data about the cycle."""
        string = \
            (f"\n\n--- INPUT PARAMETERS ---\n"
             f"---Top level parameters---\n"
             f"O/F: {self.OF}   Required thrust: {self.ThrustSea} kN    "
             f"Required CC plenum pressure: {self.P_plenum_CC} bar\n"
             f"---Propellants---\n"
             f"Fuel:"
             f"{self.fuel.CEA_card}     Pressure: {self.fuel.Pt}\n"
             f"Oxidizer:"
             f"{self.oxidizer.CEA_card}     Pressure: {self.oxidizer.Pt}\n"
             f"---Efficiencies---\n"
             f" - OP isotropic efficiency: {self.eta_isotropic_OP}   "
             f" - FP isotropic efficiency: {self.eta_isotropic_FP}\n"
             f" - OT polytropic efficiency: {self.eta_polytropic_OT}\n"
             f" - Catalyst efficiency: {self.eta_catalyst}    "
             f" - C* efficiency: {self.eta_cstar}   "
             f" - Isp efficiency: {self.eta_isp}\n"
             f" - OT pressure recovery factor: {self.Ps_Pt_OT}\n"
             f"---Pressure drop ratios---\n"
             f"Over CC injector: {self.dP_over_Pinj_CC}     "
             f"Over catalyst (bed + injector):{self.dP_over_Pinj_catalyst}\n"
             f"---Other parameters---\n"
             f"CC contraction ratio: {self.CR_CC}   CC expansion ratio: {self.eps_CC}\n"
             f"Film cooling massflow to oxidizer massflow: {self.mdot_film_over_mdot_oxid}\n"
             f"Cooling channels pressure drop: {self.cooling_channels_pressure_drop} bar    "
             f"Cooling channels temperature rise: {self.cooling_channels_temperature_rise} K\n"
             f"Catalyst CR: {self.CR_catalyst}    "
             f"OT axial velocity: {self.axial_velocity_OT} m/s\n\n"
             f"---MASSFLOWS---\n"
             f"Total massflow: {self.mdot_total} kg/s   Oxidizer massflow: {self.CP.mdot_oxidizer} kg/s     "
             f"Fuel massflow: {self.CP.mdot_fuel} kg/s\n"
             f"Film cooling massflow: {self.CP.mdot_film} kg/s\n\n"
             f"---FUEL SIDE----\n"
             f"---Fuel Pump---\n"
             f"FP pressure rise: {self.dP_FP} bar   "
             f"FP temperature rise: {self.CP.pumped_fuel.Ts - self.fuel.Ts} K   "
             f"Pump power: {self.CP.Power_FP * 1e-3} kW\n"
             f"---OXIDIZER SIDE---\n"
             f"---Oxidizer pump---\n"
             f"OP pressure rise: {self.dP_OP} bar   "
             f"OP temperature rise: {self.CP.pumped_oxidizer.Ts - self.oxidizer.Ts} K   "
             f"Pump power: {self.CP.Power_OP * 1e-3} kW\n"
             f"---Cooling channels---\n"
             f"Oxidizer temperature: {self.CP.heated_oxidizer.Ts} K     "
             f"Oxidizer pressure: {self.CP.heated_oxidizer.Pt} bar\n"
             f"---Catalyst---\n"
             f"Products static temperature: {self.CP.catalyst_products.Ts} K     "
             f"Products total temperature: {self.CP.catalyst_products.Tt} K\n"
             f"Pressure at injector: {self.CP.P_inj_catalyst} bar  "
             f"Plenum static pressure: {self.CP.catalyst_products.Ps} bar    "
             f"Plenum total pressure: {self.CP.catalyst_products.Pt} bar\n"
             f"---Oxidizer turbine---\n"
             f"Massflow: {self.CP.mdot_OT} kg/s     Turbine beta_tt: {self.CP.OT_beta_tt}   "
             f"Turbine shaft power: {self.CP.OT_shaft_power * 1e-3} kW\n"
             f"Outlet gas static tempetature: {self.CP.OT_outlet_gas.Ts} K  "
             f"Outlet gas total tempetature: {self.CP.OT_outlet_gas.Tt} K\n"
             f"Outlet gas static pressure: {self.CP.OT_outlet_gas.Ps} bar  "
             f"Outlet gas total pressure: {self.CP.OT_outlet_gas.Pt} bar\n"
             f"Gas static temperature in CC manifold: {self.CP.OT_equilibrium_gas.Ts} K  "
             f"Gas static pressure in CC manifold: {self.CP.OT_equilibrium_gas.Ps} bar\n\n"
             f"---COMBUSTION CHAMBER---\n"
             f"Pressure at injector: {self.CP.P_inj_CC} bar   Plenum pressure: {self.CP.P_plenum_CC} bar "
             f"Combustion temperature: {self.CP.CC_Tcomb} K\n"
             f"Vacuum ISP: {self.CP.IspVac_real} s   Sea ISP: {self.CP.IspSea_real} s\n"
             f"Vacuum thrust: {self.CP.ThrustVac} kN    Sea thrust: {self.CP.ThrustSea} kN\n"
             f"Throat area: {self.CP.A_t_CC} m2    Nozzle exit area:  {self.CP.A_e_CC} m2\n\n"
             f"---CEA OUTPUTS---\n\n"
             f"---Catalyst CEA output---\n"
             f"{self.CP.catalyst_CEA_output}\n\n"
             f"---OT equilibrium gas CEA output---\n"
             f"{self.CP.OT_equilibrium_gas_CEA_output}\n\n"
             f"---CC CEA Output---\n"
             f"{self.CP.CC_CEA_output}"
             )
        return string

    def get_residuals(self):
        """A function to return string about initial and final residuals."""

        string = \
            (f"\n---INITIAL RESIDUALS---\n"
             f"- Difference between actual and desired sea thrust: {self.residuals_0[0]} kN"
             f" (normalized: {self.norm_residuals_0[0]})\n"
             f"- Pressure difference between actual and desired CC plenum pressure: {self.residuals_0[1]} bar"
             f" (normalized: {self.norm_residuals_0[1]})\n"
             f"- Pressure difference between fuel and oxidizer in the CC manifold: {self.residuals_0[2]} bar"
             f" (normalized: {self.norm_residuals_0[2]})\n\n"
             f"\n---REMAINING RESIDUALS---\n"
             f"- Difference between actual and desired sea thrust: {self.residuals[0]} kN"
             f" (normalized: {self.norm_residuals[0]})\n"
             f"- Pressure difference between actual and desired CC plenum pressure: {self.residuals[1]} bar"
             f" (normalized: {self.norm_residuals[1]})\n"
             f"- Pressure difference between fuel and oxidizer in the CC manifold: {self.residuals[2]} bar"
             f" (normalized: {self.norm_residuals[2]})\n\n"
             )
        return string
