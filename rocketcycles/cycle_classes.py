from rocketcycles import cycle_functions
from rocketcycles.fluid import pyfluid_to_rocket_cycle_fluid, RocketCycleFluid
import pyfluids
import scipy.optimize as opt
import numpy as np
import warnings


class Cycle:
    def __init__(self, name=None, OF=None, oxidizer_pyfluid=None, fuel_pyfluid=None, fuel_CEA_name=None,
                 fuel_rocket_cycle_fluid=None, oxidizer_rocket_cycle_fluid=None, oxidizer_CEA_name=None,
                 T_oxidizer=None, T_fuel=None, P_oxidizer=None, P_fuel=None, eta_isotropic_OP=None,
                 eta_isotropic_FP=None, eta_isotropic_BFP=None, eta_polytropic_OT=None, eta_polytropic_FT=None,
                 eta_cstar=None, eta_cf=None, Ps_Pt_OT=None, Ps_Pt_FT=None, dP_over_Pinj_CC=None, dP_over_Pinj_OPB=None,
                 dP_over_Pinj_FPB=None, CR_FPB=None, CR_OPB=None, CR_catalyst=None, CR_CC=None, eps_CC=None,
                 mdot_film_over_mdot_oxid=None, mdot_film_over_mdot_fuel=None, dP_cooling_channels=None,
                 dP_over_Pinj_catalyst=None, dT_cooling_channels=None, axial_velocity_OT=None,
                 axial_velocity_FT=None):
        """A parent class to store shared attributes and methods for all Cycle-derives classes"""

        # Propellants
        self.fuel_pyfluid = fuel_pyfluid
        self.fuel_CEA_name = fuel_CEA_name
        self.fuel_rocket_cycle_fluid = fuel_rocket_cycle_fluid
        self.oxidier_rocket_cycle_fluid = oxidizer_rocket_cycle_fluid
        self.oxidizer_pyfluid = oxidizer_pyfluid
        self.oxidizer_CEA_name = oxidizer_CEA_name

        # Massflows
        self.mdot_total = None
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
        self.mdot_film_over_mdot_fuel = mdot_film_over_mdot_fuel
        self.mdot_film_over_mdot_oxid = mdot_film_over_mdot_oxid

        # OF ratios
        self.OF = OF
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
        self.CC_with_film_CEA_output = None

        # Pressures
        self.P_inj_FPB = None
        self.P_inj_OPB = None
        self.P_inj_catalyst = None
        self.P_inj_CC = None
        self.P_plenum_CC = None
        self.P_plenum_CC_required = None
        self.dP_OP = None
        self.dP_FP = None
        self.dP_BFP = None
        self.P_oxidizer = P_oxidizer
        self.P_fuel = P_fuel
        self.dP_over_Pinj_CC = dP_over_Pinj_CC
        self.dP_over_Pinj_OPB = dP_over_Pinj_OPB
        self.dP_over_Pinj_FPB = dP_over_Pinj_FPB
        self.dP_cooling_channels = dP_cooling_channels
        self.dP_over_Pinj_catalyst = dP_over_Pinj_catalyst

        # Pressure ratios
        self.FT_beta_tt = None
        self.OT_beta_tt = None

        # CC parameters
        self.IspVac_real = None
        self.IspSea_real = None
        self.ThrustVac = None
        self.ThrustSea = None
        self.Thrust_required = None
        self.A_t_CC = None
        self.A_e_CC = None

        # Temperatures
        self.dT_OP = None
        self.dT_FP = None
        self.dT_BFP = None
        self.CC_Tcomb = None
        self.T_FPB = None
        self.T_OPB = None
        self.T_FPB_required = None
        self.T_OPB_required = None
        self.T_oxidizer = T_oxidizer
        self.T_fuel = T_fuel
        self.dT_cooling_channels = dT_cooling_channels

        # Efficiencies
        self.eta_isotropic_OP = eta_isotropic_OP
        self.eta_isotropic_FP = eta_isotropic_FP
        self.eta_isotropic_BFP = eta_isotropic_BFP
        self.eta_polytropic_OT = eta_polytropic_OT
        self.eta_polytropic_FT = eta_polytropic_FT
        self.eta_cstar = eta_cstar
        self.eta_cf = eta_cf
        self.Ps_Pt_OT = Ps_Pt_OT
        self.Ps_Pt_FT = Ps_Pt_FT

        # Other
        self.name = name
        self.OT_molar_Cp_average = None
        self.OT_molar_Cp_average = None
        self.FT_gamma_average = None
        self.FT_gamma_average = None
        self.sea_level_nozzle_operation_mode = None
        self.axial_velocity_OT = axial_velocity_OT
        self.axial_velocity_FT = axial_velocity_FT
        self.CR_FPB = CR_FPB
        self.CR_catalyst = CR_catalyst
        self.CR_OPB = CR_OPB
        self.CR_CC = CR_CC
        self.eps_CC = eps_CC

    def get_full_output(self, get_CEA_inputs=False):
        """A function to return the string with data about the cycle.

        :param boolean get_CEA_inputs: True or False statement whether to print CEA outputs for parts of the cycle
            where it was used
        """

        string = \
            (f"\n\n--- INPUT PARAMETERS ---\n"
             f"---Top level parameters---\n"
             f"O/F: {self.OF}   Required thrust: {self.Thrust_required} kN    "
             f"Required CC plenum pressure: {self.P_plenum_CC_required} bar\n"
             f"---Propellants---\n"
             f"Fuel:"
             f"{self.fuel_CEA_name}   Temperature: {self.T_fuel} K   Pressure: {self.P_fuel}\n"
             f"Oxidizer:"
             f"{self.oxidizer_CEA_name}   Temperature: {self.T_oxidizer} K   Pressure: {self.P_oxidizer}\n"
             f"---Efficiencies---\n"
             f" - C* efficiency: {self.eta_cstar}   "
             f" - Cf efficiency: {self.eta_cf}\n"
             f" - OP isotropic efficiency: {self.eta_isotropic_OP}   "
             f" - FP isotropic efficiency: {self.eta_isotropic_FP}\n")
        if self.name == "FFSC":
            string += (
                f" - OT polytropic efficiency: {self.eta_polytropic_OT}  "
                f" - FT polytropic efficiency: {self.eta_polytropic_FT}\n"
                f" - OT pressure recovery factor: {self.Ps_Pt_OT}"
                f" - FT pressure recovery factor: {self.Ps_Pt_FT}\n")
        elif self.name == "ORSC":
            string += (
                f" - BFP isotropic efficiency: {self.eta_isotropic_BFP}\n"
                f" - OT polytropic efficiency: {self.eta_polytropic_OT}  "
                f" - OT pressure recovery factor: {self.Ps_Pt_OT}\n")
        elif self.name == "CC":
            string += (
                f" - OT polytropic efficiency: {self.eta_polytropic_OT}"
                f" - OT pressure recovery factor: {self.Ps_Pt_OT}\n")
        string += (
            f"---Pressure drop ratios---\n"
            f"Over CC injector: {self.dP_over_Pinj_CC}\n")
        if self.name == "FFSC":
            string += (
                f"Over OPB injector:{self.dP_over_Pinj_OPB}       "
                f"Over FPB injector:{self.dP_over_Pinj_FPB}\n")
        elif self.name == "ORSC":
            string += (
                f"Over OPB injector:{self.dP_over_Pinj_OPB}\n")
        elif self.name == "CC":
            string += (
                f"Over catalyst (bed + injector):{self.dP_over_Pinj_catalyst}\n")
        string += (
            f"---Other parameters---\n"
            f"CC contraction ratio: {self.CR_CC}   CC expansion ratio: {self.eps_CC}\n")
        if self.name == "FFSC":
            string += (
                f"Required OPB temperature: {self.T_OPB_required} K    "
                f"Required FPB temperature: {self.T_FPB_required} K\n"
                f"OPB CR: {self.CR_OPB}    FPB CR: {self.CR_FPB} \n"
                f"OT axial velocity: {self.axial_velocity_OT} m/s      "
                f"FT axial velocity: {self.axial_velocity_FT} m/s\n"
                f"Film cooling massflow to fuel massflow: {self.mdot_film_over_mdot_fuel}\n")
        if self.name == "ORSC":
            string += (
                f"Required OPB temperature: {self.T_OPB_required} K     OPB CR: {self.CR_OPB}\n"
                f"OT axial velocity: {self.axial_velocity_OT} m/s\n"
                f"Film cooling massflow to fuel massflow: {self.mdot_film_over_mdot_fuel}\n")
        elif self.name == "CC":
            string += (
                f"Catalyst CR: {self.CR_catalyst}    "
                f"OT axial velocity: {self.axial_velocity_OT} m/s\n"
                f"Film cooling massflow to fuel massflow: {self.mdot_film_over_mdot_oxid}\n")
        string += (
            f"Cooling channels pressure drop: {self.dP_cooling_channels} bar    "
            f"Cooling channels temperature rise: {self.dT_cooling_channels} K\n\n"
            f"---MASSFLOWS---\n"
            f"Total massflow: {self.mdot_total:.5g} kg/s   Oxidizer massflow: {self.mdot_oxidizer:.5g} kg/s     "
            f"Fuel massflow: {self.mdot_fuel:.5g} kg/s\n")
        if self.name == "FFSC":
            string += (
                f"Oxidizer crossflow massflow: {self.mdot_crossflow_oxidizer:.5g} kg/s  "
                f"Fuel crossflow massflow: {self.mdot_crossflow_fuel:.5g} kg/s  ")
        elif self.name == "ORSC":
            string += (
                f"Fuel crossflow massflow: {self.mdot_crossflow_fuel:.5g} kg/s  ")
        string += (
            f"Film cooling massflow: {self.mdot_film:.5g} kg/s\n\n"
            f"---FUEL SIDE----\n"
            f"---Fuel Pump---\n"
            f"FP pressure rise: {self.dP_FP:.5g} bar   "
            f"FP temperature rise: {self.dT_FP:.5g} K   "
            f"Pump power: {self.Power_FP * 1e-3:.5g} kW\n")
        if self.name == "FFSC":
            string += (
                f"---Cooling channels---\n"
                f"Fuel temperature: {self.heated_fuel.Ts:.5g} K     Fuel pressure: {self.heated_fuel.Ps:.5g} bar\n"
                f"---Fuel preburner---\n"
                f"Fuel massflow: {self.mdot_f_FPB:.5g} kg/s     Preburner OF: {self.OF_FPB:.5g}\n"
                f"Products static temperature: {self.FPB_products.Ts:.5g} K     "
                f"Products total temperature: {self.FPB_products.Tt:.5g} K\n"
                f"Pressure at injector: {self.P_inj_FPB:.5g} bar  "
                f"Plenum static pressure: {self.FPB_products.Ps:.5g} bar    "
                f"Plenum total pressure: {self.FPB_products.Pt:.5g} bar\n"
                f"---Fuel turbine---\n"
                f"Massflow: {self.mdot_FT:.5g} kg/s     Turbine beta_tt: {self.FT_beta_tt:.5g}\n"
                f"Outlet gas static tempetature: {self.FT_outlet_gas.Ts:.5g} K  "
                f"Outlet gas total tempetature: {self.FT_outlet_gas.Tt:.5g} K\n"
                f"Outlet gas static pressure: {self.FT_outlet_gas.Ps:.5g} bar  "
                f"Outlet gas total pressure: {self.FT_outlet_gas.Pt:.5g} bar\n"
                f"Gas static temperature in CC manifold: {self.FT_equilibrium_gas.Ts:.5g} K  "
                f"Gas static pressure in CC manifold: {self.FT_equilibrium_gas.Ps:.5g} bar\n\n")
        elif self.name == "ORSC":
            if self.dP_BFP > 0:
                string += (
                    f"---Booster Fuel Pump---\n"
                    f"BFP pressure rise: {self.dP_BFP:.5g} bar   "
                    f"BFP temperature rise: {self.dT_BFP:.5g} K   "
                    f"Booster pump power: {self.Power_BFP * 1e-3:.5g} kW\n")
            string += (
                f"---Cooling channels---\n"
                f"Fuel temperature: {self.heated_fuel.Ts:.5g} K     Fuel pressure: {self.heated_fuel.Ps:.5g} bar\n\n")
        string += (
            f"---OXIDIZER SIDE---\n"
            f"---Oxidizer pump---\n"
            f"OP pressure rise: {self.dP_OP:.5g} bar   "
            f"OP temperature rise: {self.dT_OP:.5g} K   "
            f"Pump power: {self.Power_OP * 1e-3:.5g} kW\n")
        if self.name == "CC":
            string += (
                f"---Cooling channels---\n"
                f"Oxidizer temperature: {self.heated_oxidizer.Ts:.5g} K     "
                f"Oxidizer pressure: {self.heated_oxidizer.Pt:.5g} bar\n"
                f"---Catalyst---\n"
                f"Products static temperature: {self.catalyst_products.Ts:.5g} K     "
                f"Products total temperature: {self.catalyst_products.Tt:.5g} K\n"
                f"Pressure at injector: {self.P_inj_catalyst:.5g} bar  "
                f"Plenum static pressure: {self.catalyst_products.Ps:.5g} bar    "
                f"Plenum total pressure: {self.catalyst_products.Pt:.5g} bar\n")
        elif self.name == "FFSC" or self.name == "ORSC":
            string += (
                f"---Oxidizer preburner---\n"
                f"Oxidizer massflow: {self.mdot_ox_OPB:.5g} kg/s     Preburner OF: {self.OF_OPB:.5g}\n"
                f"Products static temperature: {self.OPB_products.Ts:.5g} K     "
                f"Products total temperature: {self.OPB_products.Tt:.5g} K\n"
                f"Pressure at injector: {self.P_inj_OPB:.5g} bar  "
                f"Plenum static pressure: {self.OPB_products.Ps:.5g} bar    "
                f"Plenum total pressure: {self.OPB_products.Pt:.5g} bar\n")
        string += (
            f"---Oxidizer turbine---\n"
            f"Massflow: {self.mdot_OT:.5g} kg/s     Turbine beta_tt: {self.OT_beta_tt:.5g}\n"
            f"Outlet gas static tempetature: {self.OT_outlet_gas.Ts:.5g} K  "
            f"Outlet gas total tempetature: {self.OT_outlet_gas.Tt:.5g} K\n"
            f"Outlet gas static pressure: {self.OT_outlet_gas.Ps:.5g} bar  "
            f"Outlet gas total pressure: {self.OT_outlet_gas.Pt:.5g} bar\n"
            f"Gas static temperature in CC manifold: {self.OT_equilibrium_gas.Ts:.5g} K  "
            f"Gas static pressure in CC manifold: {self.OT_equilibrium_gas.Ps:.5g} bar\n\n"
            f"---COMBUSTION CHAMBER---\n"
            f"Pressure at injector: {self.P_inj_CC:.5g} bar   Plenum pressure: {self.P_plenum_CC:.5g} bar   "
            f"Combustion temperature: {self.CC_Tcomb:.5g} K\n"
            f"Vacuum ISP: {self.IspVac_real:.5g} s   Sea ISP: {self.IspSea_real:.5g} s\n"
            f"Vacuum thrust: {self.ThrustVac:.5g} kN    Sea thrust: {self.ThrustSea:.5g} kN\n"
            f"Throat area: {self.A_t_CC:.5g} m2    Nozzle exit area:  {self.A_e_CC:.5g} m2\n\n")
        if get_CEA_inputs:
            if self.name == "FFSC":
                string += (
                    f"---CEA OUTPUTS---\n\n"
                    f"---FPB CEA output---\n"
                    f"{self.FPB_CEA_output}\n\n"
                    f"---FT equilibrium gas CEA output---\n"
                    f"{self.FT_equilibrium_gas_CEA_output}\n\n"
                    f"---OPB CEA output---\n"
                    f"{self.OPB_CEA_output}\n\n"
                    f"---OT equilibrium gas CEA output---\n"
                    f"{self.OT_equilibrium_gas_CEA_output}\n\n")
            elif self.name == "ORSC":
                string += (
                    f"---CEA OUTPUTS---\n\n"
                    f"---OPB CEA output---\n"
                    f"{self.OPB_CEA_output}\n\n"
                    f"---OT equilibrium gas CEA output---\n"
                    f"{self.OT_equilibrium_gas_CEA_output}\n\n")
            elif self.name == "CC":
                string += (
                    f"---CEA OUTPUTS---\n\n"
                    f"---Catalyst CEA output---\n"
                    f"{self.catalyst_CEA_output}\n\n"
                    f"---OT equilibrium gas CEA output---\n"
                    f"{self.OT_equilibrium_gas_CEA_output}\n\n")
            string += (
                f"---CC CEA Output---\n"
                f"{self.CC_CEA_output}\n\n")
            if self.CC_with_film_CEA_output is not None:
                string += (
                    f"---CC with film coolant CEA Output---\n"
                    f"{self.CC_with_film_CEA_output}\n\n")
        return string


class FFSC_LRE(Cycle):
    def __init__(self, OF, oxidizer_pyfluid, fuel_pyfluid, fuel_CEA_name, oxidizer_CEA_name, T_oxidizer, T_fuel,
                 P_oxidizer, P_fuel, eta_isotropic_OP, eta_isotropic_FP, eta_polytropic_OT, eta_polytropic_FT,
                 eta_cstar, eta_cf, Ps_Pt_OT, Ps_Pt_FT, dP_over_Pinj_CC, dP_over_Pinj_OPB, dP_over_Pinj_FPB, CR_FPB,
                 CR_OPB, CR_CC, eps_CC, mdot_film_over_mdot_fuel, dP_cooling_channels,
                 dT_cooling_channels, axial_velocity_OT, axial_velocity_FT):
        """A class to analyse and size full flow staged combustion cycle.

        :param float or int OF: Oxidizer-to-Fuel ratio of the cycle
        :param pyfluids.FluidsList oxidizer_pyfluid: PyFluids FluidsList object representing oxidizer
        :param pyfluids.FluidsList fuel_pyfluid: PyFluids FluidsList object representing fuel
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
        :param float or int eta_cstar: C* efficiency of the CC (-)
        :param float or int eta_cf: Cf efficiency of the CC (-) at the sea level
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
        :param float or int dP_cooling_channels: Pressure drop in the cooling channels (bar)
        :param float or int dT_cooling_channels: Temperature rise in the cooling channels (K)
        :param float or int axial_velocity_OT: Axial velocity across oxidizer turbine (m/s)
        :param float or int axial_velocity_FT: Axial velocity across fuel turbine (m/s)
        """

        # Parent class call
        super().__init__(name="FFSC", OF=OF, oxidizer_pyfluid=oxidizer_pyfluid, fuel_pyfluid=fuel_pyfluid,
                         fuel_CEA_name=fuel_CEA_name, oxidizer_CEA_name=oxidizer_CEA_name, T_oxidizer=T_oxidizer,
                         T_fuel=T_fuel, P_oxidizer=P_oxidizer, P_fuel=P_fuel, eta_isotropic_OP=eta_isotropic_OP,
                         eta_isotropic_FP=eta_isotropic_FP, eta_polytropic_OT=eta_polytropic_OT,
                         eta_polytropic_FT=eta_polytropic_FT, eta_cstar=eta_cstar, eta_cf=eta_cf, Ps_Pt_OT=Ps_Pt_OT,
                         Ps_Pt_FT=Ps_Pt_FT, dP_over_Pinj_CC=dP_over_Pinj_CC, dP_over_Pinj_OPB=dP_over_Pinj_OPB,
                         dP_over_Pinj_FPB=dP_over_Pinj_FPB, CR_FPB=CR_FPB, CR_OPB=CR_OPB, CR_CC=CR_CC, eps_CC=eps_CC,
                         mdot_film_over_mdot_fuel=mdot_film_over_mdot_fuel,
                         dP_cooling_channels=dP_cooling_channels,
                         dT_cooling_channels=dT_cooling_channels,
                         axial_velocity_OT=axial_velocity_OT, axial_velocity_FT=axial_velocity_FT)

    def analyze_cycle(self, mdot_total, mdot_crossflow_fuel_over_mdot_fuel, dP_FP, mdot_crossflow_ox_over_mdot_ox=None,
                      dP_OP=None, T_FPB_required=None, mdot_crossflow_ox_over_mdot_ox_bracket=(0.04, 0.11),
                      mdot_crossflow_oxidizer_xtol=1e-3, mdot_crossflow_oxidizer_maxiter=1000):
        """A function to analyze the cycle. If all arguments are given, "analysis" mode is set. If only
            the first three are given, "sizing" mode is set and other two parameters will be calculated inside the
             function itself. Oxidizer crossflow massflow will be calculated based on required FPB temperature,
              while oxidizer pressure pump will be determined from minimum pressure required to pump oxidizer into FPB.

        :param int or float mdot_total: Total propellant massflow (kg/s)
        :param int or float mdot_crossflow_ox_over_mdot_ox: Oxidizer crossflow massflow to oxidizer massflow (-)
        :param int or float mdot_crossflow_fuel_over_mdot_fuel: Fuel crossflow massflow to fuel massflow (-)
        :param int or float dP_FP: Pressure rise (bar) in the fuel pump.
        :param int or float dP_OP: Pressure rise (bar) in the oxidizer pump.
        :param int or float T_FPB_required: Required fuel preburner temperature (K) when "sizing" mode is used.
        :param tuple mdot_crossflow_ox_over_mdot_ox_bracket: tuple of floats corresponding to the bracket in which to
            search for mdot_crossflow_ox_over_mdot_ox in "sizing" mode. By default, set to 0.04 (-) and 0.11 (-),
             which should correspond to any reasonable preburner temperature.
        :param float mdot_crossflow_oxidizer_xtol: Absolute convergence tolerance (kg/s) on
            found mdot_crossflow_oxidizer in "sizing" mode. By default, set to 1e-3.
        :param int mdot_crossflow_oxidizer_maxiter: Maximum number of iterations for calculating
            mdot_crossflow_oxidizer in "sizing" mode. By default, 1000.
        """

        # First assign class attributes
        self.mdot_total = mdot_total
        self.dP_FP = dP_FP
        self.T_FPB_required = T_FPB_required
        # Set the modes
        if mdot_crossflow_ox_over_mdot_ox is not None and dP_OP is not None:
            mode = "analysis"
            self.dP_OP = dP_OP
        elif mdot_crossflow_ox_over_mdot_ox is None and dP_OP is None:
            mode = "sizing"
            if T_FPB_required is None:
                warnings.simplefilter("error", UserWarning)
                warnings.warn("T_FPB_required cannot be None in sizing mode")
        else:
            warnings.simplefilter("error", UserWarning)
            warnings.warn("mdot_crossflow_ox_over_mdot_ox and dP_OP must be both either None or float/int")

        # Now calculate oxidizer, fuel, film crossflow massflows
        self.mdot_fuel = mdot_total / (1 + self.OF)
        self.mdot_oxidizer = self.OF * self.mdot_fuel
        self.mdot_film = self.mdot_fuel * self.mdot_film_over_mdot_fuel
        self.mdot_crossflow_fuel = mdot_crossflow_fuel_over_mdot_fuel * self.mdot_fuel
        # If mode is analysis, oxidizer crossflow is already known. Otherwise, it will be calculated based on FPB
        # temperature later on.
        if mode == "analysis":
            self.mdot_crossflow_oxidizer = mdot_crossflow_ox_over_mdot_ox * self.mdot_oxidizer

        # Create Pyfluids Fluid objects for propellants with correct units (Pa and deg C)
        self.fuel = pyfluids.Fluid(self.fuel_pyfluid).with_state(
            pyfluids.Input.pressure(self.P_fuel * 1e5), pyfluids.Input.temperature(self.T_fuel - 273.15))
        self.oxidizer = pyfluids.Fluid(self.oxidizer_pyfluid).with_state(
            pyfluids.Input.pressure(self.P_oxidizer * 1e5), pyfluids.Input.temperature(self.T_oxidizer - 273.15))

        # Go over fuel side of the system. First calculate states after fuel pump and power required to drive it.
        self.pumped_fuel, self.w_pumped_fuel = cycle_functions.calculate_state_after_pump_for_pyfluids(
            fluid=self.fuel, delta_P=dP_FP, efficiency=self.eta_isotropic_FP)
        self.Power_FP = self.w_pumped_fuel * self.mdot_fuel

        # Calculate state after cooling channels and change both heated and pumped
        # fuel into RocketCycleFluid object. Pumped fuel will be still used later on for oxygen preburner,
        # hence it is changed to RocketCycleFluid.
        self.heated_fuel, self.mdot_cooling_channels_outlet = (
            cycle_functions.calculate_state_after_cooling_channels_for_Pyfluids(
                fluid=self.pumped_fuel, mdot_coolant=self.mdot_fuel, mdot_film=self.mdot_film,
                pressure_drop=self.dP_cooling_channels,
                temperature_rise=self.dT_cooling_channels))
        # Phase below is given as "liquid", so that total and static pressure are assumed to be about equal after the
        # cooling channels, but it will also work when there are two-phases with large vapor content or the fluid is
        # supercritical due to pressure recovery in the manifold afterward
        self.heated_fuel = \
            pyfluid_to_rocket_cycle_fluid(fluid=self.heated_fuel, CEA_name=self.fuel_CEA_name, type="fuel",
                                          phase="liquid")
        self.pumped_fuel = \
            pyfluid_to_rocket_cycle_fluid(fluid=self.pumped_fuel, CEA_name=self.fuel_CEA_name, type="fuel",
                                          phase="liquid")
        # Get temperature rise across the pump
        self.dT_FP = self.pumped_fuel.Ts - self.T_fuel

        # Now perform calculations related to fuel preburner. First calculate its pressure based on fuel total pressure.
        self.P_inj_FPB = self.heated_fuel.Pt / (1 + self.dP_over_Pinj_FPB)

        # Before we calculate state after fuel preburner, we need to get our pumped oxidizer. For sizing mode, oxidizer
        # pump outlet pressure must simply match fuel pressure after cooling channels, so that both can inject
        # propellants into FPB at the same pressure ratio. Alternatively, if mode is analysis, dP_OP is already known.
        # We can already calculate power required to drive oxidizer pump as well.
        # Get pressure rise if mode is sizing:
        if mode == "sizing":
            self.dP_OP = self.heated_fuel.Pt - self.P_oxidizer
        # Get state after oxidizer pump and power required to drive it
        self.pumped_oxidizer, self.w_pumped_oxidizer = cycle_functions.calculate_state_after_pump_for_pyfluids(
            fluid=self.oxidizer, delta_P=self.dP_OP, efficiency=self.eta_isotropic_OP)
        self.Power_OP = self.w_pumped_oxidizer * self.mdot_oxidizer
        # Change oxidizer into RocketCycleFluid object.
        self.pumped_oxidizer = \
            pyfluid_to_rocket_cycle_fluid(fluid=self.pumped_oxidizer, CEA_name=self.oxidizer_CEA_name, type="oxidizer",
                                          phase="liquid")
        # Calculate temperature rise
        self.dT_OP = self.pumped_oxidizer.Ts - self.T_oxidizer

        # Fuel massflow into FPB needs to be calculated too:
        self.mdot_f_FPB = self.mdot_cooling_channels_outlet - self.mdot_crossflow_fuel

        # For sizing mode, oxidizer massflow that will allow to achieve certain FPB temperature needs to be found.
        # For analysis, it is already known.
        # Find oxidizer massflow if sizing mode is used:
        if mode == "sizing":
            # Oxidizer massflow that will allow to achieve certain FPB temperature can be found using bisection
            # algorithm. First define a function for it to solve.
            def calculate_FPB_temperature_residual(mdot_ox):
                # Get its OF ratio.
                OF_FPB = mdot_ox / self.mdot_f_FPB
                # Get results for FPB.
                FPB_CEA_output, FPB_products = \
                    cycle_functions.calculate_state_after_preburner(OF=OF_FPB, preburner_inj_pressure=self.P_inj_FPB,
                                                                    CR=self.CR_FPB, fuel=self.heated_fuel,
                                                                    oxidizer=self.pumped_oxidizer)
                # Return residual. Total temperature is used, since this is what will be next to the walls due to gas
                # slowing down due to boundary layer.
                return FPB_products.Tt - self.T_FPB_required

            # Now solve the function to find the massflow that satisfies requirements.
            self.mdot_crossflow_oxidizer = \
                opt.toms748(calculate_FPB_temperature_residual,
                            a=mdot_crossflow_ox_over_mdot_ox_bracket[0] * self.mdot_oxidizer,
                            b=mdot_crossflow_ox_over_mdot_ox_bracket[1] * self.mdot_oxidizer,
                            maxiter=mdot_crossflow_oxidizer_maxiter, xtol=mdot_crossflow_oxidizer_xtol)
        # Now get FPB OF ratio.
        self.OF_FPB = self.mdot_crossflow_oxidizer / self.mdot_f_FPB
        # And get results for FPB.
        self.FPB_CEA_output, self.FPB_products = \
            cycle_functions.calculate_state_after_preburner(OF=self.OF_FPB, preburner_inj_pressure=self.P_inj_FPB,
                                                            CR=self.CR_FPB, fuel=self.heated_fuel,
                                                            oxidizer=self.pumped_oxidizer)

        # Calculate state after fuel turbine
        self.mdot_FT = self.mdot_f_FPB + self.mdot_crossflow_oxidizer
        (self.FT_beta_tt, self.FT_outlet_gas, self.FT_equilibrium_gas, self.FT_equilibrium_gas_CEA_output,
         self.FT_molar_Cp_average, self.FT_gamma_average) = \
            cycle_functions.calculate_state_after_turbine(massflow=self.mdot_FT, turbine_power=self.Power_FP,
                                                          turbine_polytropic_efficiency=self.eta_polytropic_FT,
                                                          preburner_products=self.FPB_products,
                                                          turbine_axial_velocity=self.axial_velocity_FT,
                                                          pressure_recovery_factor=self.Ps_Pt_FT)

        # Now go over oxidizer side of the system. Calculate state after oxygen preburner. Again first calculate
        # oxidizer massflow through it and preburner OF.
        self.mdot_ox_OPB = self.mdot_oxidizer - self.mdot_crossflow_oxidizer
        self.OF_OPB = self.mdot_ox_OPB / self.mdot_crossflow_fuel
        # For determining preburner pressure, use minimum propellant pressure
        self.P_inj_OPB = min(self.pumped_oxidizer.Pt, self.pumped_fuel.Pt) / (1 + self.dP_over_Pinj_OPB)
        # Get preburner results
        self.OPB_CEA_output, self.OPB_products = \
            cycle_functions.calculate_state_after_preburner(OF=self.OF_OPB, preburner_inj_pressure=self.P_inj_OPB,
                                                            CR=self.CR_OPB, fuel=self.pumped_fuel,
                                                            oxidizer=self.pumped_oxidizer)

        # Calculate state after oxygen turbine.
        self.mdot_OT = self.mdot_ox_OPB + self.mdot_crossflow_fuel
        (self.OT_beta_tt, self.OT_outlet_gas, self.OT_equilibrium_gas, self.OT_equilibrium_gas_CEA_output,
         self.OT_molar_Cp_average, self.OT_gamma_average) = \
            cycle_functions.calculate_state_after_turbine(massflow=self.mdot_OT, turbine_power=self.Power_OP,
                                                          turbine_polytropic_efficiency=self.eta_polytropic_OT,
                                                          preburner_products=self.OPB_products,
                                                          turbine_axial_velocity=self.axial_velocity_OT,
                                                          pressure_recovery_factor=self.Ps_Pt_OT)

        # Calculate combustion chamber performance. CC pressure at injector is determined wrt to minimum propellant
        # pressure. Total pressure is used because the gas should slow down in the turbine outlet manifold.
        self.P_inj_CC = min(self.FT_equilibrium_gas.Ps, self.OT_equilibrium_gas.Ps) / (1 + self.dP_over_Pinj_CC)
        (self.CC_CEA_output, self.CC_with_film_CEA_output, self.P_plenum_CC, self.IspVac_real, self.IspSea_real,
         self.CC_Tcomb, self.ThrustVac, self.ThrustSea, self.A_t_CC, self.A_e_CC,
         self.sea_level_nozzle_operation_mode) = \
            cycle_functions.calculate_combustion_chamber_performance(
                mdot_oxidizer=self.mdot_OT, mdot_fuel=self.mdot_FT, oxidizer=self.OT_equilibrium_gas,
                fuel=self.FT_equilibrium_gas, CC_pressure_at_injector=self.P_inj_CC, CR=self.CR_CC, eps=self.eps_CC,
                eta_cstar=self.eta_cstar, eta_cf=self.eta_cf, mdot_film=self.mdot_film,
                coolant_CEA_card=self.pumped_fuel.CEA_card)


class ORSC_LRE(Cycle):
    def __init__(self, OF, oxidizer_pyfluid, fuel_pyfluid, fuel_CEA_name, oxidizer_CEA_name, T_oxidizer, T_fuel,
                 P_oxidizer, P_fuel, eta_isotropic_OP, eta_isotropic_FP, eta_isotropic_BFP, eta_polytropic_OT,
                 eta_cstar, eta_cf, Ps_Pt_OT, dP_over_Pinj_CC, dP_over_Pinj_OPB, CR_OPB, CR_CC, eps_CC,
                 mdot_film_over_mdot_fuel, dP_cooling_channels, dT_cooling_channels,
                 axial_velocity_OT):
        """A class to analyse and size oxygen rich staged combustion cycle.

        :param float or int OF: Oxidizer-to-Fuel ratio of the cycle
        :param pyfluids.FluidsList oxidizer_pyfluid: PyFluids object representing oxidizer
        :param pyfluids.FluidsList fuel_pyfluid: PyFluids object representing fuel
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
        :param float or int eta_cstar: C* efficiency of the CC (-)
        :param float or int eta_cf: Cf efficiency of the CC (-) at the sea level
        :param float Ps_Pt_OT: Pressure recovery factor for oxidizer turbine. It is a ratio of static to total pressure
         recovered in diffuser and manifold after turbine (-)
        :param float or int dP_over_Pinj_OPB: Pressure drop ratio for the oxidizer preburner (-)
        :param float or int dP_over_Pinj_CC: Pressure drop ratio for the combustion chamber (-)
        :param float or int CR_OPB: Contraction ratio of the oxidizer preburner (-), used as a measure of its size
        (as preburner does not usually have a sonic throat)
        :param float or int CR_CC: Contraction ratio of the combustion chamber (-)
        :param float or int eps_CC: Expansion ratio of the combustion chamber (-)
        :param float or int mdot_film_over_mdot_fuel: Ratio of the fuel film cooling massflow to total fuel massflow (-)
        :param float or int dP_cooling_channels: Pressure drop in the cooling channels (bar)
        :param float or int dT_cooling_channels: Temperature rise in the cooling channels (K)
        :param float or int axial_velocity_OT: Axial velocity across oxidizer turbine (m/s)
        """

        # Parent class call
        super().__init__(name="ORSC", OF=OF, oxidizer_pyfluid=oxidizer_pyfluid, fuel_pyfluid=fuel_pyfluid,
                         fuel_CEA_name=fuel_CEA_name, oxidizer_CEA_name=oxidizer_CEA_name, T_oxidizer=T_oxidizer,
                         T_fuel=T_fuel, P_oxidizer=P_oxidizer, P_fuel=P_fuel, eta_isotropic_OP=eta_isotropic_OP,
                         eta_isotropic_BFP=eta_isotropic_BFP, eta_isotropic_FP=eta_isotropic_FP,
                         eta_polytropic_OT=eta_polytropic_OT, eta_cstar=eta_cstar, eta_cf=eta_cf, Ps_Pt_OT=Ps_Pt_OT,
                         dP_over_Pinj_CC=dP_over_Pinj_CC, dP_over_Pinj_OPB=dP_over_Pinj_OPB,
                         CR_OPB=CR_OPB, CR_CC=CR_CC, eps_CC=eps_CC, mdot_film_over_mdot_fuel=mdot_film_over_mdot_fuel,
                         dP_cooling_channels=dP_cooling_channels,
                         dT_cooling_channels=dT_cooling_channels,
                         axial_velocity_OT=axial_velocity_OT)

    def analyze_cycle(self, mdot_total, dP_OP, dP_FP, mdot_crossflow_fuel_over_mdot_fuel=None, T_OPB_required=None,
                      mdot_crossflow_fuel_over_mdot_fuel_bracket=(0.04, 0.11), mdot_crossflow_fuel_xtol=1e-3,
                      mdot_crossflow_oxidizer_maxiter=1000):
        """A function to analyze the cycle. If all the first four arguments are given, the mode is set to "analysis"
            and the algorithm will simply analyze the cycle for the given arguments. If
             mdot_crossflow_fuel_over_mdot_fuel is omitted, the mode is "sizing" and fuel crossflow massflow will be
              calculated to achieve given T_OPB_required.

        :param int or float mdot_total: Total massflow (kg/s)
        :param int or float dP_FP: Pressure rise (bar) in the fuel pump.
        :param int or float dP_OP: Pressure rise (bar) in the oxidizer pump.
        :param int or float mdot_crossflow_fuel_over_mdot_fuel: Fuel crossflow massflow to fuel massflow (-)
        :param float or int T_OPB_required: Required oxidizer preburner temperature (K) if "sizing" mode is used.
        :param tuple mdot_crossflow_fuel_over_mdot_fuel_bracket: The tuple of floats corresponding to the bracket in
            which to search for mdot_crossflow_fuel_over_mdot_fuel. By default, set to 0.04 (-) and 0.10 (-), which
             should correspond to any reasonable preburner temperature.
        :param float mdot_crossflow_fuel_xtol: Absolute convergence tolerance (kg/s) on
            found mdot_crossflow_fuel in "sizing" mode. By default, set to 1e-3.
        :param inter mdot_crossflow_oxidizer_maxiter: Maximum number of iterations for calculating
            mdot_crossflow_fuel in "sizing" mode. By default, 1000.
        """

        # First assign class attributes
        self.mdot_total = mdot_total
        self.dP_OP = dP_OP
        self.dP_FP = dP_FP
        self.T_OPB_required = T_OPB_required
        # Set the modes
        if mdot_crossflow_fuel_over_mdot_fuel is not None:
            mode = "analysis"
        elif mdot_crossflow_fuel_over_mdot_fuel is None:
            mode = "sizing"
            if T_OPB_required is None:
                warnings.simplefilter("error", UserWarning)
                warnings.warn("T_OPB_required cannot be None in sizing mode")

        # Now calculate oxidizer, fuel, film crossflow massflows
        self.mdot_fuel = mdot_total / (1 + self.OF)
        self.mdot_oxidizer = self.OF * self.mdot_fuel
        self.mdot_film = self.mdot_fuel * self.mdot_film_over_mdot_fuel
        # If mode is analysis, oxidizer crossflow is already known. Otherwise, it will be calculated based on FPB
        # temperature later on.
        if mode == "analysis":
            self.mdot_crossflow_fuel = mdot_crossflow_fuel_over_mdot_fuel * self.mdot_fuel

        # Create Pyfluids Fluid objects for propellants with correct units (Pa and deg C)
        self.fuel = pyfluids.Fluid(self.fuel_pyfluid).with_state(
            pyfluids.Input.pressure(self.P_fuel * 1e5), pyfluids.Input.temperature(self.T_fuel - 273.15))
        self.oxidizer = pyfluids.Fluid(self.oxidizer_pyfluid).with_state(
            pyfluids.Input.pressure(self.P_oxidizer * 1e5), pyfluids.Input.temperature(self.T_oxidizer - 273.15))

        # First go over the oxidizer side of the system. Get state after oxidizer pump and power required to drive it
        self.pumped_oxidizer, self.w_pumped_oxidizer = cycle_functions.calculate_state_after_pump_for_pyfluids(
            fluid=self.oxidizer, delta_P=dP_OP, efficiency=self.eta_isotropic_OP)
        self.Power_OP = self.w_pumped_oxidizer * self.mdot_oxidizer
        # Change oxidizer into RocketCycleFluid object.
        self.pumped_oxidizer = pyfluid_to_rocket_cycle_fluid(fluid=self.pumped_oxidizer,
                                                             CEA_name=self.oxidizer_CEA_name,
                                                             type="oxidizer", phase="liquid")
        # Calculate temperature rise in oxidizer pump
        self.dT_OP = self.pumped_oxidizer.Ts - self.T_oxidizer
        # Assign oxidizer massflow into oxidizer preburner
        self.mdot_ox_OPB = self.mdot_oxidizer

        # Now perform calculations related to oxidizer preburner. First calculate its pressure based on oxidizer total
        # pressure.
        self.P_inj_OPB = self.pumped_oxidizer.Pt / (1 + self.dP_over_Pinj_OPB)

        # Before we calculate state after oxidizer preburner, we need to get our pumped amd boosted fuel. The pressure
        # rise across the main fuel pump is given, so the pressure rise in the boosted pump needs to be calculated.
        # First calculate states after the main fuel pump and power required to drive it.
        self.pumped_fuel, self.w_pumped_fuel = \
            cycle_functions.calculate_state_after_pump_for_pyfluids(fluid=self.fuel, delta_P=dP_FP,
                                                                    efficiency=self.eta_isotropic_FP)
        self.Power_FP = self.w_pumped_fuel * self.mdot_fuel
        # Temperature rise in the fuel pump can be calculated too
        self.dT_FP = self.pumped_fuel.temperature + 273.15 - self.T_fuel
        # Now calculate the booster pump pressure rise
        self.dP_BFP = self.pumped_oxidizer.Pt - (self.pumped_fuel.pressure / 1e5)  # bar
        # If it is positive, get the state after booster pump
        if self.dP_BFP > 0:
            self.boosted_fuel, self.w_boosted_fuel = \
                cycle_functions.calculate_state_after_pump_for_pyfluids(fluid=self.pumped_fuel, delta_P=self.dP_BFP,
                                                                        efficiency=self.eta_isotropic_BFP)
            # Change boosted fuel into RocketCycleFluid object
            self.boosted_fuel = pyfluid_to_rocket_cycle_fluid(fluid=self.boosted_fuel, CEA_name=self.fuel_CEA_name,
                                                              type="fuel", phase="liquid")
            # And temperature rise in the booster pump can be calculated as well
            self.dT_BFP = self.boosted_fuel.Ts - self.pumped_fuel.temperature
            # Create placeholder for boosted fuel for use in preburner
            pressurized_fuel = self.boosted_fuel
            # Otherwise pressurized_fuel for use in preburner is the one after the main pumps
        else:
            pressurized_fuel = pyfluid_to_rocket_cycle_fluid(fluid=self.pumped_fuel, CEA_name=self.fuel_CEA_name,
                                                             type="fuel", phase="liquid")

        # For sizing mode, fuel massflow that will allow to achieve certain OPB temperature needs to be found.
        # For analysis, it is already known.
        # Find oxidizer massflow if sizing mode is used:
        if mode == "sizing":
            # Fuel massflow that will allow to achieve certain FPB temperature can be found using bisection
            # algorithm. First define a function for it to solve.
            def calculate_OPB_temperature_residual(mdot_f):
                # Get its OF ratio.
                OF_OPB = self.mdot_ox_OPB / mdot_f
                # Get results for FPB.
                OPB_CEA_output, OPB_products = \
                    cycle_functions.calculate_state_after_preburner(OF=OF_OPB, preburner_inj_pressure=self.P_inj_OPB,
                                                                    CR=self.CR_OPB,
                                                                    fuel=pressurized_fuel,
                                                                    oxidizer=self.pumped_oxidizer)
                # Return residual. Total temperature is used, since this is what will be next to the walls due to gas
                # slowing down due to boundary layer.
                return OPB_products.Tt - self.T_OPB_required

            # Now solve the function to find the massflow that satisfies requirements. 0.04 and 0.10 are good lower
            # and upper bounds for any reasonable OPB temperature.
            self.mdot_crossflow_fuel = opt.toms748(calculate_OPB_temperature_residual,
                                                   a=mdot_crossflow_fuel_over_mdot_fuel_bracket[0] * self.mdot_fuel,
                                                   b=mdot_crossflow_fuel_over_mdot_fuel_bracket[1] * self.mdot_fuel,
                                                   maxiter=mdot_crossflow_oxidizer_maxiter,
                                                   xtol=mdot_crossflow_fuel_xtol)
        # Now get OPB OF ratio.
        self.OF_OPB = self.mdot_ox_OPB / self.mdot_crossflow_fuel
        # And get results for FPB.
        self.OPB_CEA_output, self.OPB_products = \
            cycle_functions.calculate_state_after_preburner(OF=self.OF_OPB, preburner_inj_pressure=self.P_inj_OPB,
                                                            CR=self.CR_OPB, fuel=pressurized_fuel,
                                                            oxidizer=self.pumped_oxidizer)
        # Also get the power required to drive the booster pump
        self.Power_BFP = self.w_boosted_fuel * self.mdot_crossflow_fuel

        # Calculate state after oxygen turbine.
        self.mdot_OT = self.mdot_oxidizer + self.mdot_crossflow_fuel
        self.OT_shaft_power = self.Power_BFP + self.Power_FP + self.Power_OP
        (self.OT_beta_tt, self.OT_outlet_gas, self.OT_equilibrium_gas, self.OT_equilibrium_gas_CEA_output,
         self.OT_molar_Cp_average, self.OT_gamma_average) = \
            cycle_functions.calculate_state_after_turbine(massflow=self.mdot_OT, turbine_power=self.OT_shaft_power,
                                                          turbine_polytropic_efficiency=self.eta_polytropic_OT,
                                                          preburner_products=self.OPB_products,
                                                          turbine_axial_velocity=self.axial_velocity_OT,
                                                          pressure_recovery_factor=self.Ps_Pt_OT)

        # Calculate state after cooling channels and change heated fuel into RocketCycleFluid object.
        self.mdot_cooling_channels_inlet = self.mdot_fuel - self.mdot_crossflow_fuel
        self.heated_fuel, self.mdot_cooling_channels_outlet = \
            cycle_functions.calculate_state_after_cooling_channels_for_Pyfluids(
                fluid=self.pumped_fuel, mdot_coolant=self.mdot_cooling_channels_inlet, mdot_film=self.mdot_film,
                pressure_drop=self.dP_cooling_channels,
                temperature_rise=self.dT_cooling_channels)
        # Now both pumped and heated fuel can be changed to RocketCycleFluid
        self.pumped_fuel = pyfluid_to_rocket_cycle_fluid(fluid=self.pumped_fuel, CEA_name=self.fuel_CEA_name,
                                                         type="fuel", phase="liquid")
        # Phase below is given as "liquid", so that total and static pressure are assumed to be about equal after the
        # cooling channels, but it will also work when there are two-phases with large vapor content or the fluid is
        # supercritical due to pressure recovery in the manifold afterward
        self.heated_fuel = pyfluid_to_rocket_cycle_fluid(fluid=self.heated_fuel, CEA_name=self.fuel_CEA_name,
                                                         type="fuel", phase="liquid")

        # Calculate combustion chamber performance. The smaller pressure must be taken (so that CC pressure is always
        # smaller than the injected propellant pressure)
        self.P_inj_CC = min(self.OT_equilibrium_gas.Ps, self.heated_fuel.Pt) / (1 + self.dP_over_Pinj_CC)
        # Now get CC results.
        (self.CC_CEA_output, self.CC_with_film_CEA_output, self.P_plenum_CC, self.IspVac_real, self.IspSea_real,
         self.CC_Tcomb, self.ThrustVac, self.ThrustSea, self.A_t_CC, self.A_e_CC,
         self.sea_level_nozzle_operation_mode) = cycle_functions.calculate_combustion_chamber_performance(
            mdot_oxidizer=self.mdot_OT, mdot_fuel=self.mdot_cooling_channels_outlet, oxidizer=self.OT_equilibrium_gas,
            fuel=self.heated_fuel, CC_pressure_at_injector=self.P_inj_CC, CR=self.CR_CC, eps=self.eps_CC,
            eta_cstar=self.eta_cstar, eta_cf=self.eta_cf, mdot_film=self.mdot_film,
            coolant_CEA_card=self.pumped_fuel.CEA_card)


class ClosedCatalyst_LRE(Cycle):
    def __init__(self, OF, oxidizer_rocket_cycle_fluid, fuel_rocket_cycle_fluid, eta_isotropic_OP, eta_isotropic_FP,
                 eta_polytropic_OT, eta_cstar, eta_cf, Ps_Pt_OT, dP_over_Pinj_CC, dP_over_Pinj_catalyst, CR_catalyst,
                 CR_CC, eps_CC, mdot_film_over_mdot_oxid, dP_cooling_channels,
                 dT_cooling_channels, axial_velocity_OT):
        """A class to analyse and size closed catalyst cycle.

        :param float or int OF: Oxidizer-to-Fuel ratio of the cycle
        :param RocketCycleFluid oxidizer_rocket_cycle_fluid: Rocket Cycle Fluid object representing oxidizer
        :param RocketCycleFluid fuel_rocket_cycle_fluid: Rocket Cycle Fluid object representing fuel
        :param float or int eta_isotropic_OP: Isotropic efficiency of the oxidizer pump (-)
        :param float or int eta_isotropic_FP: Isotropic efficiency of the fuel pump (-)
        :param float or int eta_polytropic_OT: Polytropic efficiency of the oxidizer turbine (-)
        :param float or int eta_cstar: C* efficiency of the CC (-)
        :param float or int eta_cf: Cf efficiency of the CC (-)
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
        :param float or int dP_cooling_channels: Pressure drop in the cooling channels (bar)
        :param float or int dT_cooling_channels: Temperature rise in the cooling channels (K)
        :param float or int axial_velocity_OT: Axial velocity across oxidizer turbine (m/s)
        """
        super().__init__(name="CC", OF=OF, oxidizer_rocket_cycle_fluid=oxidizer_rocket_cycle_fluid,
                         fuel_rocket_cycle_fluid=fuel_rocket_cycle_fluid, eta_isotropic_OP=eta_isotropic_OP,
                         eta_isotropic_FP=eta_isotropic_FP, eta_polytropic_OT=eta_polytropic_OT, eta_cstar=eta_cstar,
                         eta_cf=eta_cf, Ps_Pt_OT=Ps_Pt_OT, dP_over_Pinj_CC=dP_over_Pinj_CC, CR_catalyst=CR_catalyst,
                         CR_CC=CR_CC, eps_CC=eps_CC, mdot_film_over_mdot_oxid=mdot_film_over_mdot_oxid,
                         dP_cooling_channels=dP_cooling_channels, dP_over_Pinj_catalyst=dP_over_Pinj_catalyst,
                         dT_cooling_channels=dT_cooling_channels, axial_velocity_OT=axial_velocity_OT)

    def analyze_cycle(self, mdot_total, dP_OP, dP_FP):
        """A function to analyze the cycle for given arguments.

        :param int or float mdot_total: Total massflow (kg/s)
        :param int or float dP_FP: Pressure rise (bar) in the fuel pump.
        :param int or float dP_OP: Pressure rise (bar) in the oxidizer pump.

        :return: ClassParameters object storing data about the cycle.
        """

        # Assign attributes
        self.mdot_total = mdot_total
        self.dP_OP = dP_OP
        self.dP_FP = dP_FP

        # Create propellants
        self.fuel = self.fuel_rocket_cycle_fluid
        self.oxidizer = self.oxidier_rocket_cycle_fluid

        # First calculate oxidizer, fuel, film crossflow massflows
        self.mdot_fuel = mdot_total / (1 + self.OF)
        self.mdot_oxidizer = self.OF * self.mdot_fuel
        self.mdot_film = self.mdot_oxidizer * self.mdot_film_over_mdot_oxid

        # First go over the oxidizer side of the system. Get state after oxidizer pump and power required to drive it
        self.pumped_oxidizer, self.w_pumped_oxidizer = \
            cycle_functions.calculate_state_after_pump(fluid=self.oxidizer, delta_P=dP_OP,
                                                       efficiency=self.eta_isotropic_OP)
        self.Power_OP = self.w_pumped_oxidizer * self.mdot_oxidizer
        self.dT_OP = self.pumped_oxidizer.Ts - self.oxidizer.Ts

        # Calculate state after cooling channels.
        self.heated_oxidizer, self.mdot_cooling_channels_outlet = \
            cycle_functions.calculate_state_after_cooling_channels(
                fluid=self.pumped_oxidizer, mdot_coolant=self.mdot_oxidizer, mdot_film=self.mdot_film,
                pressure_drop=self.dP_cooling_channels,
                temperature_rise=self.dT_cooling_channels)

        # Now perform calculations related to catalyst bed. First calculate its (outlet) pressure based on oxidizer
        # injection pressure.
        self.P_inj_catalyst = self.heated_oxidizer.Pt / (1 + self.dP_over_Pinj_catalyst)

        # And get results for catalyst. Here OPB related parameters in CP are used as placeholder for catalyst
        # parameters.
        self.catalyst_CEA_output, self.catalyst_products = \
            cycle_functions.calculate_state_after_preburner(preburner_inj_pressure=self.P_inj_catalyst,
                                                            CR=self.CR_catalyst,
                                                            monopropellant=self.heated_oxidizer)

        # Now calculate state after fuel pump and power required to drive it.
        self.pumped_fuel, self.w_pumped_fuel = cycle_functions.calculate_state_after_pump(
            fluid=self.fuel, delta_P=dP_FP, efficiency=self.eta_isotropic_FP)
        self.Power_FP = self.w_pumped_fuel * self.mdot_fuel
        self.dT_FP = self.pumped_fuel.Ts - self.fuel.Ts

        # Calculate state after oxygen turbine.
        self.OT_shaft_power = self.Power_FP + self.Power_OP
        self.mdot_OT = self.mdot_cooling_channels_outlet
        (self.OT_beta_tt, self.OT_outlet_gas, self.OT_equilibrium_gas, self.OT_equilibrium_gas_CEA_output,
         self.OT_molar_Cp_average, self.OT_gamma_average) = \
            cycle_functions.calculate_state_after_turbine(massflow=self.mdot_OT, turbine_power=self.OT_shaft_power,
                                                          turbine_polytropic_efficiency=self.eta_polytropic_OT,
                                                          preburner_products=self.catalyst_products,
                                                          turbine_axial_velocity=self.axial_velocity_OT,
                                                          pressure_recovery_factor=self.Ps_Pt_OT)

        # Calculate combustion chamber performance. The smaller pressure must be taken (so that CC pressure is always
        # smaller than the injected propellant pressure)
        self.P_inj_CC = min(self.OT_equilibrium_gas.Ps, self.pumped_fuel.Pt) / (1 + self.dP_over_Pinj_CC)
        # Now get CC results.
        (self.CC_CEA_output, self.CC_with_film_CEA_output, self.P_plenum_CC, self.IspVac_real, self.IspSea_real,
         self.CC_Tcomb, self.ThrustVac, self.ThrustSea, self.A_t_CC, self.A_e_CC,
         self.sea_level_nozzle_operation_mode) = cycle_functions.calculate_combustion_chamber_performance(
            mdot_oxidizer=self.mdot_OT, mdot_fuel=self.mdot_fuel, oxidizer=self.OT_equilibrium_gas,
            fuel=self.pumped_fuel, CC_pressure_at_injector=self.P_inj_CC, CR=self.CR_CC, eps=self.eps_CC,
            eta_cstar=self.eta_cstar, eta_cf=self.eta_cf, mdot_film=self.mdot_film,
            coolant_CEA_card=self.pumped_oxidizer.CEA_card)


class CycleSizing:
    def __init__(self, cycle, P_CC_plenum_required, mdot_total_0, dP_FP_0, lb, ub, T_OPB_required=None,
                 T_FPB_required=None, mdot_crossflow_fuel_over_mdot_fuel_0=None, ThrustVac_required=None,
                 ThrustSea_required=None, dP_OP_0=None, inner_loop_bracket=(0.04, 0.11), inner_loop_xtol=1e-3,
                 inner_loop_maxiter=1000, jac="3-point", method="dogbox", loss="soft_l1", tr_solver="exact",
                 tr_options=None, xtol=1e-3, ftol=1e-3, gtol=None, fscale=0.5, max_nfev=None, diff_step=1e-3,
                 verbose=2):
        """ A class to size the cycle to achieve given thrust, CC pressure, preburner temperatures (if present).

        :param FFSC_LRE or ORCS_LRE or CC_LRE cycle: Cycle subclass representing cycle which is to be sized.
        :param float or int ThrustVac_required: Vacuum thrust (kN) required (either this argument or ThrustSea_required
            must be given)
        :param float or int ThrustSea_required: Sea level thrust (kN) required (either this argument or
            ThrustVac_required must be given)
        :param float or int P_CC_plenum_required: Combustion chamber plenum pressure (bar) required
        :param float or int T_OPB_required: Oxidizer preburner temperature (K) required (if such preburner is present)
        :param float or int T_FPB_required: Fuel preburner temperature (K) required (if such preburner is present)
        :param float or int mdot_total_0: Initial solution estimate for total propellant massflow (kg/s)
        :param float or int mdot_crossflow_fuel_over_mdot_fuel_0: Initial solution estimate for fuel crossflow massflow
            to fuel massflow ratio (if oxidizer preburner is present)
        :param float or int dP_FP_0: Initial solution estimate for fuel pump pressure rise (bar)
        :param float or int dP_OP_0: Initial solution estimate for oxidizer pump pressure rise (bar)
        :param tuple inner_loop_bracket: A tuple of floats representing oxidizer crossflow to oxidizer massflow/
            fuel crossflow to fuel massflow for inner loop, which calculates oxidizer/fuel
            crossflow based on required fuel/oxidizer preburner temperatures
        :param float inner_loop_xtol: Absolute convergence tolerance (kg/s) for terminating inner loop,
            which calculates oxidizer/fuel crossflow based on required preburner temperatures
        :param int inner_loop_maxiter: Maximum number of iterations for the inner loop, which calculates oxidizer/fuel
            crossflow based on required fuel/oxidizer preburner temperatures
        :param list lb: A list of lower bounds for sized variables. For FFSC these are mdot_total,
            mdot_crossflow_fuel_over_mdot_fuel, dP_FP. For ORSC and CC these are mdot_total, dP_OP, dP_FP.
        :param list ub: A list of upper bounds for sized variables. For FFSC these are mdot_total,
            mdot_crossflow_fuel_over_mdot_fuel, dP_FP. For ORSC and CC these are mdot_total, dP_OP, dP_FP.
        :param string jac: jac argument for scipy.optimize.least_squares
        :param string method: method argument for scipy.optimize.least_squares
        :param string loss: loss argument for scipy.optimize.least_squares
        :param string tr_solver: tr_solver argument for scipy.optimize.least_squares
        :param dict tr_options: tr_options argument for scipy.optimize.least_squares
        :param float or int xtol: xtol argument for scipy.optimize.least_squares
        :param float or int ftol: ftol argument for scipy.optimize.least_squares
        :param float or int gtol: gtol argument for scipy.optimize.least_squares
        :param float or int fscale: fscale argument for scipy.optimize.least_squares
        :param float or int diff_step: diff_step argument for scipy.optimize.least_squares
        :param int max_nfev: max_nfev argument for scipy.optimize.least_squares
        :param int verbose: verbose argument for scipy.optimize.least_squares
        """

        # First raise warnings if incorrect arguments are given
        if ThrustSea_required is None and ThrustVac_required is None:
            warnings.simplefilter("error", UserWarning)
            warnings.warn("Either ThrustSea_required or ThrustVac_required must be a float or integer")
        elif ThrustSea_required is not None and ThrustVac_required is not None:
            warnings.simplefilter("error", UserWarning)
            warnings.warn("Only one attribute from ThrustSea_required or ThrustVac_required can be a float or integer")

        if cycle.name == "FFSC" and any(param is None for param in [T_OPB_required, T_FPB_required,
                                                                    mdot_crossflow_fuel_over_mdot_fuel_0]):
            warnings.simplefilter("error", UserWarning)
            warnings.warn("T_OPB_required, T_FPB_required, mdot_total_0, mdot_crossflow_fuel_over_mdot_fuel_0, dP_FP_0"
                          " cannot be None for FFSC cycle")
        elif cycle.name == "ORSC" and any(param is None for param in [dP_OP_0, T_OPB_required]):
            warnings.simplefilter("error", UserWarning)
            warnings.warn("mdot_total_0, dP_OP_0, dP_FP_0, T_OPB_required cannot be None for ORSC cycle")
        elif cycle.name == "CC" and dP_OP_0 is None:
            warnings.simplefilter("error", UserWarning)
            warnings.warn("mdot_total_0, dP_OP_0, dP_FP_0 cannot be None for CC cycle")
        elif cycle.name != "FFSC" and cycle.name != "ORSC" and cycle.name != "CC":
            warnings.simplefilter("error", UserWarning)
            warnings.warn("cycle.name not recognized")

        # Assign required values and class attributes
        if ThrustVac_required is not None:
            cycle.Thrust_required = ThrustVac_required
            self.Thrust_req_operating_point = "vacuum"
        elif ThrustSea_required is not None:
            cycle.Thrust_required = ThrustSea_required
            self.Thrust_req_operating_point = "sea_level"
        cycle.P_plenum_CC_required = P_CC_plenum_required

        # Assign other class attributes
        self.cycle = cycle
        self.inner_loop_bracket = inner_loop_bracket
        self.inner_loop_xtol = inner_loop_xtol
        self.inner_loop_maxiter = inner_loop_maxiter
        self.cycle.T_OPB_required = T_OPB_required
        self.cycle.T_FPB_required = T_FPB_required

        # Create reference vector with non-normalized initial solution estimate. Also create a vector for normalizing
        # residuals later on
        if cycle.name == "FFSC":
            self.x_ref = np.array([mdot_total_0, mdot_crossflow_fuel_over_mdot_fuel_0, dP_FP_0])
            self.y_ref = np.array([cycle.Thrust_required, cycle.P_plenum_CC_required, cycle.T_OPB_required])
        elif cycle.name == "ORSC" or cycle.name == "CC":
            self.x_ref = np.array([mdot_total_0, dP_OP_0, dP_FP_0])
            self.y_ref = np.array([cycle.Thrust_required, cycle.P_plenum_CC_required, cycle.P_plenum_CC_required])
        # Now assemble normalized initial solution estimate vector
        self.x0 = np.ones(3)
        # Assemble and normalize bounds with reference vector
        self.norm_bounds = (lb / self.x_ref, ub / self.x_ref)

        # Get initial residuals, both non- and non-normalized.
        self.norm_residuals_0 = self.calculate_normalized_residuals(self.x0)
        self.residuals_0 = self.norm_residuals_0 * self.y_ref

        # Get the solution. Least-squares approach is used, because it is the only method that would converge.
        # Solution is normalized, so it needs to be scaled again.
        self.result = opt.least_squares(fun=self.calculate_normalized_residuals, x0=self.x0, jac=jac,
                                        bounds=self.norm_bounds, method=method, loss=loss, tr_solver=tr_solver,
                                        tr_options=tr_options, xtol=xtol, ftol=ftol, gtol=gtol, f_scale=fscale,
                                        verbose=verbose, diff_step=diff_step, max_nfev=max_nfev)
        self.x = self.result.x * self.x_ref

        # Get residuals for the solution, both normalized and non-normalized.
        self.norm_residuals = self.calculate_normalized_residuals(self.result.x)
        self.residuals = self.norm_residuals * self.y_ref

    def calculate_normalized_residuals(self, x):
        """A function to get the residuals, which need to be zero to satisfy all constraints.

        :param np.ndarray x: Normalized estimate of the solution.
        """

        # First analyse cycle if FFSC is used
        if self.cycle.name == "FFSC":
            # Retrieve non-normalized arguments
            [mdot_total, mdot_crossflow_fuel_over_mdot_fuel, dP_FP] = x * self.x_ref
            # Analyze the cycle
            self.cycle.analyze_cycle(mdot_total=mdot_total, dP_FP=dP_FP, T_FPB_required=self.cycle.T_FPB_required,
                                     mdot_crossflow_fuel_over_mdot_fuel=mdot_crossflow_fuel_over_mdot_fuel)
        # Now do it if ORSC is used
        elif self.cycle.name == "ORSC":
            # Retrieve non-normalized arguments
            [mdot_total, dP_OP, dP_FP] = x * self.x_ref
            # Analyze the cycle
            self.cycle.analyze_cycle(mdot_total=mdot_total, dP_OP=dP_OP, dP_FP=dP_FP,
                                     T_OPB_required=self.cycle.T_OPB_required)
        # Now do it if CC is used
        elif self.cycle.name == "CC":
            # Retrieve non-normalized arguments
            [mdot_total, dP_OP, dP_FP] = x * self.x_ref
            # Analyze the cycle
            self.cycle.analyze_cycle(mdot_total=mdot_total, dP_OP=dP_OP, dP_FP=dP_FP)

        # Get residuals. All residuals are normalized wrt some reference scales.
        # Get thrust residual
        if self.Thrust_req_operating_point == "sea_level":
            residual_thrust = (self.cycle.ThrustSea - self.cycle.Thrust_required) / self.cycle.Thrust_required
        elif self.Thrust_req_operating_point == "vacuum":
            residual_thrust = (self.cycle.ThrustVac - self.cycle.Thrust_required) / self.cycle.Thrust_required

        # Get CC pressure residual
        residual_CC_pressure = \
            (self.cycle.P_plenum_CC - self.cycle.P_plenum_CC_required) / self.cycle.P_plenum_CC_required

        # If FFSC is used, get OPB temperature residuals. Again total temperature is used, bcs it will be closer to
        # adiabatic temperature near the walls.
        if self.cycle.name == "FFSC":
            residual_T_OPB = (self.cycle.OPB_products.Tt - self.cycle.T_OPB_required) / self.cycle.T_OPB_required
            # Return the residuals for FFSC
            return [residual_thrust, residual_CC_pressure, residual_T_OPB]

        # If ORSC or CC is used, get propellants pressure difference in the manifold residual. It is desirable that
        # these are the same to limit pressure drops
        elif self.cycle.name == "ORSC" or self.cycle.name == "CC":
            residual_dP_propellants = \
                (self.cycle.pumped_fuel.Pt - self.cycle.OT_equilibrium_gas.Ps) / self.cycle.P_plenum_CC_required
            # Return the residuals for ORSC/CC
            return [residual_thrust, residual_CC_pressure, residual_dP_propellants]

    def get_residuals(self):
        """A function to return string about initial and final residuals."""

        string = \
            (f"\n---INITIAL RESIDUALS---\n"
             f"- Difference between actual and desired thrust : {self.residuals_0[0]:.5g} kN"
             f" (normalized: {self.norm_residuals_0[0]:.5g})\n"
             f"- Pressure difference between actual and desired CC plenum pressure: {self.residuals_0[1]:.5g} bar"
             f" (normalized: {self.norm_residuals_0[1]:.5g})\n")
        if self.cycle.name == "FFSC":
            string += (
                f"- Temperature difference between actual and desired temperature in OPB: {self.residuals_0[2]:.5g} K"
                f" (normalized: {self.norm_residuals_0[2]:.5g})\n\n")
        elif self.cycle.name == "ORSC" or self.cycle.name == "CC":
            string += (
                f"- Pressure difference between fuel and oxidizer in the CC manifold: {self.residuals_0[2]:.5g} bar"
                f" (normalized: {self.norm_residuals_0[2]:.5g})\n\n")
        string += (
            f"\n---REMAINING RESIDUALS---\n"
            f"- Difference between actual and desired thrust: {self.residuals[0]:.5g} kN"
            f" (normalized: {self.norm_residuals[0]:.5g})\n"
            f"- Pressure difference between actual and desired CC plenum pressure: {self.residuals[1]:.5g} bar"
            f" (normalized: {self.norm_residuals[1]:.5g})\n")
        if self.cycle.name == "FFSC":
            string += (
                f"- Temperature difference between actual and desired temperature in OPB: {self.residuals[2]:.5g} K"
                f" (normalized: {self.norm_residuals[2]:.5g})\n\n")
        elif self.cycle.name == "ORSC" or self.cycle.name == "CC":
            string += (
                f"- Pressure difference between fuel and oxidizer in the CC manifold: {self.residuals[2]:.5g} bar"
                f" (normalized: {self.norm_residuals[2]:.5g})\n\n")
        return string
