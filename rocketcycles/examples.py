import cycles as RC
import pyfluids
from fluid import RocketCycleFluid


# An example of FFSC LRE using SpaceX Raptor engine - first in analysis mode
Raptor = RC.FFSC_LRE(OF=3.6, oxidizer=pyfluids.FluidsList.Oxygen, fuel=pyfluids.FluidsList.Methane,
                     fuel_CEA_name="CH4(L)", oxidizer_CEA_name="O2(L)", T_oxidizer=80, T_fuel=100, P_oxidizer=4,
                     P_fuel=4, eta_isotropic_OP=0.87, eta_isotropic_FP=0.84,
                     eta_polytropic_OT=0.9, eta_polytropic_FT=0.9, eta_FPB=0.99, eta_OPB=0.99, eta_cstar=0.99,
                     eta_isp=0.95, Ps_Pt_FT=0.9, Ps_Pt_OT=0.9, dP_over_Pinj_CC=0.15, dP_over_Pinj_OPB=0.15,
                     dP_over_Pinj_FPB=0.15, CR_CC=2.5, CR_FPB=4, CR_OPB=4, eps_CC=35, mdot_film_over_mdot_fuel=0.05,
                     cooling_channels_pressure_drop=190, cooling_channels_temperature_rise=100, axial_velocity_OT=200,
                     axial_velocity_FT=300, mdot_total_0=710, mdot_crossflow_ox_over_mdot_ox_0=0.075,
                     mdot_crossflow_f_over_mdot_f_0=0.045, dP_FP_0=880, dP_OP_0=690, mode="analysis")
print(Raptor.get_full_output())

# It is possible to retrieve individual parameters stored in FFSC.LRE object or its CP attribute.
print(f"Temperature of combustion in CC is: {Raptor.CP.CC_Tcomb} K")

# Now in sizing mode:
Raptor = RC.FFSC_LRE(OF=3.6, ThrustSea=2230, oxidizer=pyfluids.FluidsList.Oxygen, fuel=pyfluids.FluidsList.Methane,
                     fuel_CEA_name="CH4(L)", oxidizer_CEA_name="O2(L)", T_oxidizer=80, T_fuel=100, P_oxidizer=4,
                     P_fuel=4, P_plenum_CC=300, T_FPB=900, T_OPB=900, eta_isotropic_OP=0.87, eta_isotropic_FP=0.84,
                     eta_polytropic_OT=0.9, eta_polytropic_FT=0.9, eta_FPB=0.99, eta_OPB=0.99, eta_cstar=0.99,
                     eta_isp=0.95, dP_over_Pinj_CC=0.14, dP_over_Pinj_OPB=0.14, dP_over_Pinj_FPB=0.14, CR_CC=2.5,
                     CR_FPB=4, CR_OPB=4, eps_CC=35, mdot_film_over_mdot_fuel=0.05, Ps_Pt_OT=0.95, Ps_Pt_FT=0.95,
                     cooling_channels_pressure_drop=190, cooling_channels_temperature_rise=100, axial_velocity_OT=200,
                     axial_velocity_FT=300, mdot_total_0=710, mdot_crossflow_f_over_mdot_f_0=0.045, dP_FP_0=880,
                     lb=[510, 0.04, 680], ub=[910, 0.1, 1080], jac="3-point", method="dogbox", loss="soft_l1",
                     tr_solver="exact", xtol=1e-8, mode="sizing")
print(Raptor.get_full_output())
print(Raptor.get_residuals())

# An example of ORSC LRE using Propalox - first in analysis mode
orsc_engine = RC.ORSC_LRE(OF=3.6, oxidizer=pyfluids.FluidsList.Oxygen, fuel=pyfluids.FluidsList.nPropane,
                          fuel_CEA_name="C3H8(L)", oxidizer_CEA_name="O2(L)", T_oxidizer=80, T_fuel=90, P_oxidizer=4,
                          P_fuel=4, eta_isotropic_OP=0.87, eta_isotropic_FP=0.84, eta_isotropic_BFP=0.84, Ps_Pt_OT=0.9,
                          eta_polytropic_OT=0.9, eta_OPB=0.99, eta_cstar=0.99, eta_isp=0.95, dP_over_Pinj_CC=0.15,
                          dP_over_Pinj_OPB=0.15, CR_CC=2.5, CR_OPB=4, eps_CC=35, mdot_film_over_mdot_fuel=0.05,
                          cooling_channels_pressure_drop=190, cooling_channels_temperature_rise=100,
                          axial_velocity_OT=200, mdot_total_0=710, mdot_crossflow_f_over_mdot_f_0=0.07, dP_FP_0=520,
                          dP_OP_0=800, mode="analysis")
print(orsc_engine.get_full_output())

# Now in sizing mode
orsc_engine = RC.ORSC_LRE(OF=3.6, oxidizer=pyfluids.FluidsList.Oxygen, fuel=pyfluids.FluidsList.nPropane,
                          fuel_CEA_name="C3H8(L)", oxidizer_CEA_name="O2(L)", T_oxidizer=80, T_fuel=90, P_oxidizer=4,
                          P_fuel=4, eta_isotropic_OP=0.87, eta_isotropic_FP=0.84, eta_isotropic_BFP=0.84, Ps_Pt_OT=0.9,
                          eta_polytropic_OT=0.9, eta_OPB=0.99, eta_cstar=0.99, eta_isp=0.95, dP_over_Pinj_CC=0.15,
                          dP_over_Pinj_OPB=0.15, CR_CC=2.5, CR_OPB=4, eps_CC=35, mdot_film_over_mdot_fuel=0.05,
                          cooling_channels_pressure_drop=190, cooling_channels_temperature_rise=100,
                          axial_velocity_OT=200, mdot_total_0=710, dP_FP_0=520,
                          dP_OP_0=700, lb=[510, 500, 300], ub=[910, 800, 800], ThrustSea=2230, P_plenum_CC=300,
                          T_OPB=800, jac="3-point", method="dogbox", loss="soft_l1", tr_solver="exact", xtol=1e-8,
                          mode="sizing")
print(orsc_engine.get_full_output())
print(orsc_engine.get_residuals())



# An example of Closed Catalyst LRE using H2O2-RP1 based on Ursa Major Draper engine - first in analysis mode
# Define oxidizer
H2O2 = RocketCycleFluid(species=["H2O2(L)", "H2O(L)"], mass_fractions=[0.98, 0.02], temperature=298.95, type="oxid",
                        phase="liquid")
H2O2.Pt = 3  # bar
H2O2.Ps = 3  # bar
H2O2.density = 1398.86  # kg/m^3

# Define fuel
JP10 = RocketCycleFluid(species=["JP-10(L)"], mass_fractions=[1], temperature=298.15, type="fuel",
                        phase="liquid", species_molar_Cp=[236.49])
JP10.Pt = 3  # bar
JP10.Ps = 3  # bar
JP10.density = 931.8  # kg/m^3

Draper = RC.ClosedCatalyst_LRE(OF=6.75, oxidizer=H2O2, fuel=JP10, eta_isotropic_OP=0.87, eta_isotropic_FP=0.84,
                               eta_polytropic_OT=0.9, eta_catalyst=0.99, eta_cstar=0.99, eta_isp=0.95, Ps_Pt_OT=0.9,
                               dP_over_Pinj_CC=0.15, dP_over_Pinj_catalyst=0.05, CR_CC=2.5, CR_catalyst=4, eps_CC=35,
                               mdot_film_over_mdot_oxid=0.02, cooling_channels_pressure_drop=15,
                               cooling_channels_temperature_rise=100, axial_velocity_OT=400, mdot_total_0=10,
                               dP_FP_0=130, dP_OP_0=200, mode="analysis")
print(Draper.get_full_output())

# Now in sizing mode
Draper = RC.ClosedCatalyst_LRE(OF=7, oxidizer=H2O2, fuel=JP10, eta_isotropic_OP=0.87, eta_isotropic_FP=0.84,
                               eta_polytropic_OT=0.9, eta_catalyst=0.99, eta_cstar=0.99, eta_isp=0.95, Ps_Pt_OT=0.9,
                               dP_over_Pinj_CC=0.15, dP_over_Pinj_catalyst=0.05, CR_CC=2.5, CR_catalyst=4, eps_CC=35,
                               mdot_film_over_mdot_oxid=0.05, cooling_channels_pressure_drop=15,
                               cooling_channels_temperature_rise=100, axial_velocity_OT=400, mdot_total_0=4,
                               dP_FP_0=130, dP_OP_0=200, mode="sizing", ThrustSea=10, P_plenum_CC=150,
                               lb=[3, 100, 100], ub=[20, 200, 400], jac="3-point", method="dogbox",
                               loss="soft_l1", tr_solver="exact", xtol=1e-8)
print(Draper.get_full_output())
print(Draper.get_residuals())