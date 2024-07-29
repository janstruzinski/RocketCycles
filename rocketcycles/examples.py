import cycles as RC
import pyfluids

# An example of FFSC LRE using SpaceX Raptor engine - first in analysis mode
Raptor = RC.FFSC_LRE(OF=3.6, oxidizer=pyfluids.FluidsList.Oxygen, fuel=pyfluids.FluidsList.Methane,
                     fuel_CEA_name="CH4(L)", oxidizer_CEA_name="O2(L)", T_oxidizer=80, T_fuel=100, P_oxidizer=4,
                     P_fuel=4, eta_isotropic_OP=0.87, eta_isotropic_FP=0.84,
                     eta_polytropic_OT=0.9, eta_polytropic_FT=0.9, eta_FPB=0.99, eta_OPB=0.99, eta_cstar=0.99,
                     eta_isp=0.95, dP_over_Pinj_CC=0.15, dP_over_Pinj_OPB=0.15, dP_over_Pinj_FPB=0.15, CR_CC=2.5,
                     CR_FPB=4, CR_OPB=4, eps_CC=35, mdot_film_over_mdot_fuel=0.05,
                     cooling_channels_pressure_drop=190, cooling_channels_temperature_rise=100, axial_velocity_OT=200,
                     axial_velocity_FT=300, mdot_total_0=710, mdot_crossflow_ox_over_mdot_ox_0=0.075,
                     mdot_crossflow_f_over_mdot_f_0=0.045, dP_FP_0=880, dP_OP_0=690, mode="analysis")
print(Raptor.get_full_output())

# It is possible to retrieve individual parameters stored in FFSC.LRE object or its CP attribute.
print(f"Temperature of combustion in CC is: {Raptor.CP.CC_Tcomb} K")

# Now in sizing mode:
Raptor = RC.FFSC_LRE(OF=3.6, ThrustSea=2230, oxidizer=pyfluids.FluidsList.Oxygen, fuel=pyfluids.FluidsList.Methane,
                     fuel_CEA_name="CH4(L)", oxidizer_CEA_name="O2(L)", T_oxidizer=80, T_fuel=100, P_oxidizer=4,
                     P_fuel=4, P_plenum_CC=300, T_FPB=860, T_OPB=760, eta_isotropic_OP=0.87, eta_isotropic_FP=0.84,
                     eta_polytropic_OT=0.9, eta_polytropic_FT=0.9, eta_FPB=0.99, eta_OPB=0.99, eta_cstar=0.99,
                     eta_isp=0.95, dP_over_Pinj_CC=0.15, dP_over_Pinj_OPB=0.15, dP_over_Pinj_FPB=0.15, CR_CC=2.5,
                     CR_FPB=4, CR_OPB=4, eps_CC=35, mdot_film_over_mdot_fuel=0.05,
                     cooling_channels_pressure_drop=190, cooling_channels_temperature_rise=100, axial_velocity_OT=200,
                     axial_velocity_FT=300, mdot_total_0=710, mdot_crossflow_f_over_mdot_f_0=0.045, dP_FP_0=880,
                     lb=[510, 0.04, 680], ub=[910, 0.1, 1080], jac="3-point", method="dogbox", loss="soft_l1",
                     tr_solver="exact", xtol=1e-8, mode="sizing")
print(Raptor.get_full_output())
print(Raptor.get_residuals())

# An example of ORSC LRE using Propalox - first in analysis mode
orsc_engine = RC.ORSC_LRE(OF=3.6, oxidizer=pyfluids.FluidsList.Oxygen, fuel=pyfluids.FluidsList.nPropane,
                          fuel_CEA_name="C3H8(L)", oxidizer_CEA_name="O2(L)", T_oxidizer=80, T_fuel=90, P_oxidizer=4,
                          P_fuel=4, eta_isotropic_OP=0.87, eta_isotropic_FP=0.84, eta_isotropic_BFP=0.84,
                          eta_polytropic_OT=0.9, eta_OPB=0.99, eta_cstar=0.99, eta_isp=0.95, dP_over_Pinj_CC=0.15,
                          dP_over_Pinj_OPB=0.15, CR_CC=2.5, CR_OPB=4, eps_CC=35, mdot_film_over_mdot_fuel=0.05,
                          cooling_channels_pressure_drop=190, cooling_channels_temperature_rise=100,
                          axial_velocity_OT=200, mdot_total_0=710, mdot_crossflow_f_over_mdot_f_0=0.07, dP_FP_0=520,
                          dP_OP_0=800, mode="analysis")
print(orsc_engine.get_full_output())

# Now in sizing mode
orsc_engine = RC.ORSC_LRE(OF=3.6, oxidizer=pyfluids.FluidsList.Oxygen, fuel=pyfluids.FluidsList.nPropane,
                          fuel_CEA_name="C3H8(L)", oxidizer_CEA_name="O2(L)", T_oxidizer=80, T_fuel=90, P_oxidizer=4,
                          P_fuel=4, eta_isotropic_OP=0.87, eta_isotropic_FP=0.84, eta_isotropic_BFP=0.84,
                          eta_polytropic_OT=0.9, eta_OPB=0.99, eta_cstar=0.99, eta_isp=0.95, dP_over_Pinj_CC=0.15,
                          dP_over_Pinj_OPB=0.15, CR_CC=2.5, CR_OPB=4, eps_CC=35, mdot_film_over_mdot_fuel=0.05,
                          cooling_channels_pressure_drop=190, cooling_channels_temperature_rise=100,
                          axial_velocity_OT=200, mdot_total_0=710, dP_FP_0=520,
                          dP_OP_0=700, lb=[510, 500, 300], ub=[910, 800, 800], ThrustSea=2230, P_plenum_CC=300,
                          T_OPB=800, jac="3-point", method="dogbox", loss="soft_l1", tr_solver="exact", xtol=1e-8,
                          mode="sizing")
print(orsc_engine.get_full_output())
print(orsc_engine.get_residuals())

