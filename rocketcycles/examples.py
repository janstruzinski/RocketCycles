import cycle_classes as RC
import pyfluids
from fluid import RocketCycleFluid

# An example of FFSC LRE using SpaceX Raptor engine - first in analysis mode
Raptor = RC.FFSC_LRE(OF=3.6, oxidizer_pyfluid=pyfluids.FluidsList.Oxygen, fuel_pyfluid=pyfluids.FluidsList.Methane,
                     fuel_CEA_name="CH4(L)", oxidizer_CEA_name="O2(L)", T_oxidizer=80, T_fuel=100, P_oxidizer=4,
                     P_fuel=4, eta_isotropic_OP=0.87, eta_isotropic_FP=0.84,
                     eta_polytropic_OT=0.9, eta_polytropic_FT=0.9, eta_cstar=0.99,
                     eta_cf=0.95, Ps_Pt_FT=0.95, Ps_Pt_OT=0.95, dP_over_Pinj_CC=0.1, dP_over_Pinj_OPB=0.1,
                     dP_over_Pinj_FPB=0.1, CR_CC=2.5, CR_FPB=4, CR_OPB=4, eps_CC=35, mdot_film_over_mdot_fuel=0.05,
                     dP_cooling_channels=190, dT_cooling_channels=100, axial_velocity_OT=200,
                     axial_velocity_FT=300, include_film_in_cstar=False)
Raptor.analyze_cycle(mdot_total=710, mdot_crossflow_fuel_over_mdot_fuel=0.045, dP_FP=880,
                     mdot_crossflow_ox_over_mdot_ox=0.075, dP_OP=690)
print(Raptor.get_full_output())

# It is possible to retrieve individual parameters stored in FFSC.LRE object.
print(f"Temperature of combustion in CC is: {Raptor.CC_Tcomb} K")

# Now with Cycle Sizing:
RaptorSizing = RC.CycleSizing(cycle=Raptor, ThrustSea_required=2230, P_CC_plenum_required=300, T_OPB_required=900,
                              T_FPB_required=900, mdot_total_0=710, mdot_crossflow_fuel_over_mdot_fuel_0=0.045,
                              dP_FP_0=880, lb=[510, 0.04, 680], ub=[910, 0.1, 1080])

print(RaptorSizing.cycle.get_full_output())
print(RaptorSizing.get_residuals())

# An example of ORSC LRE using Propalox - first in analysis mode
orsc_engine = RC.ORSC_LRE(OF=3.6, oxidizer_pyfluid=pyfluids.FluidsList.Oxygen,
                          fuel_pyfluid=pyfluids.FluidsList.nPropane, fuel_CEA_name="C3H8(L)",
                          oxidizer_CEA_name="O2(L)", T_oxidizer=85, T_fuel=90, P_oxidizer=4, P_fuel=4,
                          eta_isotropic_OP=0.87, eta_isotropic_FP=0.84, eta_isotropic_BFP=0.84, Ps_Pt_OT=0.95,
                          eta_polytropic_OT=0.9, eta_cstar=0.99, eta_cf=0.95, dP_over_Pinj_CC=0.1,
                          dP_over_Pinj_OPB=0.1, CR_CC=4, CR_OPB=4, eps_CC=35, mdot_film_over_mdot_fuel=0.05,
                          dP_cooling_channels=150, dT_cooling_channels=100,
                          axial_velocity_OT=200, include_film_in_cstar=True)
orsc_engine.analyze_cycle(mdot_total=710, dP_FP=520, dP_OP=700, mdot_crossflow_fuel_over_mdot_fuel=0.08)
print(orsc_engine.get_full_output())

# Now in sizing mode
ORSC_Sizing = RC.CycleSizing(cycle=orsc_engine, ThrustSea_required=2168, P_CC_plenum_required=200,
                             T_OPB_required=900, mdot_total_0=710, dP_FP_0=520, dP_OP_0=700, lb=[510, 500, 300],
                             ub=[910, 800, 850])
print(ORSC_Sizing.cycle.get_full_output())
print(ORSC_Sizing.get_residuals())

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

Draper = RC.ClosedCatalyst_LRE(OF=6.75, oxidizer_rocket_cycle_fluid=H2O2, fuel_rocket_cycle_fluid=JP10,
                               eta_isotropic_OP=0.87, eta_isotropic_FP=0.84, eta_polytropic_OT=0.9, eta_cstar=0.99,
                               eta_cf=0.95, Ps_Pt_OT=0.9, dP_over_Pinj_CC=0.15, dP_over_Pinj_catalyst=0.05, CR_CC=2.5,
                               CR_catalyst=4, eps_CC=35, mdot_film_over_mdot_oxid=0.02,
                               dP_cooling_channels=15, dT_cooling_channels=100,
                               axial_velocity_OT=400, include_film_in_cstar=True)
Draper.analyze_cycle(mdot_total=10, dP_FP=130, dP_OP=200)
print(Draper.get_full_output())

# Now do it in sizing mode
Draper_Sizing = RC.CycleSizing(cycle=Draper, ThrustSea_required=10, P_CC_plenum_required=150, mdot_total_0=10,
                               dP_FP_0=130, dP_OP_0=200, lb=[3, 100, 100], ub=[20, 300, 300])
print(Draper_Sizing.cycle.get_full_output())
print(Draper_Sizing.get_residuals())