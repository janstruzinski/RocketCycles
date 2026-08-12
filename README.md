# RocketCycles

## Introduction

Hi,

Welcome to RocketCycles. It is a Python library dedicated to staged combustion rocket engine cycles. It
allows to both analyse them for given arguments and size them for given constraints in order to get combustion chamber
performance, pressure ratio of the turbines or pressure rise in the pumps, crossflow massflows etc.

I hope you will find the library useful!

Cheers,
Jan

## General Overview

cycle_functions.py stores functions representing pumps, turbines, preburners, combustion chambers. They can be
assembled together into a staged combustion rocket engine cycle.

fluid.py stores RocketCycleFluid class that conveniently allows to store data about propellants or
preburner products, as well as to get their thermophysical properties using NASA 9 polynomials or real gas density
using PyFluids (wrapper about CoolProp).

cycle_classes.py stores definitions of classes representing different staged combustion rocket engine cycles. Cycle
class is a parent class with shared attributes and methods. FFSC_LRE, ORSC_LRE, CC_LRE are subclasses containing
analysis functions specific to these cycles. These can be used as inspiration if you want to assemble your own closed
cycle. CycleSizing allows to size any of the cycles above for given arguments.

examples.py shows how to call these classes and perform the analysis.

tests.py stores unit tests for the package.

It is highly recommened that you see the code or Model Documentation below, so that you know what is happening under
the hood and are aware of any assumptions.

## Installation

For installation, first install nasaPoly from https://github.com/ptgodart/nasaPoly.git

To add JP-10 and RP-1, find raw.dat in nasaPoly installation directory and add the following data to other species:

```text
JP-10(L)          Exo-tetrahydrodicyclopentadiene. Smith,1979.
 0 g 6/01 C  10.00H  16.00    0.00    0.00    0.00 0   136.234040    -122800.400
    298.150      0.0000  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0            0.000
RP-1              Mehta et.al. AIAA 95-2962 1995. Hcomb(high) = 19923.BTU/#
 0 gll/00 C   1.00H   1.95    0.00    0.00    0.00 1   13.9761830     -24717.700
    298.150      0.0000  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0            0.000
```

To install RocketCycles, do:
pip3 install git+https://github.com/janstruzinski/RocketCycles.git

## Disclaimer

I wrote this whole library by myself. LLM was only used for the generation of the documentation below.

## Model Documentation

RocketCycles is a steady-state, one-dimensional system model. Component states and powers are calculated in
`cycle_functions.py`; fluid properties are represented by `RocketCycleFluid` in `fluid.py`; and complete engine
architectures and the cycle-sizing solver are assembled in `cycle_classes.py`. PyFluids/CoolProp supplies real-fluid
states for the pure propellants used by `FFSC_LRE` and `ORSC_LRE`, NASA polynomial data supplies species enthalpy and
heat capacity, and RocketCEA supplies equilibrium chemistry and rocket performance.

Pressure-drop inputs named `dP_over_Pinj_*` are ratios of pressure drop to downstream injector-face pressure. Therefore,
the injector pressure used throughout the cycle models is

$$P_{inj}=\frac{P_{upstream}}{1+(\Delta P/P_{inj})}.$$

Only the explicitly specified pump, cooling-channel, injector/catalyst, finite-area combustor, turbine, diffuser and
manifold effects are included. The model does not independently calculate pipe friction, valve losses, heat transfer,
shaft mechanical losses or transient behavior. Cooling-channel pressure drop and temperature rise, film-flow fraction,
component efficiencies, turbine axial velocities and pressure-recovery factors are user inputs.

### Pump

`calculate_state_after_pump_for_pyfluids()` is used by the FFSC and ORSC models. It calls PyFluids'
`compression_to_pressure()` at the requested outlet pressure and isentropic efficiency. The returned specific pump work
is the real-fluid enthalpy rise,

$$w_p=h_{out}-h_{in},$$

and every cycle obtains pump shaft power from

$$\mathcal{P}_p=\dot m_p w_p.$$

`calculate_state_after_pump()` provides a separate liquid model for `RocketCycleFluid` and is used by
`ClosedCatalyst_LRE`. Compression is decomposed into isothermal compression followed by isobaric heating due to pump
inefficiency. With inlet density $\rho_{in}$, liquid bulk modulus $K$, volumetric expansion coefficient $\alpha$ and
pressure rise $\Delta P$, the intermediate isothermal density is

$$\rho_{out,isoth}=\frac{\rho_{in}}{1-\Delta P/K}.$$

The useful and actual specific works are

$$w_{useful}=\frac{P_{out}}{\rho_{out,isoth}}-\frac{P_{in}}{\rho_{in}}, \qquad w_p=\frac{w_{useful}}{\eta_p}.$$

The dissipated work is converted to a temperature increase using constant frozen heat capacity, and thermal expansion
then corrects the density:

$$\Delta T=\frac{w_p-w_{useful}}{c_p}, \qquad \rho_{out}=\frac{\rho_{out,isoth}}{1+\alpha\Delta T}.$$

The outlet total pressure is $P_{in}+\Delta P$. Static and total pressure are treated as equal for these liquid states.
The model therefore depends on the supplied density, bulk modulus, expansion coefficient and heat capacity; it does not
model pump geometry or cavitation.

### Preburner

`calculate_state_after_preburner()` handles either a fuel/oxidizer mixture at a supplied mixture ratio or a single
monopropellant, as used for the catalyst bed. It registers the inlet `RocketCycleFluid.CEA_card` strings with RocketCEA
so the reactant composition, temperature and enthalpy enter the equilibrium calculation. The supplied contraction ratio
`CR` is passed as RocketCEA's finite-area combustor ratio `fac_CR`; this represents the preburner's cross-sectional size
and introduces the Rayleigh-line pressure loss.

RocketCEA calculates the equilibrium combustion temperature, product composition, equilibrium heat capacity,
viscosity, chamber Mach number and sonic velocity at the specified injector pressure. Product static velocity is
$V=Ma$. The preburner plenum pressure is recovered from RocketCEA's injector plane-to-combustor pressure ratio:

$$P_{plenum}=\frac{P_{inj}}{(P_{inj}/P_{comb})_{CEA}}.$$

The product species returned above a mass-fraction threshold of $10^{-6}$ are normalized by
`reformat_CEA_mass_fractions()` and stored in a gas `RocketCycleFluid`. Mixtures with $O/F\geq1$ are tagged as
oxidizer-rich and mixtures below one as fuel-rich. A monopropellant is assigned an artificial $O/F$ solely for this tag;
RocketCEA does not use that value in the monopropellant calculation.

The RocketCEA chamber temperature and plenum pressure are static quantities. The code assigns the calculated product
velocity and obtains total temperature and total pressure from the frozen-mixture relations in `RocketCycleFluid`:

$$T_t=T_s+\frac{V^2}{2c_{p,frozen}}, \qquad P_t=\frac{P_s}{(T_s/T_t)^{\gamma/(\gamma-1)}}.$$

The preburner calculation assumes equilibrium chemistry in the combustor and neglects injector Joule-Thomson heating.
No explicit combustion-efficiency factor is applied. In cycle sizing, the required preburner total temperature is met
by changing crossflow mass flow and repeatedly calling this function with a new mixture ratio.

### Turbine

`calculate_state_after_turbine()` matches turbine power to the pump power demanded by the relevant shaft. Turbine flow
is assumed to have frozen composition because of its short residence time. Axial velocity is prescribed and held
constant through the turbine, and the outlet velocity is assumed purely axial; consequently, inlet and outlet kinetic
terms cancel in the total-enthalpy balance.

The preburner outlet may have a different velocity from the turbine design velocity. The code first finds a turbine
inlet static temperature with `scipy.optimize.toms748()` such that `calculate_total_temperature()` reproduces the
preburner total temperature at the prescribed turbine axial velocity. Total pressure is retained and the corresponding
static pressure is calculated isentropically.

For turbine shaft power $\mathcal{P}_t$ and gas mass flow $\dot m_t$, the extracted mass-specific work is
$\mathcal{P}_t/\dot m_t$. The implementation converts it to a molar basis with mixture molecular weight $MW$ and sets
the target frozen outlet enthalpy to

$$w_{t,molar}=\frac{\mathcal{P}_t}{\dot m_t}MW, \qquad h_{s,out}=h_{s,in}-w_{t,molar}.$$

A second TOMS 748 root solve finds the outlet static temperature whose NASA-polynomial mixture enthalpy equals this
target. This enthalpy-based solution avoids assuming a constant heat capacity in the energy balance. An average molar
heat capacity and heat-capacity ratio are then evaluated over the expansion:

$$\bar c_p=\frac{h_{s,in}-h_{s,out}}{T_{s,in}-T_{s,out}}, \qquad \bar\gamma=\frac{\bar c_p}{\bar c_p-R}.$$

The required total-to-total turbine pressure ratio follows the polytropic expansion relation

$$\beta_{tt}=\frac{P_{t,in}}{P_{t,out}}=\left(\frac{T_{t,in}}{T_{t,out}}\right)^{\bar\gamma/[\eta_{poly}(\bar\gamma-1)]}.$$

Thus $P_{t,out}=P_{t,in}/\beta_{tt}$. The supplied diffuser/manifold pressure-recovery factor sets the downstream static
pressure as $P_{s,out}=f_{rec}P_{t,out}$. Finally, `RocketCycleFluid.equilibrate()` restores chemical equilibrium at
constant pressure and enthalpy for the slowed gas before it enters the main combustion-chamber manifold. The cycle
models equate turbine shaft power directly to pump demand; a separate gearbox or shaft-efficiency model is not present.

### Combustion Chamber

`calculate_combustion_chamber_performance()` receives the core oxidizer and fuel streams, optional film flow, injector
pressure, chamber contraction and nozzle expansion ratios, and $c^\ast$ and thrust-coefficient efficiencies. It always runs
a core-flow RocketCEA case. If film cooling is present, it can also form a combined CEA reactant card by mass-weighting
the coolant and same-side core stream; `include_film_in_cstar` chooses whether core-only or core-plus-film CEA results
set performance.

Cooling-channel heat pickup is kept in the cycle energy balance. The prescribed heat absorbed by the coolant is split
between the core reactants in proportion to core mixture ratio. Each reactant's inlet temperature in its CEA card is
reduced by $\Delta T=q/(\dot m c_p)$ before combustion is evaluated. Cooling-channel state functions themselves use
the prescribed pressure drop and temperature rise, remove film mass from the channel outlet, and calculate

$$\dot Q_{coolant}=\dot m_{channel,out}(h_{out}-h_{in}).$$

RocketCEA supplies ideal vacuum specific impulse, characteristic velocity, equilibrium combustion temperature,
injector-to-plenum pressure ratio and sea-level performance/separation mode. Chamber plenum pressure is calculated with
the same finite-area-combustor pressure-ratio method as the preburner. If film is excluded from CEA performance, ideal
specific impulse is diluted by the core-to-total mass-flow ratio. Throat and exit areas are

$$A_t=\frac{c^\ast_\mathrm{CEA}\dot m_\mathrm{total}\eta_{c^\ast}}{P_\mathrm{plenum}}, \qquad A_e=\varepsilon A_t.$$

Real vacuum performance is calculated as

$$I_{sp,vac}=I_{sp,vac,ideal}\eta_{c^\ast}\eta_{C_f}, \qquad F_{vac}=I_{sp,vac}\dot m_{total}g_0.$$

The difference between ideal and real vacuum specific impulse is treated as an absolute loss and subtracted from
RocketCEA's ideal sea-level specific impulse. Sea-level thrust is then $F_{sea}=I_{sp,sea}\dot m_{total}g_0$. This is a
system-level performance model: RocketCEA supplies equilibrium chemistry and one-dimensional nozzle performance, while
the supplied efficiencies represent nonideal combustion and thrust losses. Detailed injector geometry, atomization,
finite-rate chemistry, chamber heat transfer and nozzle contour losses are not resolved.

### Cycle

`Cycle` is the superclass in `cycle_classes.py`. Its constructor stores the inputs shared by the engine architectures:
propellant definitions and inlet states, overall mixture ratio, efficiencies, pressure-drop ratios, finite-area
contraction ratios, nozzle expansion ratio, film and cooling parameters, turbine velocities and recovery factors. It
also initializes a common set of result attributes for mass flows, fluid states, powers, pressures, temperatures,
pressure ratios, areas, thrust and CEA text output.

The superclass does not implement `analyze_cycle()`. `FFSC_LRE`, `ORSC_LRE` and `ClosedCatalyst_LRE` inherit its data
layout and each implement the component order and flow routing for one architecture. An analysis mutates the cycle
object with its current solution. `get_full_output()` reads those attributes and returns a report, optionally including
the saved RocketCEA outputs.

All architectures first split total propellant flow using the overall mixture ratio:

$$\dot m_f=\frac{\dot m_{total}}{1+O/F}, \qquad \dot m_o=(O/F)\dot m_f.$$

Film and crossflow quantities are fractions of their corresponding main propellant flow. At each injector, the smaller
available upstream pressure is used when two streams must enter together. Cooling and film mass are tracked explicitly,
and turbine products are equilibrated before the final chamber calculation. The following subsections give the exact
execution order used by each subclass.

### FFSC

`FFSC_LRE.analyze_cycle()` represents a full-flow staged-combustion engine with a fuel-rich preburner/turbine branch and
an oxidizer-rich preburner/turbine branch.

1. Total flow is split into fuel and oxidizer; fuel-film flow and the specified fuel crossflow are calculated.
2. `calculate_state_after_pump_for_pyfluids()` pumps all fuel and supplies fuel-pump work and power.
3. `calculate_state_after_cooling_channels_for_Pyfluids()` heats the fuel, applies the prescribed pressure loss, removes
   film flow and calculates coolant heat pickup. Pumped and heated fuel are converted to `RocketCycleFluid` objects.
4. Fuel-preburner injector pressure is the heated-fuel pressure divided by its specified pressure-drop factor.
5. The oxidizer is pumped. In direct analysis, oxidizer pump rise and oxidizer crossflow are inputs. In the sizing form
   of `analyze_cycle()`, oxidizer pump rise is set so its outlet pressure matches heated-fuel pressure, and TOMS 748
   varies oxidizer crossflow until repeated `calculate_state_after_preburner()` calls make fuel-preburner total
   temperature equal `T_FPB_required`.
6. The remaining non-film, non-crossflow fuel and oxidizer crossflow burn in the fuel-rich preburner. Their combined
   products pass through the fuel turbine, whose required power is the fuel-pump power.
7. Remaining oxidizer and fuel crossflow burn in the oxidizer-rich preburner. Its injector pressure is based on the
   smaller of pumped-fuel and pumped-oxidizer pressure.
8. Those products pass through the oxidizer turbine, whose required power is the oxidizer-pump power.
9. Equilibrated fuel- and oxidizer-turbine products enter the main chamber. Injector pressure is based on the lower
   manifold pressure, and `calculate_combustion_chamber_performance()` evaluates the chamber, film and nozzle.

When `CycleSizing` is used, fuel crossflow fraction is an outer-loop variable used to meet oxidizer-preburner
temperature, while the nested solve above determines oxidizer crossflow from fuel-preburner temperature. This separates
the two preburner-temperature constraints.

### ORSC

`ORSC_LRE.analyze_cycle()` represents an oxygen-rich staged-combustion engine with one oxidizer-rich preburner and one
turbine driving all pumps.

1. Total flow is split, fuel-film flow is assigned, and all oxidizer is pumped.
2. Oxidizer-preburner injector pressure is calculated from oxidizer-pump outlet pressure.
3. All fuel passes through the main fuel pump. If its outlet pressure is below oxidizer-pump outlet pressure, the
   preburner crossflow state is raised to the oxidizer pressure with the booster fuel pump; booster power is based only
   on crossflow mass.
4. In direct analysis, fuel crossflow fraction is supplied. In the sizing form of `analyze_cycle()`, TOMS 748 changes
   crossflow fuel and repeatedly calls `calculate_state_after_preburner()` until oxidizer-preburner total temperature
   equals `T_OPB_required`.
5. All oxidizer and crossflow fuel react in the oxidizer-rich preburner. Their products pass through the oxidizer
   turbine, which supplies oxidizer-pump, main-fuel-pump and booster-pump power.
6. Fuel not diverted as crossflow enters the cooling channels; film is removed from the channel outlet and heat pickup
   is calculated.
7. Equilibrated turbine products and heated fuel enter the main chamber. Injector pressure is based on the lower of
   oxidizer-manifold and heated-fuel pressure before chamber performance is calculated.

### ClosedCatalyst

`ClosedCatalyst_LRE.analyze_cycle()` represents a closed catalyst-decomposition cycle, such as a peroxide cycle. Unlike
the other subclasses, its inlet propellants are supplied directly as `RocketCycleFluid` objects so mixtures such as
aqueous hydrogen peroxide and fuels without complete CoolProp coverage can be represented.

1. Total flow is split and oxidizer film flow is assigned.
2. `calculate_state_after_pump()` pumps the oxidizer using the custom liquid model.
3. Oxidizer passes through the cooling channels, where the film fraction is removed and coolant heat pickup is found.
4. Catalyst outlet pressure is obtained from the combined injector/catalyst pressure-drop ratio.
   `calculate_state_after_preburner()` is called in monopropellant mode to calculate equilibrium decomposition products
   using the catalyst finite-area contraction ratio.
5. `calculate_state_after_pump()` pumps the fuel. The catalyst-product turbine then supplies both oxidizer- and
   fuel-pump power and carries the non-film oxidizer flow.
6. Equilibrated turbine products and pumped fuel enter the main chamber. Injector pressure is based on the lower stream
   pressure, and chamber performance is calculated with oxidizer film and cooling heat included.

The catalyst is thus modeled thermochemically as an equilibrium, constant-enthalpy monopropellant RocketCEA chamber;
reaction kinetics, catalyst loading, decomposition efficiency and bed geometry beyond the supplied pressure loss and
finite-area ratio are not independently modeled.

### CycleSizing

`CycleSizing` wraps an existing cycle object and uses `scipy.optimize.least_squares()` to satisfy three system-level
constraints. Exactly one required thrust condition—vacuum or sea level—must be supplied, together with required chamber
plenum pressure. The third constraint depends on architecture.

For FFSC, the outer unknown vector is

$$\mathbf{x}_{FFSC}=[\dot m_{total},\;\dot m_{f,cross}/\dot m_f,\;\Delta P_{FP}],$$

and the normalized residuals are required thrust, chamber plenum pressure and oxidizer-preburner total temperature. On
every outer function evaluation, `FFSC_LRE.analyze_cycle()` performs its nested solve for oxidizer crossflow from the
required fuel-preburner temperature and derives oxidizer-pump pressure rise from pressure matching.

For ORSC and ClosedCatalyst, the outer unknown vector is

$$\mathbf{x}_{ORSC,CC}=[\dot m_{total},\;\Delta P_{OP},\;\Delta P_{FP}].$$

Their residuals are required thrust, chamber plenum pressure and the pressure difference between pumped fuel and the
equilibrated turbine outlet in the chamber manifold. For ORSC, each outer evaluation also invokes the nested fuel-
crossflow solve that meets required oxidizer-preburner temperature. ClosedCatalyst has no preburner-temperature inner
solve.

Variables are normalized by the supplied initial estimates, bounds are divided by the same reference vector, and
residuals are normalized by required thrust, required chamber pressure and—where applicable—required preburner
temperature. This scaling makes quantities with different units comparable to the least-squares solver. By default the
implementation uses a three-point numerical Jacobian, the bounded `dogbox` method, `soft_l1` loss and the exact trust-
region solver. Solver options, tolerances, finite-difference step, bounds and inner TOMS 748 brackets are exposed as
constructor arguments.

The solver repeatedly calls and overwrites the same `Cycle` subclass instance. `result` stores SciPy's optimizer result,
`x` stores the dimensional solution, and `get_residuals()` reports initial and final dimensional and normalized
residuals. The converged cycle is available as `CycleSizing.cycle` and can be passed to `get_full_output()`.

### RocketCycleFluid

`RocketCycleFluid` is the common state and composition container in `fluid.py`. It stores species names, mass fractions,
static temperature, propellant type and a gas-like or liquid-like phase. It also stores or derives static/total pressure,
velocity, total temperature, density, viscosity, molecular weight, enthalpy, frozen heat capacity, heat-capacity ratio
and a RocketCEA reactant card. `pyfluid_to_rocket_cycle_fluid()` converts a PyFluids pure-fluid state while preserving
temperature, pressure, density and heat capacity.

`get_mixture_thermal_properties()` obtains species enthalpy, heat capacity, molecular weight and formula from
`nasaPoly`. For liquefied gases and hydrocarbons without a usable polynomial temperature range, the caller supplies
species molar heat capacity and enthalpy is extrapolated from the database reference/boiling temperature with constant
heat capacity. The generated RocketCEA card includes each species' mass fraction, enthalpy and current temperature.

Mass fractions $Y_i$ are converted to mixture molecular weight and mole fractions by

$$MW_{mix}=\left(\sum_i\frac{Y_i}{MW_i}\right)^{-1}, \qquad X_i=MW_{mix}\frac{Y_i}{MW_i}.$$

Under ideal-mixture, frozen-composition assumptions, mixture properties are mole-fraction weighted:

$$c_{p,molar}=\sum_iX_ic_{p,i}, \qquad h_{molar}=\sum_iX_ih_i, \qquad c_{p,mass}=\frac{c_{p,molar}}{MW_{mix}}.$$

For gas states, $c_v=c_p-R$ and $\gamma=c_p/c_v$. `calculate_total_temperature()`,
`calculate_static_from_total_pressure()` and `calculate_total_from_static_pressure()` use the calorically perfect,
isentropic frozen-mixture relations shown above; the methods reject liquid-phase use.

`calculate_gas_density()` augments the ideal-gas law with a simple mixture compressibility factor. CoolProp/PyFluids
provides $Z_i$ for CO, CO2, H2O, CH4, H2 and O2. Unavailable gas species use $Z=1$, while graphite and liquid water are
treated as occupying negligible gas volume. The mole-fraction-weighted value gives

$$\rho=\frac{P_sMW_{mix}}{Z_{mix}RT_s}.$$

Finally, `equilibrate()` rewrites the current CEA card as a monopropellant and asks RocketCEA for a new equilibrium at
the fluid's pressure and enthalpy. It returns a new `RocketCycleFluid` with equilibrium species, temperature, heat
capacity and viscosity plus the full CEA output. This operation is used after turbine expansion to represent chemical
relaxation in the downstream diffuser/manifold; the turbine expansion itself remains frozen.
