from rocketcea.cea_obj_w_units import CEA_Obj
import rocketcea.cea_obj as rcea
import nasaPoly
import numpy as np
import pyfluids
import warnings
import re


def pyfluid_to_rocket_cycle_fluid(fluid, CEA_name, type, phase):
    """A function to create RocketCycleFluid object from PyFluid Fluid.

    :param pyfluids.Fluid fluid: PyFluid Fluid object.
    :param str CEA_name: String representing CEA equivalent name of Fluid (for example: "CH4" in case "Methane" fluid).
    :param str type: "oxid" for oxidizer or "fuel" for fuel.
    :param str phase: String describing phase of Fluid. "gas" of gas-like properties or "liquid" for liquid-like
     properties.

    :return: RocketCycleFluid object equivalent of Fluid passed.
    """

    propellant = RocketCycleFluid(species=[CEA_name], mass_fractions=[1], type=type,
                                  temperature=fluid.temperature + 273.15, phase=phase)
    propellant.Ps = fluid.pressure / 1e5
    # If phase is liquid, total pressure is the same as static pressure
    if phase == "liquid":
        propellant.Pt = propellant.Ps
    propellant.density = fluid.density

    return propellant


def reformat_CEA_mass_fractions(mass_fractions):
    """A function to reformat mass fractions dictionary given by RocketCEA.

    :param dict mass_fractions: A dictionary with mass fractions obtained from RocketCEA.

    :return: A reformatted and consistent dictionary without "*".
    """

    # First remove the asterix in species names if present
    mass_fractions = {species.replace('*', ''): mf[1] for species, mf in mass_fractions.items()}

    # Second remove any species with zero mass fraction
    mass_fractions = {species: mf for species, mf in mass_fractions.items() if mf != 0.0}

    # Since some smaller mass fractions are not used, we need to make the dictionary consistent by changing the highest
    # mass fraction to be equal (1 - mf of all the other species)
    largest_species = max(mass_fractions, key=lambda species: mass_fractions[species])
    consistent_mf = 1 - (sum(mass_fractions.values()) - max(mass_fractions.values()))
    mass_fractions[largest_species] = consistent_mf

    return mass_fractions


class RocketCycleFluid:
    def __init__(self, species, mass_fractions, temperature, type, phase, volumetric_expansion_coefficient=0,
                 liquid_elasticity=10**16, density=None, species_molar_Cp=None):
        """
        A class to store and calculate mixture thermophysical properties bases on NASA 9 polynomials

        :param list species: A list with strings representing species' CEA names.
        :param list mass_fractions: A list with floats representing species' mass fractions (from 0 to 1).
        :param float or int temperature: A float representing fluid static temperature (in K).
        :param str type: "oxid" for oxidizer or "fuel" for fuel, "name" if neither of these
        :param str phase: String describing phase of Fluid. "gas" of gas-like properties or "liquid" for liquid-like
            properties. "gas"/"liquid" should be used also for supercritical phase, since at certain conditions even
            then it resembles one of these
        :param float or int volumetric_expansion_coefficient: An optional float representing volumetric expansion
            coefficient (in 1/K) to calculate a new density of the fluid after it passes the pump. Only for "liquid"
            phase, nominally set to 1.
        :param float or int liquid_elasticity: An optional float representing liquid elasticity (in N/m^2) to calculate
            a new density of the fluid after it passes the pump. Only for "liquid" phase, nominally set to 1.
        :param float or int density: An optional float to represent assigned fluid density (in kg/m^3). Only for
            "liquid" phase, for gas it can be calculated.
        :param list species_molar_Cp: A list of molar heat capacities (in J/mol-K) for individual species needs to be
         provided if there is any liquified gas, like O2(L), CH4(L), H2(L) or C3H8(L), and some hydrocarbons like RP-1
          or JP-10(L). This is because NASA 9 Polynomials do not provide them. For any species that are not liquified
           gases or mentioned hydrocarbons, corresponding entry in the list can be anything, as Cp will be then obtained
            from NASA 9 Polynomials.
        """

        # Assign properties
        self.species = species
        self.mass_fractions = np.array(mass_fractions)  # (-) from 0 to 1
        self.Ts = temperature                           # K
        self.type = type
        self.R = 8.31446                                # J / (K * mol)
        self.phase = phase

        # Define other properties
        self.velocity = None                # m/s
        self.Pt = None                      # bar
        self.Ps = None                      # bar
        self.Tt = None                      # K
        self.viscosity = None               # milipoise
        self.mass_Cp_equilibrium = None     # J / (kg * K)

        # If a list of heat capacities for individual species was not provided, create a zero array instead of it.
        # Otherwise, make it a property. This is done to handle situations in which NASA 9 Poly does not provide any Cp.
        if species_molar_Cp is None:
            self.species_molar_Cp = np.zeros(len(species))
        else:
            self.species_molar_Cp = species_molar_Cp

        # Generate CEA card, calculate enthalpy, Cp, gamma for ideal fluid (no enthalpy of mixing, frozen conditions)
        if self.phase == "gas":
            (self.CEA_card, self.h0, self.molar_Cp_frozen, self.mass_Cp_frozen, self.gamma, self.MW,
             self.molar_fractions) = self.get_mixture_thermal_properties()

        # In case of liquid, assign properties for liquid as well
        else:
            self.volumetric_expansion_coefficient = volumetric_expansion_coefficient
            self.liquid_elasticity = liquid_elasticity
            self.density = density
            self.CEA_card, self.h0, self.molar_Cp_frozen, self.mass_Cp_frozen, self.MW, self.molar_fractions = (
                self.get_mixture_thermal_properties())

    def check_gas_phase(self):
        """A function to check if assigned fluid phase is gas. Used to rise an error if "gas" dedicated methods are
        used for liquids."""

        # If phase is not gas, rise an error with description
        if not self.phase == "gas":
            warnings.simplefilter("error", UserWarning)
            warnings.warn("Method for gases used for liquids")

    def calculate_total_temperature(self):
        """A function to calculate total temperature based on fluid velocity."""

        # Raise an error if method is used for fluids other than gases
        self.check_gas_phase()
        # Calculate fluid total temperature. Beware that self.velocity is None by default, therefore this property
        # needs to be changed before calling this method,
        self.Tt = self.Ts + (self.velocity ** 2) / (2 * self.mass_Cp_frozen)  # K

    def calculate_static_from_total_pressure(self):
        """A function to calculate static from total pressure"""

        # Raise an error if method is used for fluids other than gases
        self.check_gas_phase()
        # Beware that total pressure needs to be assigned before the function is called
        self.Ps = self.Pt * (self.Ts / self.Tt) ** (self.gamma / (self.gamma - 1))  # bar

    def calculate_total_from_static_pressure(self):
        """A function to calculate total from static pressure"""

        # Raise an error if method is used for fluids other than gases
        self.check_gas_phase()
        # Beware that static pressure needs to be assigned before the function is called
        self.Pt = self.Ps / ((self.Ts / self.Tt) ** (self.gamma / (self.gamma - 1)))  # bar

    def calculate_gas_density(self):
        """A function to calculate gas density based on ideal gas law while taking liquids/solids species into account
         and supplementing the model by compressibility factors for available species from PyFluid."""

        # Raise an error if method is used for fluids other than gases
        self.check_gas_phase()

        # The compressibility factors of species that exist in PyFluid will be obtained,
        # after which compressibility factor of the total mixture will be calculated and used to improve ideal gas
        # density for real gas effects. It is assumed that non-gas species take negligible volume (and have
        # compressibility factor of zero).

        # This method is used because it allows to account for any solids in the mixture as they cannot be
        # incorporated in PyFluid. Furthermore, it easily allows to account for any water liquid content, which would
        # be more difficult with PyFluid (as enthalpy would need to be used, and CEA enthalpy is different from the
        # one in PyFluid), and it is still compatible with minor species that are not in PyFluid (these are just
        # assumed to follow ideal gas law, so have compressibility factor of 1).

        # Create dictionaries to store molar fractions and compressibility factors of PyFluid species
        PyFluid_molar_fractions = {"CO": 0.0, "CO2": 0.0, "H2O": 0.0, "CH4": 0.0, "H2": 0.0, "O2": 0.0}
        PyFluid_Z_factors = {"CO": 0.0, "CO2": 0.0, "H2O": 0.0, "CH4": 0.0, "H2": 0.0, "O2": 0.0}
        non_gas_molar_fraction = 0

        # Create a map from CEA species names to PyFluid objects representing them
        species_to_PyFluid = {"CO": pyfluids.FluidsList.CarbonMonoxide, "CO2": pyfluids.FluidsList.CarbonDioxide,
                               "H2O": pyfluids.FluidsList.Water, "CH4": pyfluids.FluidsList.Methane,
                               "H2": pyfluids.FluidsList.Hydrogen, "O2": pyfluids.FluidsList.Oxygen}

        # Iterate over species
        for (species, value) in zip(self.species, self.molar_fractions):
            # If species is in PyFluid, put its molar fraction value and compressibility factor in the respective
            # dictionaries
            if species in list(PyFluid_molar_fractions.keys()):
                PyFluid_molar_fractions[species] = value
                # Create PyFluid Fluid object to get its compressibility factor
                gas = pyfluids.Fluid(species_to_PyFluid[species]).with_state(
                    pyfluids.Input.pressure(self.Ps * 1e5), pyfluids.Input.temperature(self.Ts - 273.15))
                PyFluid_Z_factors[species] = gas.compressibility
            # If species is not gas phase, change molar fraction of non gas species. Only graphite and water are
            # considered, because they are the only such species in preburner products.
            elif species in ["C(gr)", "H2O(L)"]:
                non_gas_molar_fraction += value

        # Calculate molar fraction of gas species that are not in PyFluid. These are assumed to have compressibility
        # of 1.
        non_PyFluid_molar_fraction = 1 - non_gas_molar_fraction - sum(PyFluid_molar_fractions.values())

        # Calculate compressibility factor. It is a molar fraction weighted sum of compressibility factors. Species in
        # that are not in PyFluid have compressibility factor of 1 and non-gas species have compressibility factor of 0.
        # Dictionary keys and values are changed to lists and then arrays for array operation.
        mixture_compressibility = (np.sum(np.array(list(PyFluid_molar_fractions.values())) *
                                          np.array(list(PyFluid_Z_factors.values()))) + non_PyFluid_molar_fraction * 1
                                   + non_gas_molar_fraction * 0)

        # Calculate density based on ideal gas law and taking into account mixture compressibility factor
        density = self.Ps * 1e5 * (self.MW * 1e-3) / (mixture_compressibility * self.R * self.Ts)  # kg/m^3
        return density

    def get_mixture_thermal_properties(self):
        """A function to calculate mixture thermophysical properties based on NASA 9 polynomials for each species.

        :return: CEA card, mixture enthalpy (kJ/mol), mixture molar (J/(mol * K)) and mass specific heats
            (J/(kg * K)), mixture molecular weight (g / mol). In case of "gas" phase, also mixture heat ratio and
             species molar fractions
        """

        # Create lists to store the results
        h0_list = []    # kJ / (mol * kg)
        Cp_list = []    # J / (mol * K)
        MW_list = []    # g / mol
        CEA_card = ""

        # Iterate over species to retrieve their thermal properties
        for (name, mf, Cp) in zip(self.species, self.mass_fractions, self.species_molar_Cp):
            species = nasaPoly.Species(name)
            # Some of the liquified gases in NASA 9 polynomials are only able to return assigned enthalpy at boiling
            # temperature. In this case, to calculate their new enthalpy at other temperature, Cp needs to be provided
            # and used.
            # First, if possible from NASA 9 polynomials, calculate enthalpy and Cp. This will overwrite Cp = 0.
            if not species.T_ranges == []:
                h0 = species.h_0(self.Ts) * 1e-3   # kJ / mol
                Cp = species.cp_0(self.Ts)         # J / (mol * K)
            # Now, calculate enthalpy and Cp if it is not possible from NASA 9 Polynomials. This will keep Cp as is.
            else:
                # Get enthalpy of formation at boiling temperature
                h0_formation = species.h_f_0 * 1e-3  # kJ/mol
                # Now, find their boiling temperature. This can be done by getting raw data from NASA 9 Polynomials,
                # and taking the first number in the second line
                # Get data entry string
                raw_data = species.raw_entry[0]
                # Get second line
                raw_data = raw_data.split("\n", 2)[2]
                # Get the first number in the string and change it to float
                T_boiling = float(re.search(r'\d+\.\d+', raw_data).group())     # K
                # Calculate the new enthalpy
                h0 = h0_formation + (self.Ts - T_boiling) * Cp * 1e-3   # kJ/mol
            MW = species.molecular_wt          # g / mol
            chemical_formula = species.chem_formula
            # Chemical formula in NASA Poly are a bit broken - sometimes there is no space delimiter
            # before the capital letter and there are multiple zeros at the end of formula. This needs to be fixed
            # Add the space
            chemical_formula = re.sub(r"(\w)([A-Z])", r"\1   \2", chemical_formula)
            # Remove the zeros. There must be a space before 0.00 such that, for example, 10.00 is not stripped of
            # zeros.
            chemical_formula = re.sub(' 0.00', '', chemical_formula)
            # Strip whitespaces at the end and the beginning
            chemical_formula = chemical_formula.strip()
            # For each species create card string and add it to CEA card string
            species_string = (f"{self.type}   {name}   {chemical_formula}   wt%={100 * mf}\n"
                              f"h,kj/mol={h0}   t,k={self.Ts}\n")
            CEA_card += species_string
            # Append arrays with retrieved results
            h0_list.append(h0)
            Cp_list.append(Cp)
            MW_list.append(MW)

        # Change lists to arrays, so that they can be operated on
        h0_array = np.array(h0_list)
        Cp_array = np.array(Cp_list)
        MW_array = np.array(MW_list)

        # Calculate molar fractions
        mixture_MW = 1 / (np.sum(self.mass_fractions / MW_array))      # g / mol
        molar_fractions = mixture_MW * self.mass_fractions / MW_array

        # Calculate mixture thermal properties (ideal fluid assumption, frozen conditions)
        molar_Cp_mixture = np.sum(molar_fractions * Cp_array)    # J / (mol * K)
        h0_mixture = np.sum(molar_fractions * h0_array)          # kJ / (mol * kg)
        mass_Cp_mixture = (molar_Cp_mixture / (mixture_MW * 1e-3))   # J / (kg * K)

        # Return the results if mixture is not a gas
        if not self.phase == "gas":
            return CEA_card, h0_mixture, molar_Cp_mixture, mass_Cp_mixture, mixture_MW, molar_fractions

        # If it is a gas, calculate thermal properties specific to gases and return the results
        if self.phase == "gas":
            # Calculate specific gas constant and gamma
            molar_Cv_mixture = molar_Cp_mixture - self.R    # J / (mol * K)
            gamma_mixture = molar_Cp_mixture / molar_Cv_mixture
            return CEA_card, h0_mixture, molar_Cp_mixture, mass_Cp_mixture, gamma_mixture, mixture_MW, molar_fractions

    def equilibrate(self):
        """A function to equilibrate the fluid represented by RocketCycleFluid

        :return: CEA full output, new RocketCycleFluid object with equilibrated fluid
        """

        # Fluid reaches a new equilibrium at constant pressure and enthalpy. CEA Rocket problem is essentially such
        # process. Therefore, a mixture can be input to CEA as monopropellant and chamber mass fractions and temperature
        # can be obtained to find a new chemical equilibrium. Total pressure is used, because the function is
        # intended for calculating equilibrium after turbine, where the gas slowed down a lot in the manifold.

        # Modify CEA card, such that CEA thinks it is monopropellant.
        CEA_card = self.CEA_card.replace("oxidizer", "name")
        CEA_card = CEA_card.replace("fuel", "name")

        # First create a CEA object with Imperial units, such that full output can be obtained
        rcea.add_new_propellant(name="equilibrium card", card_str=CEA_card)
        equilibrium = rcea.CEA_Obj(propName="equilibrium card")
        equilibrium_CEA_output = equilibrium.get_full_cea_output(Pc=self.Ps, pc_units="bar", output="si", short_output=1)

        # Now create a CEA object with SI units to get values expressed with them
        equilibrium = CEA_Obj(propName="equilibrium card", isp_units='sec', cstar_units='m/s',
                              pressure_units='bar', temperature_units='K', sonic_velocity_units='m/s',
                              enthalpy_units='kJ/kg', density_units='kg/m^3', specific_heat_units='kJ/kg-K',
                              viscosity_units='millipoise', thermal_cond_units='W/cm-degC')

        # Get and reformat mass fractions
        equilibrium_mass_fractions = equilibrium.get_SpeciesMassFractions(Pc=self.Ps, min_fraction=1e-6)[1]
        equilibrium_mass_fractions = reformat_CEA_mass_fractions(equilibrium_mass_fractions)

        # Get static temperature
        equilibrium_temperature = equilibrium.get_Tcomb(Pc=self.Ps)     # K

        # Get equilibrium heat capacity and viscosity
        transport_properties = equilibrium.get_Chamber_Transport(Pc=self.Ps)[0:2]
        Cp_equilibrium = transport_properties[0] * 1e3      # J / (kg * K)
        viscosity = transport_properties[1]                 # milipoise

        # Create a new RocketCycleFluid
        equilibrium_fluid = RocketCycleFluid(species=list(equilibrium_mass_fractions.keys()),
                                             mass_fractions=list(equilibrium_mass_fractions.values()),
                                             temperature=equilibrium_temperature, type=self.type, phase=self.phase)
        equilibrium_fluid.Ps = self.Ps  # bar
        equilibrium_fluid.viscosity = viscosity                 # milipoise
        equilibrium_fluid.mass_Cp_equilibrium = Cp_equilibrium  # J / (kg * K)

        # Return the new RocketCycleFluid and CEA full output
        return equilibrium_fluid, equilibrium_CEA_output
