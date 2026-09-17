class AtmosphereParameters:
    g = 9.80665 # Standard Gravity [m/s^2]
    rho0 = 1.268 # Sea-Level Standard Density [kg/m^3]
    temp0 = 288.15 # Sea-Level Standard Temperature [K]
    Press0 = 101325.0 # Sea-Level Standard Pressure [Pa]
    R = 287.05 # Specific gas constant for air [J/(kg*K)]
    L = 0.0065 # Temperature lapse rate [K/m]

# Calculate standard atmoshpheric temperature at altitude (m)
def temperature(altitude):
    return AtmosphereParameters.temp0 - AtmosphereParameters.L * altitude

# Calculate standard atmospheric pressure at altitude (m)
def pressure(altitude):
    T = temperature(altitude)

    return AtmosphereParameters.Press0 * (T / AtmosphereParameters.temp0) ** (AtmosphereParameters.g / (AtmosphereParameters.R * AtmosphereParameters.L))

# Calculate standard atmospheric density at altitude (m)
def density(altitude):
    T = temperature(altitude)
    P = pressure(altitude)

    return P / (AtmosphereParameters.R * T)

