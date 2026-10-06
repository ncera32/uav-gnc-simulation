from src.aerodynamics import aerosonde_longitudinal_forces_moments, aerosonde_lateral_directional_forces_moments

from src.propulsion import aerosonde_propulsion_forces_moments

# Calculate total non-gravitational body-frame forces and moments acting on the aircraft
def aircraft_forces_moments(
        state,
        delta_e,
        delta_a,
        delta_r,
        delta_t,
        aircraft
):
    
    forces_long, moments_long = (
        aerosonde_longitudinal_forces_moments(
            state = state,
            delta_e = delta_e,
            aircraft = aircraft
        )
    )

    forces_lat, moments_lat = (
        aerosonde_lateral_directional_forces_moments(
            state = state,
            delta_a = delta_a,
            delta_r = delta_r,
            aircraft = aircraft
        )
    )

    forces_prop, moments_prop = (
        aerosonde_propulsion_forces_moments(
            state = state,
            delta_t = delta_t,
            aircraft = aircraft
        )
    )

    forces_body = (
        forces_long
        + forces_lat
        + forces_prop
    )

    moments_body = (
        moments_long
        + moments_lat
        + moments_prop
    )

    return forces_body, moments_body 





