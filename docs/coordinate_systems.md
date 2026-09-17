# Coordinate Systems and State Conventions

## Navigation Frame

The simulation uses a local North-East-Down (NED) navigation frame.

- +x_n: North
- +y_n: East
- +z_n: Down

Aircraft position is represented by:

[pn, pe, pd]

where pn, pe, and pd are measured in meters.

Altitude above the local NED origin is defined as:

h = -pd


## Body Frame

The aircraft body-fixed frame is centered at the aircraft center of gravity.

- +x_b: Forward through the nose
- +y_b: Right wing
- +z_b: Down

Body-frame translational velocity is represented by:

[u, v, w]


## Attitude

Aircraft attitude relative to the NED frame is represented using
3-2-1 Euler angles:

- phi: Roll [rad]
- theta: Pitch [rad]
- psi: Yaw/heading [rad]

Positive rotations follow the right-hand rule.


## Angular Rates

Body angular velocity is represented by:

[p, q, r]

where:

- p: Roll rate about x_b [rad/s]
- q: Pitch rate about y_b [rad/s]
- r: Yaw rate about z_b [rad/s]


## State Vector

The 12-state rigid-body aircraft state is:

x = [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]^T


## Units

The simulation uses SI units.

- Distance: meters [m]
- Time: seconds [s]
- Velocity: meters per second [m/s]
- Mass: kilograms [kg]
- Force: Newtons [N]
- Moment: Newton-meters [N*m]
- Angles: radians [rad]
- Angular rates: radians per second [rad/s]