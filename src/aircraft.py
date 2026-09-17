class AircraftParameters:
    # -------------------------------------------------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------------------------------------------------

    S = 0.55 # Wing Area [m^2]
    b = 2.9 # Wing Span [m]
    c = 0.19 # MAC [m]

    # -------------------------------------------------------------------------------------------------------------------
    # Mass Properties
    # -------------------------------------------------------------------------------------------------------------------

    m = 11.0 # UAV mass [kg]
    Ixx = 0.824 # Roll moment of inertia [kg-m^2]
    Iyy = 1.135 # Pitch moment of inertia [kg-m^2]
    Izz = 1.759 # Yaw moment of inertia [kg-m^2]
    Ixz = 0.120 # Product of inertia [kg-m^2]

    # -------------------------------------------------------------------------------------------------------------------
    # Baseline Aerodynamic Coefficients
    # -------------------------------------------------------------------------------------------------------------------

    CL0 = 0.23 # zero-angle-of-attack lift coefficient
    CD0 = 0.0424 # zero-lift/baseline drage coefficient
    CD_P = 0.043 # Parasitic drag coefficient
    CY0 = 0 # Base side force coefficient
    Cm0 = 0.0135 # Base pitching moment coefficient
    Cl0 = 0 # Base rolling moment coefficient
    Cn0 = 0 # Base yawing moment coefficient

    # -------------------------------------------------------------------------------------------------------------------
    # Stability Derivatives - Longitudinal (alpha, q)
    # -------------------------------------------------------------------------------------------------------------------
    
    # angle of attack

    CL_alpha = 5.61 # how lift changes with angle of attack
    CD_alpha = 0.132 # how drag changes with angle of attack
    Cm_alpha = -2.74 # how pitching moment changes with angle of attack

    # pitch rate 

    CL_q = 7.95 # how lift changes with pitch rate
    CD_q = 0 # how drag changes with pitch rate
    Cm_q = -38.21 # how pitching moment changes with pitch rate

    # -------------------------------------------------------------------------------------------------------------------
    # Stability Derivatives - Lateral/Directional (beta, p, r)
    # -------------------------------------------------------------------------------------------------------------------
    
    # side slip angle
    
    CY_beta = -0.83 # how side force changes with sideslip
    Cl_beta = -0.13 # how rolling moment changes with sideslip
    Cn_beta = 0.073 # how yawing moment changes with sideslip

    # roll rate

    CY_p = 0 # how side force changes with roll rate
    Cl_p = -0.51 # how rolling moment changes with roll rate
    Cn_p = -0.069 # how yawing moment changes with roll rate
    
    # yaw rate 
    
    CY_r = 0 # how side force changes with yaw rate
    Cl_r = 0.25 # how rolling moment changes with yaw rate
    Cn_r = -0.095 # how yawing moment changes with yaw rate

    # -------------------------------------------------------------------------------------------------------------------
    # Control Derivatives 
    # -------------------------------------------------------------------------------------------------------------------
    
    # Elevator

    CL_delta_e = 0.13 # effect of elevator on lift
    CD_delta_e = 0.0135 # effect of elevator on drag
    Cm_delta_e = -0.99 # effect of elevator on pitching moment

    # Aileron

    CY_delta_a = 0.075 # effect of aileron on side force
    Cl_delta_a = 0.17 # effect of aileron on rolling moment
    Cn_delta_a = -0.011 # effect of aileron on yawing moment

    # Rudder

    CY_delta_r = 0.19 # effect of rudder on side force
    Cl_delta_r = 0.0024 # effect of rudder on rolling moment
    Cn_delta_r = -0.069 # effect of rudder on yawing moment 

    # -------------------------------------------------------------------------------------------------------------------
    # Propulsion 
    # -------------------------------------------------------------------------------------------------------------------
   
    P_max = 3.87 # Max Power of Motor [kW] CHECK UNITS!!!
    V_max = 44.4 # max voltage [V]
    D_prop = 0.508 # diameter of propeller [m]
    K_V = 0.0659 # back-emf voltage constant [V-s/rad]
    K_Q = 0.0659 # motor torque constant [N-m]
    R_motor = 0.042 # resistance of motor windings [ohms]
    i0 = 1.5 # zero-torque/no-load current [Amps]

    # Max Thrust?
    # Throttle Range ?
    # Cruise thrust/power ?

    # aerodynamic coefficients for thrust and torque
    
    CQ_2 = -0.01664
    CQ_1 = 0.004970
    CQ_0 = 0.005230

    CT_2 = -0.1079
    CT_1 = -0.06044
    CT_0 = 0.09357

    # -------------------------------------------------------------------------------------------------------------------
    # Control Limits
    # -------------------------------------------------------------------------------------------------------------------
   
    # max_elevator = ?
    # max_aileron = ?
    # max_rudder = ? 