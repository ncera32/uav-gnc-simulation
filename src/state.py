# Source code for UAV state

class AircraftState:
    # State of rigid-body fixed-wing aircraft

    def __init__(
         self,
         pn=0.0,
         pe=0.0,
         pd=0.0,
         u=0.0,
         v=0.0,
         w=0.0,
         phi=0.0,
         theta=0.0,
         psi=0.0,
         p=0.0,
         q=0.0,
         r=0.0,   
    ):
        
        # Position in NED frame [m]
        self.pn = pn
        self.pe = pe
        self.pd = pd

        # Translational velocity in body frame [m/s]
        self.u = u
        self.v = v
        self.w = w

        # Euler angles [rad]
        self.phi = phi
        self.theta = theta
        self.psi = psi

        # Angular rates in body frame [rad/s]
        self.p = p
        self.q = q
        self.r = r

    # altitude above the NED origin [m]
    @property 
    def altitude(self):
        return -self.pd
