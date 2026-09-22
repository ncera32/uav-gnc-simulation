# Source code for UAV state

import numpy as np

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

    # return the state array as a 12-element NumPy array  
    def to_array(self):
        return np.array([
            self.pn,
            self.pe,
            self.pd,
            self.u,
            self.v,
            self.w,
            self.phi,
            self.theta,
            self.psi,
            self.p,
            self.q,
            self.r
        ], dtype = float)

    # Create an AircraftState from a 12-element state array
    @classmethod 
    def from_array(cls, x):
        x = np.asarray(x, dtype = float)

        if x.shape != (12,):
            raise ValueError(
                "Aircraft state array must contain exactly 12 elements"
            )
    
        return cls(
            pn = x[0],
            pe = x[1],
            pd = x[2],
            u = x[3],
            v = x[4],
            w = x[5],
            phi = x[6],
            theta = x[7],
            psi = x[8],
            p = x[9],
            q = x[10],
            r = x[11]
        )






