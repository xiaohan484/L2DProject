import math
import time

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        Configuration:
        - min_cutoff: Decrease this to reduce jitter in slow movement (stabilization).
                      Default: 1.0 (Reasonable start)
        - beta: Increase this to reduce lag during fast movement (responsiveness).
                Default: 0.0 (No speed adaptation) -> Try 0.001 to 0.007
        - d_cutoff: Usually keep as 1.0.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        
        # State variables
        self.x_prev = 0.0
        self.dx_prev = 0.0
        self.t_prev = 0.0
        self.initialized = False

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        """
        Input:
            t: Current timestamp (seconds)
            x: Noisy input value
        Output:
            Filtered value
        """
        if not self.initialized:
            self.initialized = True
            self.x_prev = x
            self.t_prev = t
            return x

        # Calculate time elapsed
        t_e = t - self.t_prev
        
        # Avoid division by zero if updates are too fast
        if t_e <= 0.0:
            return self.x_prev

        # Calculate the derivative (speed of change)
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # Dynamically adjust cutoff frequency based on speed
        # The faster the movement (abs(dx_hat)), the higher the cutoff (less filtering, less lag)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Final smoothing
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # Update state
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat