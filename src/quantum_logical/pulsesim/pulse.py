"""Class representing a single pulse."""

import matplotlib.pyplot as plt
import numpy as np


class Pulse:
    """Class representing a single pulse."""

    def __init__(self, omega, amp, phi=0):
        """Initialize the Pulses object with common parameters.

        Args:
            omega (float): Base frequency of the pulses.
            amp (float): Base amplitude of the pulses.
            phi (float): Base phase of the pulses.
        """
        self.omega = omega
        self.amp = amp
        self.phi = phi

    @staticmethod
    def gaussian(t, t0, width, nsig=6):
        """Gaussian pulse shape."""
        return np.exp(-0.5 * ((t - t0 - width * nsig / 2) / width) ** 2)

    @staticmethod
    def smoothbox(t, t0, width, k=0.5, b=3):
        """Smooth box pulse shape."""
        return 0.5 * (np.tanh(k * (t - t0) - b) - np.tanh(k * (t - t0 - width) + b))

    @staticmethod
    def box(t, t0, width):
        """Box pulse shape."""
        return np.heaviside(t - t0, 0) - np.heaviside(t - t0 - width, 0)

    def drive(self, t, args):
        """Drive function applying amplitude and frequency modulation."""
        pulse_shape = args.get("shape", Pulse.box)
        shape_params = args.get("shape_params", {})
        envelope = pulse_shape(t, **shape_params)
        return self.amp * np.cos(self.omega * t + self.phi) * envelope

    def plot_pulse(self, pulse_shape, t_list, **shape_params):
        """Plot both the pulse envelope and the modulated pulse."""
        envelope_values = [pulse_shape(t, **shape_params) for t in t_list]
        modulated_values = [
            self.drive(t, {"shape": pulse_shape, "shape_params": shape_params})
            for t in t_list
        ]
        plt.plot(
            t_list, modulated_values, label="Modulated Pulse", linestyle="--", alpha=0.7
        )
        plt.plot(
            t_list, envelope_values, label="Pulse Envelope", linewidth=2, color="red"
        )
        plt.xlabel("Time (dt)")
        plt.ylabel("Amplitude")
        plt.title("Envelope and Modulated Pulse")
        plt.legend()
        plt.show()
