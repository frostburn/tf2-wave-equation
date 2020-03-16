import numpy as np
import tensorflow as tf

class WaveEquation(object):
    def __init__(self, exitation, boundary, dx, dt=0.01, decay=1e-5, dispersion=0.0, brightness=0.0):
        if exitation.shape != boundary.shape:
            raise ValueError("Incompatible exitation and boundary shapes")
        self.dx = dx
        self.dt = dt
        self.decay = decay
        self.dispersion = dispersion
        omega = []
        for s in boundary.shape:
            wave_numbers = np.arange(s)
            wave_numbers -= s * (2*wave_numbers > s)  # Deal with TensorFlow's uncentered FFT
            expected_span = 2*np.pi
            actual_span = s*dx
            omega.insert(0, wave_numbers * expected_span / actual_span)
        self.omega = np.meshgrid(*omega)
        self.dims = len(boundary.shape)

        if self.dims == 1:
            self.fft = tf.signal.fft
            self.ifft = tf.signal.ifft
            self.omega_x = self.omega[0]
            omega2 = tf.constant(self.omega_x**2, 'complex128')
            gyre = 1j * omega2**(0.5 + dispersion)
            dampener = -decay * omega2**(1.0 - brightness)
            self.positive_gyre = tf.exp(dt * (gyre + dampener))
            self.negative_gyre = tf.exp(dt * (-gyre + dampener))

        self.t = 0.0
        u = tf.constant(exitation, 'complex128')
        self.boundary = tf.constant(boundary, 'complex128')
        self.positive_v = self.fft(u) * 0.5
        self.negative_v = self.fft(u) * 0.5

        def integrator(positive_v, negative_v):
            positive_v *= self.positive_gyre
            negative_v *= self.negative_gyre
            positive_u = self.ifft(positive_v)
            negative_u = self.ifft(negative_v)

            room = tf.cast(1 - tf.abs(self.boundary), 'complex128')
            positive_v = self.fft(positive_u * room - negative_u * self.boundary)
            negative_v = self.fft(negative_u * room - positive_u * self.boundary)

            return positive_v, negative_v

        self.integrator = tf.function(integrator)


    def step(self):
        self.positive_v, self.negative_v = self.integrator(self.positive_v, self.negative_v)
        self.t += self.dt

    def numpy(self):
        return tf.cast(self.ifft(self.positive_v + self.negative_v), 'float64').numpy()
