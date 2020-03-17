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
            omega.append(wave_numbers * expected_span / actual_span)
        self.omega = np.meshgrid(*omega, indexing='ij')
        self.dims = len(boundary.shape)

        if self.dims == 1:
            self.fft = tf.signal.fft
            self.ifft = tf.signal.ifft
            self.omega_x = self.omega[0]
            omega2 = tf.constant(self.omega_x**2, 'complex128')
        elif self.dims == 2:
            self.fft = tf.signal.fft2d
            self.ifft = tf.signal.ifft2d
            self.omega_x = self.omega[0]
            self.omega_y = self.omega[1]
            omega2 = tf.constant(self.omega_x**2 + self.omega_y**2, 'complex128')
        elif self.dims == 3:
            self.fft = tf.signal.fft3d
            self.ifft = tf.signal.ifft3d
            self.omega_x = self.omega[0]
            self.omega_y = self.omega[1]
            self.omega_z = self.omega[2]
            omega2 = tf.constant(self.omega_x**2 + self.omega_y**2 + self.omega_z**2, 'complex128')
        gyre = 1j * omega2**(0.5 + dispersion)
        dampener = -decay * omega2**(1.0 - brightness)
        self.kernel = tf.exp(dt * (gyre + dampener))

        self.t = 0.0
        u = tf.constant(exitation, 'complex128')
        self.boundary = tf.constant(boundary, 'complex128')
        self.room = tf.cast(1 - tf.abs(self.boundary), 'complex128')
        self.v = self.fft(u)

        def integrator(v):
            v *= self.kernel

            positive_u = self.ifft(v)
            negative_u = tf.math.conj(positive_u)

            v = self.fft(positive_u * self.room - negative_u * self.boundary)

            return v

        self.integrator = tf.function(integrator)


    def step(self):
        self.v = self.integrator(self.v)
        self.t += self.dt

    def numpy(self):
        return tf.cast(self.ifft(self.v), 'float64').numpy()
