import numpy as np
import tensorflow as tf


class WaveEquationBase(object):
    def __init__(self, shape, dx, dt=None, decay=0.0, dispersion=0.0, brightness=0.0):
        self.dx = dx
        self.dt = dt or dx
        self.decay = decay
        self.dispersion = dispersion
        self.brightness = brightness
        omega = []
        for s in shape:
            wave_numbers = np.arange(s)
            wave_numbers -= s * (2*wave_numbers > s)  # Deal with TensorFlow's uncentered FFT
            expected_span = 2*np.pi
            actual_span = s*dx
            omega.append(wave_numbers * expected_span / actual_span)
        self.omega = np.meshgrid(*omega, indexing='ij')
        self.dims = len(shape)

    def unpack_wave_numbers(self):
        pass

    def calculate_kernel(self):
        if self.dims == 1:
            self.fft = tf.signal.fft
            self.ifft = tf.signal.ifft
            omega2 = tf.constant(self.omega_x**2, 'complex128')
        elif self.dims == 2:
            self.fft = tf.signal.fft2d
            self.ifft = tf.signal.ifft2d
            omega2 = tf.constant(self.omega_x**2 + self.omega_y**2, 'complex128')
        elif self.dims == 3:
            self.fft = tf.signal.fft3d
            self.ifft = tf.signal.ifft3d
            omega2 = tf.constant(self.omega_x**2 + self.omega_y**2 + self.omega_z**2, 'complex128')
        gyre = 1j * omega2**(0.5 + self.dispersion)
        dampener = -self.decay * omega2**(1.0 - self.brightness)
        self.kernel = tf.exp(self.dt * (gyre + dampener))

    def unpack_wave_numbers(self):
        if self.dims == 1:
            self.omega_x = self.omega[0]
        elif self.dims == 2:
            self.omega_x = self.omega[0]
            self.omega_y = self.omega[1]
        elif self.dims == 3:
            self.omega_x = self.omega[0]
            self.omega_y = self.omega[1]
            self.omega_z = self.omega[2]


class WaveEquation(WaveEquationBase):
    def __init__(self, exitation, boundary, *args, **kwargs):
        if exitation.shape != boundary.shape:
            raise ValueError("Incompatible exitation and boundary shapes")
        super().__init__(exitation.shape, *args, **kwargs)
        self.unpack_wave_numbers()
        self.calculate_kernel()
        self.setup_integration(exitation, boundary)

    def setup_integration(self, exitation, boundary):
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


class TriangleWaveEquation(WaveEquation):
    """
    Wave equation on an equilateral triangle with chiral boundary conditions.
    """
    def __init__(self, exitation, boundary, dx=None, *args, **kwargs):
        N = int(0.5 * np.sqrt(8*len(exitation) + 1) - 0.5)
        if dx is None:
            dx = 2 / N
        u = np.zeros((N, N), dtype=complex)
        b = np.zeros((N, N), dtype=complex)
        index = 0
        for j in range(N):
            for i in range(j+1):
                u[j, i] = exitation[index]
                b[j, i] = boundary[index]
                if j < N-1:
                    u[N-1-j, N-1-i] = exitation[index]
                    b[N-1-j, N-1-i] = boundary[index]
                index += 1
        super().__init__(u, b, dx, *args, **kwargs)

    def unpack_wave_numbers(self):
        self.omega_x = (2*self.omega[0] + self.omega[1]) / np.sqrt(3)
        self.omega_y = self.omega[1]

    def numpy(self):
        u = super().numpy()
        rows = []
        for j in range(u.shape[0]):
            rows.append(u[j,:j+1])
        return np.concatenate(rows)

    @classmethod
    def xy(cls, N):
        x = []
        y = []
        for j in range(N):
            for i in range(j+1):
                x.append(-j/N + 2*i/N)
                y.append(-2*np.sqrt(3)/3 + np.sqrt(3)*j/N)
        return np.array(x), np.array(y)
