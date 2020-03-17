import pylab
import numpy as np
from matplotlib.animation import FuncAnimation
from wave_equation import WaveEquation


def animated_string():
    x = np.linspace(-1.2, 1.2, 256)
    dx = x[1]-x[0]

    u = np.exp(-260*(x-0.7)**2)

    boundary = (1 - np.exp(-x**1000))
    u *= 1-abs(boundary)

    pylab.plot(boundary)
    pylab.show()

    wave_equation = WaveEquation(u, boundary, dx)

    plots = pylab.plot(wave_equation.numpy())
    pylab.ylim(-1, 1)

    def update(frame):
        for _ in range(1):
            wave_equation.step()
        print(wave_equation.t)
        plots[0].set_ydata(wave_equation.numpy())
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=10)
    pylab.show()


def animated_membrane():
    x = np.linspace(-1.2, 1.2, 256)
    dx = x[1]-x[0]
    y = (np.arange(128)-64)*dx
    x, y = np.meshgrid(x, y, indexing='ij')

    u = np.exp(-260*(x-0.7)**2 - 260*(y-0.1)**2)

    boundary = (1 - np.exp(-(x**2 + (2*y)**2)**500))
    u *= 1-abs(boundary)

    pylab.imshow(boundary)
    pylab.show()

    wave_equation = WaveEquation(u, boundary, dx)

    plots = [pylab.imshow(wave_equation.numpy(), vmin=-0.1, vmax=0.1)]

    def update(frame):
        for _ in range(1):
            wave_equation.step()
        print(wave_equation.t)
        plots[0].set_data(wave_equation.numpy())
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=10)
    pylab.show()


def animated_room():
    x = np.linspace(-1.2, 1.2, 128)
    dx = x[1]-x[0]
    y = (np.arange(64)-32)*dx
    z = (np.arange(32)-16)*dx
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    u = np.exp(-150*(x-0.5)**2 - 150*(y-0.1)**2 - 150*(z)**2)
    u -= np.exp(-260*(x-0.5)**2 - 260*(y-0.1)**2 - 260*(z)**2)

    boundary = (1 - np.exp(-(x**2 + (2*y)**2 + (4*z)**2)**400))
    u *= 1-abs(boundary)

    pylab.imshow(boundary[:,:,16])
    pylab.show()

    wave_equation = WaveEquation(u, boundary, dx, dt=dx)

    plots = [pylab.imshow(wave_equation.numpy()[:,:,16], vmin=-0.05, vmax=0.05)]

    def update(frame):
        for _ in range(1):
            wave_equation.step()
        print(wave_equation.t)
        plots[0].set_data(wave_equation.numpy()[:,:,16])
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=10)
    pylab.show()


def animated_point_source_in_room():
    from scipy.special import dawsn
    N = 128
    x = np.linspace(-2, 2, N+1)[:-1] + 0j
    dx = np.real(x[1]-x[0])
    M = N//2
    y = z = x
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    # Inwards surging spherical wave with a gaussian profile
    # TODO: Figure out how to do an actual point source
    r = np.sqrt(x*x+y*y+z*z)
    r1 = 20*r-15
    u = -(np.exp(-r1**2) + 2j/np.sqrt(np.pi)*dawsn(r1)) / (r+1)

    plots = pylab.plot(np.real(u[M,M]))
    plots.extend(pylab.plot(np.imag(u[M,M])))
    pylab.show()

    boundary = 0*x

    wave_equation = WaveEquation(u, boundary, dx, dt=dx, decay=0)

    plots = [pylab.imshow(wave_equation.numpy()[:,:,M], vmin=-0.05, vmax=0.05)]

    def update(frame):
        wave_equation.step()
        print(wave_equation.t)
        plots[0].set_data(wave_equation.numpy()[:,:,M])
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(18), init_func=lambda: plots, blit=True, repeat=True, interval=20)
    pylab.show()


def animated_point_source_on_string():
    x = np.linspace(-1.2, 1.2, 128) + 0j
    dx = np.real(x[1]-x[0])

    u = 0*x
    u[64] = 1

    v = np.fft.fftn(u)
    v[65:] *= 0
    v[64] *= 0.5
    u = np.fft.ifftn(v)

    pylab.plot(np.real(u))
    pylab.plot(np.imag(u))
    pylab.show()

    boundary = 0*x

    wave_equation = WaveEquation(u, boundary, dx, dt=dx, decay=0)

    plots = pylab.plot(wave_equation.numpy())

    def update(frame):
        for _ in range(1):
            wave_equation.step()
        print(wave_equation.t)
        plots[0].set_ydata(wave_equation.numpy())
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=10)
    pylab.show()


if __name__ == '__main__':
    # animated_string()
    # animated_membrane()
    # animated_room()
    animated_point_source_in_room()
    # animated_point_source_on_string()
