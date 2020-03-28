import pylab
import numpy as np
from matplotlib.animation import FuncAnimation
from wave_equation import WaveEquation, TriangleWaveEquation
import shapes


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

    boundary = 0*x

    wave_equation = WaveEquation(u, boundary, dx, dt=dx, decay=0)

    u = wave_equation.ifft(wave_equation.v)[:,:,M]
    plots = [pylab.imshow(np.real(u), vmin=-0.05, vmax=0.05, extent=(-2, 2,-2,2))]
    plots.extend(pylab.plot(np.linspace(-2,2,N), np.real(u[M])))
    plots.extend(pylab.plot(np.linspace(-2,2,N), np.imag(u[M])))

    def update(frame):
        wave_equation.step()
        print(wave_equation.t)
        u = wave_equation.ifft(wave_equation.v)[:,:,M]
        plots[0].set_data(np.real(u))
        plots[1].set_ydata(np.real(u[M]))
        plots[2].set_ydata(np.imag(u[M]))
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


def room_reverb(animate=False):
    from scipy.special import dawsn
    N = 256
    x = np.linspace(-2, 2, N+1)[:-1] + 0j
    dx = np.real(x[1]-x[0])
    M = N//2
    L = M//2
    y = (np.arange(M)-L)*dx
    z = (np.arange(M)-L)*dx
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    # Inwards surging spherical wave with a gaussian profile
    # TODO: Figure out how to do an actual point source
    r = np.sqrt((x-0.4)**2+(y-0.06)**2+(z+0.01)**2)
    r1 = 45*r-5
    u = -(np.exp(-r1**2) + 2j/np.sqrt(np.pi)*dawsn(r1)) / (r+1)

    boundary = 1-np.exp(-(0.6*x)**1000-(1.1*y)**1000-(1.2*z)**1000)

    if animate:
        pylab.imshow(np.real(boundary[M]))
        pylab.show()
        pylab.imshow(np.real(boundary[:,L]))
        pylab.show()

    wave_equation = WaveEquation(u, boundary, dx, dt=dx, decay=0)

    u = wave_equation.ifft(wave_equation.v)[M]
    if animate:
        plots = [pylab.imshow(np.real(u), vmin=-0.05, vmax=0.05, extent=(-2, 2,-2,2))]
        plots.extend(pylab.plot(np.linspace(-2,2,M), np.real(u[L])))
        plots.extend(pylab.plot(np.linspace(-2,2,M), np.imag(u[L])))

    left = []
    right = []

    def step(frame):
        wave_equation.step()
        u = wave_equation.ifft(wave_equation.v)[M]
        left.append(np.real(u[L, L-8]))
        right.append(np.real(u[L, L+7]))
        if frame == 0:
            print("t={}, saving to /tmp/...".format(wave_equation.t))
            np.save("/tmp/left.npy", np.array(left, dtype=float))
            np.save("/tmp/right.npy", np.array(right, dtype=float))
        return u

    def update(frame):
        u = step(frame)
        plots[0].set_data(np.real(u))
        plots[1].set_ydata(np.real(u[L]))
        plots[2].set_ydata(np.imag(u[L]))
        return plots

    if animate:
        FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=20)
        pylab.show()

        pylab.plot(left)
        pylab.show()

        pylab.plot(right)
        pylab.show()
    else:
        print("Calculating room reverb. Press Ctrl+C to stop collecting the impulse response.")
        while True:
            for i in range(100):
                step(i)


def triangular_chiral_toroid_membrane():
    x, y = TriangleWaveEquation.xy(512)

    t = TriangleWaveEquation(np.exp(-1000*(x*x+y*y)), 0*x, decay=0)

    for _ in range(222):
        t.step()

    pylab.tricontourf(x, y, t.numpy())
    pylab.show()


def profile_in_2D():
    x = np.linspace(-4, 4, 512)
    dx = x[1] - x[0]
    x, y = np.meshgrid(x, x, indexing='ij')

    w = WaveEquation(np.exp(-100*(x*x+y*y)), 0*x, dx)

    for _ in range(20):
        pylab.plot(w.numpy()[256, 256:])
        for _ in range(12):
            w.step()
    a = np.linspace(0, 4, 256)
    pylab.plot(0.1*(a+0.01)**-0.5)
    pylab.plot(-0.05*(a+0.01)**-0.5)
    pylab.ylim(-0.12, 0.22)
    pylab.show()


def hollow_cube():
    N = 128
    M = N//2
    x = np.linspace(-1.2, 1.2, N)
    dx = x[1] - x[0]
    x, y, z = np.meshgrid(x, x, x, indexing='ij')

    cube = shapes.cube(x, y, z)

    boundary = np.logical_or(cube > 1, cube < 0.7)

    u = np.exp(-1000*(x**2 + (y-0.95)**2 + (z-0.2)**2))
    u *= (1-boundary)

    wave_equation = WaveEquation(u, boundary, dx, dt=dx, decay=0)

    u = wave_equation.numpy()
    plots = [pylab.imshow(u[M], vmin=-0.04, vmax=0.04)]

    def update(frame):
        wave_equation.step()
        print(wave_equation.t)
        u = wave_equation.numpy()
        plots[0].set_data(u[M])
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=10)
    pylab.show()


def tetrahedron_scaffold():
    N = 128
    M = N // 2
    x = np.linspace(-5, 5, N)
    dx = x[1] - x[0]
    x, y, z = np.meshgrid(x, x, x, indexing='ij')
    boundary = np.logical_not(shapes.tetrahedron_scaffold(x, y, z, width=0.4))

    pylab.imshow(boundary[M], extent=(-5,5,-5,5))
    pylab.show()

    u = np.exp(-100*(x**2 + (y-0.1)**2 + (z+1.6)**2))
    u *= (1-boundary)

    wave_equation = WaveEquation(u, boundary, dx, dt=dx, decay=0.0001)

    u = wave_equation.numpy()
    plots = [pylab.imshow(u[M], vmin=-0.02, vmax=0.02)]

    def update(frame):
        wave_equation.step()
        print(wave_equation.t)
        u = wave_equation.numpy()
        plots[0].set_data(u[M])
        return plots

    FuncAnimation(pylab.gcf(), update, frames=range(100), init_func=lambda: plots, blit=True, repeat=True, interval=10)
    pylab.show()


if __name__ == '__main__':
    # animated_string()
    # animated_membrane()
    # animated_room()
    # animated_point_source_in_room()
    # animated_point_source_on_string()
    # room_reverb()
    # triangular_chiral_toroid_membrane()
    # profile_in_2D()
    hollow_cube()
    # tetrahedron_scaffold()

