from __future__ import division

import argparse
import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import progressbar

import shapes
from util import write_audio
from wave_equation import WaveEquation


def triangle(N):
    x = np.arange(-1, 1, 2/N)
    dx = x[1] - x[0]
    x, y = np.meshgrid(x, x, indexing='ij')

    theta = 0.29
    c = np.cos(theta)
    s = np.sin(theta)
    x, y = x*c + y*s - 0.1, y*c - x*s + 0.1

    triangle_ = shapes.equilateral(x, y, 3)

    return x, y, dx, (triangle_ > 0.52)


def tetrahedron(N):
    x = np.arange(-1, 1, 2/N)
    dx = x[1] - x[0]
    x, y, z = np.meshgrid(x, x, x, indexing='ij')

    tetrahedron_ = shapes.regular_tetrahedron(x, y, z)

    return x, y, z, dx, (tetrahedron_ > 0.4)


def strike_triangle_drum(args, integrator_params):
    N = args.grid_width
    x, y, dx, boundary = triangle(N)

    ex_x, ex_y = np.random.randn(2) * args.variation
    sharpness = args.sharpness * np.exp(np.random.randn() * args.variation)

    exitation = np.exp(-sharpness**2 * ((x-ex_x)**2 + (y-ex_y)**2))
    exitation *= 1 - boundary

    integrator = WaveEquation(exitation, boundary, dx, **integrator_params)

    left_channel = []
    right_channel = []

    for i in progressbar.progressbar(range(int(args.sample_rate * args.duration))):
        left_channel.append(np.real(integrator.u[int(N*0.4), N//2].numpy()))
        right_channel.append(np.real(integrator.u[int(N*0.7), N//2].numpy()))
        integrator.step()

    data = np.array([left_channel, right_channel])
    data /= abs(data).max()

    write_audio(args.outfile, data, sample_rate=args.sample_rate)


def triangle_resonances(args, integrator_params):
    N = args.grid_width
    x, y, dx, boundary = triangle(N)

    def vibrate(t):
        x_ = args.variation * (1 + np.cos(0.01*t))
        y_ = args.variation * np.sin(0.01 * t)
        return np.exp(-args.sharpness**2 * (x_**2 + y_**2)) * np.sin(0.5*t + 0.005*t*t) * (1-boundary)

    integrator_params['continuous_exitation'] = vibrate
    integrator = WaveEquation(0*x, boundary, dx, **integrator_params)

    left_channel = []
    right_channel = []

    for i in progressbar.progressbar(range(int(args.sample_rate * args.duration))):
        left_channel.append(np.real(integrator.u[int(N*0.4), N//2].numpy()))
        right_channel.append(np.real(integrator.u[int(N*0.7), N//2].numpy()))
        integrator.step()

    data = np.array([left_channel, right_channel])
    data /= abs(data).max()

    write_audio(args.outfile, data, sample_rate=args.sample_rate)


def strike_tetrahedron(args, integrator_params):
    N = args.grid_width
    x, y, z, dx, boundary = tetrahedron(N)

    ex_x, ex_y, ex_z = np.random.randn(3) * args.variation * 0.2 + np.array([0.55, 0.1, 0.13])
    exitation = np.exp(-args.sharpness**2*((x-ex_x)**2 + (y-ex_y)**2 + (z-ex_z)**2))
    exitation *= 1 - boundary

    integrator = WaveEquation(exitation, boundary, dx, **integrator_params)

    left_channel = []
    right_channel = []

    for i in progressbar.progressbar(range(int(args.sample_rate * args.duration))):
        left_channel.append(np.real(integrator.u[int(N*0.4), int(N*0.3), int(N*0.4)].numpy()))
        right_channel.append(np.real(integrator.u[int(N*0.4), int(N*0.3), int(N*0.36)].numpy()))
        integrator.step()

    data = np.array([left_channel, right_channel])
    data /= abs(data).max()

    write_audio(args.outfile, data, sample_rate=args.sample_rate)


if __name__ == '__main__':
    episodes = {
        'drum': strike_triangle_drum,
        'resonance': triangle_resonances,
        'tetrahedron': strike_tetrahedron,
    }


    parser = argparse.ArgumentParser(description='Render audio samples')
    parser.add_argument('episode', choices=episodes.keys())
    parser.add_argument('outfile', type=str, help='Output file name')
    parser.add_argument('--grid-width', type=int, help='Number of grid points along a grid edge', metavar='N', default=128)
    parser.add_argument('--sample-rate', type=int, help='Sampling rate of the resulting audio file', default=48000)
    parser.add_argument('--duration', type=float, help='Duration of audio to render', default=1.0)
    parser.add_argument('--sharpness', type=float, help='Sharpness of the initial hit', default=10.0)
    parser.add_argument('--variation', type=float, help='Add randomness to the initial conditions', default=0.1)
    parser.add_argument('--decay', type=float, help='Frequence decay, higher value -> duller sound', default=1e-5)
    parser.add_argument('--brightness', type=float, help='Frequence decay curve shape, higher value -> brighter sound', default=0.0)
    parser.add_argument('--dispersion', type=float, help='Frequence dispersion, non-zero -> metallic sound', default=0.0)
    args = parser.parse_args()

    integrator_params = {
        'decay': args.decay,
        'brightness': args.brightness,
        'dispersion': args.dispersion,
    }

    episodes[args.episode](args, integrator_params)
