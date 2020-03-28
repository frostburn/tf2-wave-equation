from __future__ import division

import argparse
import numpy as np
import pylab
import progressbar

import shapes
from util import write_audio
from wave_equation import WaveEquation


def process_triangle_drum(args, integrator_params):
    N = args.grid_width
    x = np.arange(-1, 1, 2/N)
    dx = x[1] - x[0]
    x, y = np.meshgrid(x, x, indexing='ij')

    theta = 0.29
    c = np.cos(theta)
    s = np.sin(theta)
    x, y = x*c + y*s - 0.1, y*c - x*s + 0.1

    triangle = shapes.equilateral(x, y, 3)

    boundary = (triangle > 0.52)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render audio samples')
    parser.add_argument('outfile', type=str, help='Output file name')
    parser.add_argument('--grid-width', type=int, help='Number of grid points along a grid edge', default=128)
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

    process_triangle_drum(args, integrator_params)
