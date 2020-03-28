import numpy as np

def equilateral(x, y, n=3):
    """
    Uniform distance field with n-gonal symmetry.
    """

    u = x
    for i in range(1, n):
        theta = 2*np.pi * i / n
        u = np.maximum(u, x*np.cos(theta) + y*np.sin(theta))
    return u


def regular_tetrahedron(x, y, z):
    """
    Uniform distance field with tethedral symmetry.
    """

    u = x / np.sqrt(1.5) - z / np.sqrt(3)
    u = np.maximum(u, -x / np.sqrt(1.5) - z / np.sqrt(3))
    u = np.maximum(u, y / np.sqrt(1.5) + z / np.sqrt(3))
    u = np.maximum(u, -y / np.sqrt(1.5) + z / np.sqrt(3))
    return u


def tetrahedron_scaffold(x, y, z, width=0.1):
    """
    The edge framework of the tetrahedron of specified edge width.
    """
    u = regular_tetrahedron(x, y, z)
    scaffold = (u < 1)
    u = regular_tetrahedron(x - 2*width / np.sqrt(1.5), y, z + 2*width / np.sqrt(3))
    scaffold = np.logical_and(scaffold, u > 1 - width)
    u = regular_tetrahedron(x + 2*width / np.sqrt(1.5), y, z + 2*width / np.sqrt(3))
    scaffold = np.logical_and(scaffold, u > 1 - width)
    u = regular_tetrahedron(x, y - 2*width / np.sqrt(1.5), z - 2*width / np.sqrt(3))
    scaffold = np.logical_and(scaffold, u > 1 - width)
    u = regular_tetrahedron(x, y + 2*width / np.sqrt(1.5), z - 2*width / np.sqrt(3))
    scaffold = np.logical_and(scaffold, u > 1 - width)
    return scaffold


def cube(x, y, z):
    """
    Uniform distance field with cubical symmetry.
    """

    u = abs(x)
    u = np.maximum(u, abs(y))
    u = np.maximum(u, abs(z))
    return u


def regular_octahedron(x, y, z):
    """
    Uniform distance field with octahedral symmetry.
    """

    u = abs(x + y + z)
    u = np.maximum(u, abs(x + y - z))
    u = np.maximum(u, abs(x - y + z))
    u = np.maximum(u, abs(x - y - z))

    return u / np.sqrt(3)


def stellated_octahedron(x, y, z):
    """
    Uniform distance field with stellated octahederal symmetry.
    """
    return np.minimum(
        regular_tetrahedron(x, y, z),
        regular_tetrahedron(x, y, -z)
    )


def regular_dodecahedron(x, y, z):
    """
    Uniform distance field with dodecahedral symmetry.
    """
    phi = 0.5 * (1 + np.sqrt(5))
    u = 0*x
    for s1 in [-1, 1]:
        for s2 in [-phi, phi]:
            normal = [0, s1, s2]
            for _ in range(3):
                u = np.maximum(u, x * normal[0] + y * normal[1] + z * normal[2])
                normal = [normal[2], normal[0], normal[1]]
    return u / np.sqrt(1 + phi**2)


def regular_icosahedron(x, y, z):
    """
    Uniform distance field with icosahedral symmetry.
    """
    phi = 0.5 * (1 + np.sqrt(5))
    normals = []
    u = regular_octahedron(x, y, z) * np.sqrt(3)
    for s in [-1, 1]:
        u = np.maximum(u, abs(phi*x + y*s/phi))
        u = np.maximum(u, abs(phi*y + z*s/phi))
        u = np.maximum(u, abs(phi*z + x*s/phi))
    return u / np.sqrt(3)
