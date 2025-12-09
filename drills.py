import math
import random

from functools import reduce

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

# Drill 1.1.1

def complex_add(a1, b1, a2, b2):
    return (a1 + a2, b1 + b2)

def complex_mul(a1, b1, a2, b2):
    return (a1 * a2 - b1 * b2, a1 * b2 + b1 * a2)

# Drill 1.2.1

def complex_sub(a1, b1, a2, b2):
    return complex_add(a1, b1, -a2, -b2)

def complex_div(a1, b1, a2, b2):
    # The real and imaginary parts share a denominator when dividing complex numbers
    denom = a2 ** 2 + b2 ** 2

    real = (a1 * a2 + b1 * b2) / denom
    imag = (a2 * b1 - a1 * b2) / denom
    
    return (real, imag)

# Drill 1.3.1
def cart2pol(a, b):
    ro = (a ** 2 + b ** 2) ** 0.5

    theta = None

    # the b/a division in formula causes problems for a == 0. Handle that separately.
    if a == 0:

        # If b is also 0, we should default theta to 0
        theta = (b != 0) * math.pi / 2

        # b < 0 means it's 3/2*pi radians, 1/2*pi otherwise
        theta += (b < 0) * math.pi

    else:
        theta = math.atan(b / a)
        
        # Since atan() gives values between -pi/2 .. pi/2, we need to
        # adjust the offset using the original real and imaginary part's signs.
        if a < 0:
            if b >= 0:
                theta += math.pi 
            else:
                theta -= math.pi


    return (ro, theta)

def pol2cart(ro, theta):
    a = ro * math.cos(theta)
    b = ro * math.sin(theta)

    return (a, b)

# Drill 1.3.2
def animate_complex_transform(points, transform=(1, 0), max_graphed_coord=1, n_frames=20):

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    
    zipped = list(zip(*points))
    scatter = ax.scatter(zipped[0], zipped[1], label="complex datapoints")
    scatter_transform = ax.scatter([1.], [0.], color='r', label="transform")
    ax.set_aspect(1.)
    ax.set_xlabel("real")
    ax.set_ylabel("imaginary")
    ax.grid()
    ax.legend()

    transform_polar = cart2pol(*transform)
    ax.set_title(f"transform = {transform[0]:.5} + ({transform[1]:.5})i; ro={transform_polar[0]:.5}, theta={(transform_polar[1] / (2 * math.pi) * 360):.5}deg")

    def init():
        ax.set_xlim(-max_graphed_coord, max_graphed_coord)
        ax.set_ylim(-max_graphed_coord, max_graphed_coord)
        return scatter,

    def update(frame):
        partial_transform = frame[-1]
        frame = frame[:-1]

        scatter.set_offsets(frame)
        scatter_transform.set_offsets([partial_transform])

        return ax, scatter, scatter_transform

    # Calculate a series of frames between unchanged and fully transformed points
    frames = []

    # Transform in stages: Show start state, scale, rotate, show end state
    n_stages = 4

    for frame_no in range(n_frames + 1):
        frame = []

        frames_per_stage = n_frames / n_stages
        stage_frame_no = frame_no % frames_per_stage

        # Note: Each partial transform is represented in polar. Polar
        # coordinates fully separate scaling and rotation, which
        # cannot be said for Cartesian representation. 
        if frame_no < frames_per_stage:
            # Show start state using trivial transform
            partial_transform_polar = (1, 0)

        elif frame_no < frames_per_stage * 2:
            # Scale - average between 1.0 and target scale, weighted by frame
            partial_transform_polar = (
                (frames_per_stage - stage_frame_no) / frames_per_stage + transform_polar[0] * stage_frame_no / frames_per_stage,
                0
                )
        elif frame_no < frames_per_stage * 3:
            # Rotate
            partial_transform_polar = (
                transform_polar[0],
                transform_polar[1] * stage_frame_no / frames_per_stage,
                )
        else:
            # Show end state
            partial_transform_polar = transform_polar

        partial_transform = pol2cart(*partial_transform_polar)

        for point in points:

            frame_point = complex_mul(*point, *partial_transform)
            frame.append(frame_point)

        frame.append(partial_transform)
        frames.append(frame)

            
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)

    plt.show()
            


# Run this to just see animate_complex_transform in action
def demo_complex_transform(n_points=10, span=10, transform_max_scale=3, custom_points=[], custom_transform=None):
    points = [] if len(custom_points) == 0 else custom_points

    if len(points) == 0:
        for _ in range(n_points):
            a = random.random() * span - span / 2
            b = random.random() * span - span / 2
            points.append((a, b))
    
    transform_polar_ro = cart2pol(*custom_transform)[0] if custom_transform is not None else (random.random() * 2 * transform_max_scale - transform_max_scale)
    transform_polar_theta = cart2pol(*custom_transform)[1] if custom_transform is not None else (random.random() * 4 * math.pi - 2 * math.pi) 

    transform = pol2cart(transform_polar_ro, transform_polar_theta) if custom_transform is None else custom_transform

    animate_complex_transform(points, transform, max_graphed_coord=(1.1 * max(math.fabs(transform_polar_ro * span / 2), span / 2)))


# Drill 1.3.3
def complex_pol_mul(ro1, th1, ro2, th2):
    return (ro1 * ro2, th1 + th2)

def complex_pol_div(ro1, th1, ro2, th2):
    return (ro1 / ro2, th1 - th2)

# Drill 2.1.1

# Generic for-each with dimension verification
def complex_v_binary_op(v1, v2, binary_op):
    len1 = len(v1)
    len2 = len(v2)
    if (len1 != len2):
        raise ValueError(f"vector size mismatch: {len1} != {len2}")

    ret = []

    for elem1, elem2 in zip(v1, v2):
        ret.append(binary_op(*elem1, *elem2))

    return ret

def complex_v_add_v(v1, v2):
    return complex_v_binary_op(v1, v2, complex_add)

def complex_v_scalar_mul(a, b, v):
    mul_vec = [(a, b)] * len(v)

    return complex_v_binary_op(mul_vec, v, complex_mul)

def complex_v_inverse(v):
    return complex_v_scalar_mul(-1, 0, v)


# Drill 2.2.1

# Builds on the vector variant, verifies additional dimension
def complex_m_binary_op(m1, m2, binary_op):
    rows1 = len(m1)
    rows2 = len(m2)

    if (rows1 != rows2):
        raise ValueError(f"matrix row count mismatch: {rows1} != {rows2}")

    ret = []

    for row1, row2 in zip(m1, m2):
        try:
            ret.append(complex_v_binary_op(row1, row2, binary_op))
        except ValueError(msg):
            raise ValueError(f"matrix column count mismatch: {msg}")

    return ret


def complex_m_add_m(m1, m2):
    return complex_m_binary_op(m1, m2, complex_add)

def complex_m_scalar_mul(a, b, m):
    ret = []

    for row in m:
        ret.append(complex_v_scalar_mul(a, b, row))

    return ret

def complex_m_inverse(m):
    return complex_m_scalar_mul(-1, 0, m)

# Drill 2.2.2 and 2.2.3 (the matmul is generic for matmul(Cmxn, Cnxp))

def complex_m_transpose(m):

    rows = len(m)

    if (rows == 0):
        return []
    
    columns = len(m[0])

    # Note: ret has inverted indices, thus rows are used for columns
    # and vice versa.
    ret = [[None for _ in range(rows)] for _ in range(columns)]

    for i in range(rows):
        for j in range(columns):
            ret[j][i] = m[i][j]

    return ret

def complex_matmul(m1, m2):
    # Not the canonical inner product in C, but still useful for matrix multiplication
    def complex_v_dot_no_conjugate(v1, v2):
        products_v = complex_v_binary_op(v1, v2, complex_mul)

        return reduce(lambda accum, prod: complex_add(*accum, *prod), products_v)

    rows = len(m1)

    # Transpose for easier dimension verification/column access
    m1t = complex_m_transpose(m1)
    m2t = complex_m_transpose(m2)

    columns = len(m2t)


    if (len(m1t) != len(m2)):
        raise ValueError(f"Matrix dimension mismatch: {len(m1t)} columns does not multiply with {len(m2)} rows")
    

    ret = [[None for _ in range(columns)] for _ in range(rows)]

    for i in range(rows):
        for j in range(columns):
            print(f"{i}/{rows}, {j}/{columns}")
            ret[i][j] = complex_v_dot_no_conjugate(m1[i], m2t[j])

    return ret

# Drill 2.4.1

def complex_conjugate(a, b):
    return (a, -b)

# A.k.a. dagger
def complex_m_adjoint(m):
    rows = len(m)

    if (rows == 0):
        return []

    columns = len(m[0])

    m_conjugate = [[complex_conjugate(*m[i][j]) for j in range(columns)] for i in range(rows)]

    return complex_m_transpose(m_conjugate)
        
# A.k.a. dot product, called inner to distinguish from complex_matmul() helper. Note: expects row vectors (regular Python lists)
def complex_v_inner_product(v1, v2):
    # Start with column vector to fit the equation in the book
    v1t = complex_m_transpose([v1])

    # This becomes row vector as expected by the formula
    v1t_dagger = complex_m_adjoint(v1t)

    # Use column vector to fit the equations in the book
    v2t = complex_m_transpose([v2])

    return complex_matmul(v1t_dagger, v2t)[0][0] # unwrap the scalar from 1x1

    
