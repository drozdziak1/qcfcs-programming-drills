import math

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

    return (ro, theta)

def pol2cart(ro, theta):
    a = ro * math.cos(theta)
    b = ro * math.sin(theta)

    return (a, b)
