
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
