using ForwardDiff

# Define a function
f(x) = x^2 + 3x + 2

# Compute the derivative at a point
x = 2.0
dfdx = ForwardDiff.derivative(f, x)

println("The derivative of f at x = $x is $dfdx")
