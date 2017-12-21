# PHASE_BOUNDARIES_IMPROVED

Note on the analytic curve intersection:

For 2 variables:

```
# Setting "x" and "y" to be symbolic:
x, y = sym.symbols('x y', real=True)
```

For 1 variable it is also valid the `sym.symbols`:

```
# Setting "x" to be symbolic:
x = sym.symbols('x', real=True)
```

Or, is also valid `sym.Symbol`:

```
x = sym.Symbol('x')
```
If therre are only complex solutions, the 

```P = sym.Symbol('P', real=True)```

will not return anything, since we are asking to return always a real solution.
