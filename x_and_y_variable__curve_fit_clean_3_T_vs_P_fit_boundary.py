import numpy as np
from numpy import sqrt as sqrt
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sympy as sym
from sympy import lambdify
from matplotlib import cm
from sympy.solvers import solve
import subprocess
from itertools import chain

# Function to fit:
def func(X, a0, a1, a2, a3, a4, a5):
     x, y = X
     return a0 + a1*y + a2*x + a3*y**2 + a4*x**2  + a5*x*y 

# Load the raw T, P, G data: 17 volumes calcite I and 4 volumes calcite II, for instance:
y_data, z_data, x_data  = np.loadtxt('/home/david/Trabajo/structures/SCRIPT_ON_ALL_THE_PHASES/Calcite_I_over_17_volumes/SHRINK_3_3/crystal17_Pcrystal/volumes/grabbing_exact_value_of_freqs/solid_1__xyz_sorted_as_P_wise.dat').T

y_data_2, z_data_2, x_data_2  = np.loadtxt('/home/david/Trabajo/structures/SCRIPT_ON_ALL_THE_PHASES/Calcite_II_correct_description/scelphono_121_1-21_210__SHRINK_3_3/new/solid_1__xyz_sorted_as_P_wise.dat').T

# Calling non linear curve_fit
popt, pcov = curve_fit(func, (x_data, y_data), z_data) 
popt_2, pcov_2 = curve_fit(func, (x_data_2, y_data_2), z_data_2) 

print 'popt = ', popt
print 'pcov = ', pcov

perr = np.sqrt(np.diag(pcov))

print 'pcov = ', pcov
print 'perr = np.sqrt(np.diag(pcov))', perr


a0 =popt[0] 
a1 =popt[1] 
a2 =popt[2] 
a3 =popt[3] 
a4 =popt[4]    
a5 =popt[5]  
print 'a0 = ', a0
print 'a1 = ', a1
print 'a2 = ', a2
print 'a3 = ', a3
print 'a4 = ', a4
print 'a5 = ', a5

print 'a0 = ', a0

a0_s2 =popt_2[0] 
a1_s2 =popt_2[1] 
a2_s2 =popt_2[2] 
a3_s2 =popt_2[3] 
a4_s2 =popt_2[4]    
a5_s2 =popt_2[5]  

print 'a0_s2 = ', a0_s2

print """ 

The equations are the following:

G_I  (T, P) = a0 + a1*y + a2*x + a3*y**2 + a4*x**2  + a5*x*y
G_II (T, P) = a0_s2 + a1_s2*y + a2_s2*x + a3_s2*y**2 + a4_s2*x**2  + a5_s2*x*y

"""
print('G_I  (T, P) = ({a0}) + ({a1})*P + ({a2})*T  ({a3})*P**2  ({a4})*T**2  + ({a5})*T*P'.format(a0 = a0, a1 = a1, a2 = a2, a3 = a3, a4 = a4, a5 = a5))

print """
"""
print('G_II  (T, P) = ({a0_s2}) + ({a1_s2})*P + ({a2_s2})*T  ({a3_s2})*P**2  ({a4_s2})*T**2  + ({a5_s2})*T*P'.format(a0_s2 = a0_s2, a1_s2 = a1_s2, a2_s2 = a2_s2, a3_s2 = a3_s2, a4_s2 = a4_s2, a5_s2 = a5_s2))

print """
"""

print """
G_I  (T, P) = G_II (T, P)

"""
# Set the boundaries of T here:
x_mesh = np.linspace(x_data[0], x_data[-1], 20)
x_mesh_2 = np.linspace(x_data_2[0], x_data_2[-1], 20)
print x_mesh[0]
print x_mesh[-1]

print 'y_data[0] = ', y_data[0] 
print 'y_data[-1] = ', y_data[-1] 

print 'y_data_2[0] = ', y_data_2[0] 
print 'y_data_2[-1] = ', y_data_2[-1] 

# Set the boundaries for P here:
y_mesh = np.linspace(y_data[0], y_data[-1], 20)
y_mesh_2 = np.linspace(y_data_2[0], y_data_2[-1], 20)
print y_mesh[0]
print y_mesh[-1]

xx, yy = np.meshgrid(x_mesh, y_mesh)
xx_2, yy_2 = np.meshgrid(x_mesh_2, y_mesh_2)

z_fit = a0 + a1*yy + a2*xx + a3*yy**2 + a4*xx**2  + a5*xx*yy		
z_fit_2 = a0_s2 + a1_s2*yy_2 + a2_s2*xx_2 + a3_s2*yy_2**2 + a4_s2*xx_2**2  + a5_s2*xx_2*yy_2		

## Solving the intersection:
print """ 

The equations are the following:

z_I  (x, y) = a0 + a1*y + a2*x + a3*y**2 + a4*x**2  + a5*x*y
z_II (x, y) = a0_s2 + a1_s2*y + a2_s2*x + a3_s2*y**2 + a4_s2*x**2  + a5_s2*x*y

"""
print('z_I  (x, y) = ({a0}) + ({a1})*y + ({a2})*x  ({a3})*y**2  ({a4})*x**2  + ({a5})*x*y'.format(a0 = a0, a1 = a1, a2 = a2, a3 = a3, a4 = a4, a5 = a5))

print """
"""
print('z_II  (x, y) = ({a0_s2}) + ({a1_s2})*y + ({a2_s2})*x  ({a3_s2})*y**2  ({a4_s2})*x**2  + ({a5_s2})*x*y'.format(a0_s2 = a0_s2, a1_s2 = a1_s2, a2_s2 = a2_s2, a3_s2 = a3_s2, a4_s2 = a4_s2, a5_s2 = a5_s2))

print """
"""

print """
The intersection of both surfaces is satisfied when:

z_I(x, y) = z_II(x, y)

In other words, I am looking for the expression of the function y=y(x)

"""
# Setting "x" and "y" to be symbolic:
x, y = sym.symbols('x y', real=True)

def z_I(x,y):
        return   a0 + a1*y + a2*x + a3*y**2 + a4*x**2  + a5*x*y

def z_II(x,y):
        return   a0_s2 + a1_s2*y + a2_s2*x + a3_s2*y**2 + a4_s2*x**2  + a5_s2*x*y

sol = sym.solve(z_I(x,y) - z_II(x,y), y)
print 'sol =', sol

print 'sol[0] = ', sol[0]

cross = sym.solve(sol[0]-sol[1])
print ' cross = ', cross

####
y_sol_1 = sol[0]
y_sol_2 = sol[1]

y_sol_1.subs({x:2000.0})
print 'y_sol_1.subs({x:2000.0}) = ', y_sol_1.subs({x:2000.0})

print 'type(x) = ', type(x)
y = sym.symbols('y', real=True)
print 'type(x) = ', type(x)

##### Plotting:
# Use this to turn on matplotlib 1.5 defaults:
#style.use('classic')

# set "fig" and "ax" varaibles
fig = plt.figure()
ax = fig.gca(projection='3d')

# If you set "classic", use also this, 
# in order to capture z-labels right:
#z_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
#ax.zaxis.set_major_formatter(z_formatter)

# Plot the original function
ax.plot_surface(xx, yy, z_fit, color='y', alpha=0.5)
ax.plot_surface(xx_2, yy_2, z_fit_2, color='g', alpha=0.5)

# Plot the initial scattered points
ax.scatter(x_data, y_data, z_data, color='r', marker='o') # 'ro') #color='r', marker='o')
ax.scatter(x_data_2, y_data_2, z_data_2, color='b', marker='o' ) #, '^') 


ax.set_xlabel('T (K)')
ax.set_ylabel('P (GPa)')
ax.set_zlabel('\nGibbs free energy / F.unit (a.u.)', linespacing=3)


####### Calcite I scattered:
# In a new figure, each surface separately:
# set "fig" and "ax" varaibles
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the initial scattered points
ax.scatter(x_data, y_data, z_data, color='r', marker='o') # 'ro') #color='r', marker='o')

ax.set_xlabel('\nT (K)')
ax.set_ylabel('P (GPa)')
ax.set_zlabel('\nGibbs free energy / F.unit (a.u.)', linespacing=3)
ax.set_title('\n\nCalcite I', linespacing=3)
xlabels=[0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
ax.set_xticklabels(xlabels,rotation=90,
                  verticalalignment='baseline',#)#,
                  horizontalalignment='left')

ylabels = [-4, -2, 0, 2, 4, 6, 8, 10] 
ax.set_yticklabels(ylabels,rotation=0,
                  verticalalignment='baseline')#,
#                 horizontalalignment='left')

fig.savefig("Calcite_I_scattered.pdf",  bbox_inches='tight', pad_inches=0.3)#, tight_layout() )#, bbox_inches=bbox)

###### Calcite I surface ########
# In a new figure, each surface separately:
# set "fig" and "ax" varaibles
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the original function
ax.plot_surface(xx, yy, z_fit, color='y', alpha=0.5)

# Plot the initial scattered points
ax.scatter(x_data, y_data, z_data, color='r', marker='o') # 'ro') #color='r', marker='o')

ax.set_xlabel('\nT (K)')
ax.set_ylabel('P (GPa)')
ax.set_zlabel('\nGibbs free energy / F.unit (a.u.)', linespacing=3)
ax.set_title('\n\nCalcite I', linespacing=3)
xlabels=[0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
ax.set_xticklabels(xlabels,rotation=90,
                  verticalalignment='baseline',#)#,
                  horizontalalignment='left')

ylabels = [-4, -2, 0, 2, 4, 6, 8, 10] 
ax.set_yticklabels(ylabels,rotation=0,
                  verticalalignment='baseline')#,
#                 horizontalalignment='left')

fig.savefig("Calcite_I.pdf" )#, bbox_inches=bbox)

######  Calcite II scattered ###########
# In a new figure, each surface separately:
# set "fig" and "ax" varaibles
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the initial scattered points
ax.scatter(x_data_2, y_data_2, z_data_2, '^s') #color='r', marker='o')

ax.set_xlabel('\nT (K)')
ax.set_ylabel('P (GPa)')
ax.set_zlabel('\nGibbs free energy / F.unit (a.u.)', linespacing=3)
ax.set_title('\n\nCalcite II', linespacing=3)

xlabels=[0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
ax.set_xticklabels(xlabels,rotation=90,
                  verticalalignment='baseline',#)#,
                  horizontalalignment='left')

ylabels = [8, 10, 12, 14, 16]
ax.set_yticklabels(ylabels,rotation=0,
                  verticalalignment='baseline')#,
#                 horizontalalignment='left')

fig.savefig("Calcite_II_scattered.pdf" )#, bbox_inches=bbox)

######  Calcite II surface ###########
# In a new figure, each surface separately:
# set "fig" and "ax" varaibles
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the original function
ax.plot_surface(xx_2, yy_2, z_fit_2, color='g', alpha=0.5)

# Plot the initial scattered points
ax.scatter(x_data_2, y_data_2, z_data_2, '^s') #color='r', marker='o')


ax.set_xlabel('\nT (K)')
ax.set_ylabel('P (GPa)')
ax.set_zlabel('\nGibbs free energy / F.unit (a.u.)', linespacing=3)
ax.set_title('\n\nCalcite II', linespacing=3)

xlabels=[0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
ax.set_xticklabels(xlabels,rotation=90,
                  verticalalignment='baseline',#)#,
                  horizontalalignment='left')

ylabels = [8, 10, 12, 14, 16]
ax.set_yticklabels(ylabels,rotation=0,
                  verticalalignment='baseline')#,
#                 horizontalalignment='left')

fig.savefig("Calcite_II.pdf" )#, bbox_inches=bbox)

###### Calcite I and II surfaces ########
# In a new figure, each surface separately:
# set "fig" and "ax" varaibles
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the original function
ax.plot_surface(xx, yy, z_fit, color='y', alpha=0.5)

# Plot the initial scattered points
ax.scatter(x_data, y_data, z_data, color='r', marker='o') # 'ro') #color='r', marker='o')

# Plot the original function
ax.plot_surface(xx_2, yy_2, z_fit_2, color='g', alpha=0.5)

# Plot the initial scattered points
ax.scatter(x_data_2, y_data_2, z_data_2, '^s') #color='b', marker='o') #'^s') 

ax.set_xlabel('\nT (K)')
ax.set_ylabel('P (GPa)')
ax.set_zlabel('\nGibbs free energy / F.unit (a.u.)', linespacing=3)
ax.set_title('\n\nCalcite I and II', linespacing=3)
xlabels=[0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
ax.set_xticklabels(xlabels,rotation=90,
                  verticalalignment='baseline',#)#,
                  horizontalalignment='left')

ylabels = [-5, 0, 5, 10, 15] 
ax.set_yticklabels(ylabels,rotation=0,
                  verticalalignment='baseline')#,
#                 horizontalalignment='left')

fig.savefig("Calcite_I_and_II.pdf" )#, bbox_inches=bbox)



###### Calcite I and II surfaces ########
# In a new figure, each surface separately:
# set "fig" and "ax" varaibles
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the original function
ax.plot_surface(xx, yy, z_fit, color='y', alpha=0.9) #, opacity=0.9)

# Plot the initial scattered points
ax.scatter(x_data, y_data, z_data, color='r', marker='o') # 'ro') #color='r', marker='o')

# Plot the original function
ax.plot_surface(xx_2, yy_2, z_fit_2, color='g', alpha=0.5)

# Plot the initial scattered points
ax.scatter(x_data_2, y_data_2, z_data_2, '^s') #color='b', marker='o') #'^s') 

ax.set_xlabel('\nT (K)')
ax.set_ylabel('P (GPa)')
ax.set_zlabel('\nGibbs free energy / F.unit (a.u.)', linespacing=3)
ax.set_title('\n\nCalcite I and II', linespacing=3)
xlabels=[0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
ax.set_xticklabels(xlabels,rotation=90,
                  verticalalignment='baseline',#)#,
                  horizontalalignment='left')

ylabels = [-5, 0, 5, 10, 15] 
ax.set_yticklabels(ylabels,rotation=0,
                  verticalalignment='baseline')#,
#                 horizontalalignment='left')

fig.savefig("Calcite_I_and_II_opaque.pdf" )#, bbox_inches=bbox)


###### Calcite I and II scattered ########
# In a new figure, each surface separately:
# set "fig" and "ax" varaibles
fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the initial scattered points
ax.scatter(x_data, y_data, z_data, color='r', marker='o') # 'ro') #color='r', marker='o')

# Plot the initial scattered points
ax.scatter(x_data_2, y_data_2, z_data_2, '^s') #color='r', marker='o')

ax.set_xlabel('\nT (K)')
ax.set_ylabel('P (GPa)')
ax.set_zlabel('\nGibbs free energy / F.unit (a.u.)', linespacing=3)
ax.set_title('\n\nCalcite I and II', linespacing=3)
xlabels=[0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
ax.set_xticklabels(xlabels,rotation=90,
                  verticalalignment='baseline',#)#,
                  horizontalalignment='left')

ylabels = [-5, 0, 5, 10, 15] 
ax.set_yticklabels(ylabels,rotation=0,
                  verticalalignment='baseline')#,
#                 horizontalalignment='left')

fig.savefig("Calcite_I_and_II_scattered.pdf" )#, bbox_inches=bbox)


lam_y_sol_1 = lambdify(x, y_sol_1, modules=['numpy'])
lam_y_sol_2 = lambdify(x, y_sol_2, modules=['numpy'])

# New figure for the y=y(x) function:
fig = plt.figure()
x_vals = np.linspace(10.0, 2000.0, 100)

y_vals_sol_1 = lam_y_sol_1(x_vals)
y_vals_sol_2 = lam_y_sol_2(x_vals)

plt.plot(x_vals, y_vals_sol_1, "fuchsia")
plt.plot(x_vals, y_vals_sol_2, "black")

plt.xlabel('T (K)')
plt.ylabel('P (GPa)')
plt.title('Exact expression of P=P(T)\nas a result of making $G^{I}(T,P)=G^{II}(T,P)$')
tics_shown =  [10, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250]
plt.xticks(tics_shown)
plt.grid()
fig.savefig("two_solutions.pdf",  bbox_inches='tight' )

# New figure for the y=y(x) function in circle:
fig = plt.figure()
x_vals_circle = np.linspace(10.0, 2000.0*100, 10000)

y_vals_sol_1_circle = lam_y_sol_1(x_vals_circle)
y_vals_sol_2_circle = lam_y_sol_2(x_vals_circle)

plt.plot(x_vals_circle, y_vals_sol_1_circle, "fuchsia")
plt.plot(x_vals_circle, y_vals_sol_2_circle, "black")

plt.xlabel('T (K)')
plt.ylabel('P (GPa)')
plt.title('Exact expression of P=P(T)\nas a result of making $G^{I}(T,P)=G^{II}(T,P)$')
plt.grid()
fig.savefig("circle_2_solutions_T_1e4_points.pdf",  bbox_inches='tight' )


print 'Performing the collisions program....'

# PERFORMING THE MATCHES with the 100 volumes::
# Load data:
# x -> T 
# y -> P
# z -> G

# Load the interpolated T, P, G data: 100 data points, for instance:
V1, y1, x1, z1  = np.loadtxt('/home/david/Trabajo/structures/SCRIPT_ON_ALL_THE_PHASES/Calcite_I_over_17_volumes/SHRINK_3_3/crystal17_Pcrystal/volumes/grabbing_exact_value_of_freqs/USING_100_VOLUMES/Vs_Ps_Gs.dat').T

V2, y2, x2, z2  = np.loadtxt('/home/david/Trabajo/structures/SCRIPT_ON_ALL_THE_PHASES/Calcite_II_correct_description/scelphono_121_1-21_210__SHRINK_3_3/new/USING_100_VOLUMES/Vs_Ps_Gs.dat').T


def within_tolerance(p1, p2):
    tol_z = 0.005       # Tests: tol_z = 0.00005 = tol_y, tol_z = 20.0 does not work
    tol_x = 20.0        #        tol_z = 0.0005 = tol_y, tol_z = 20.0 does not work
    tol_y = 0.005 

    x1, y1, z1 = p1
    x2, y2, z2 = p2

    return abs(x1 - x2) < tol_x and abs(y1 - y2) < tol_y and abs(z1 - z2) < tol_z

points_1 = list(zip(x1, y1, z1))
points_2 = list(zip(x2, y2, z2))

collisions_2 = []

for p1 in points_1:
    matches = [p2 for p2 in points_2 if within_tolerance(p1, p2)]
    collisions_2.append(matches)

collisions_1 = []

for p2 in points_2:
    matches = [p1 for p1 in points_1 if within_tolerance(p1, p2)]
    collisions_1.append(matches)

print 'collisions_1 = ', collisions_1
print 'collisions_2 = ', collisions_2

collisions_1 = [i for i in chain.from_iterable(collisions_1)]
print  'collisions in calcite 1 = ', collisions_1 

collisions_2 = [i for i in chain.from_iterable(collisions_2)]
print  'collisions in calcite 2 = ', collisions_2 


output_array_1 = np.vstack((collisions_1))
np.savetxt('collisions_1.dat', output_array_1, header="T(K) \t            P(GPa) \t   G per F unit (a.u)", fmt="%0.13f")

output_array_2 = np.vstack((collisions_2))
np.savetxt('collisions_2.dat', output_array_2, header="T(K) \t            P(GPa) \t   G per F unit (a.u)", fmt="%0.13f")

T1, P1, G1  = np.loadtxt('./collisions_1.dat').T

T2, P2, G2  = np.loadtxt('./collisions_2.dat').T


# Quadratic fit of T=T(P):
fitting = np.polyfit(P1, T1, 1)
fit = np.poly1d(fitting)

print """
HHHHHHHHHHHHHHHHHHHHHHHHHHHHH
HHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
fit = """, fit

fitting = np.polyfit(P1, T1, 2)
fit = np.poly1d(fitting)

print """
HHHHHHHHHHHHHHHHHHHHHHHHHHHHH
HHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
fit = """, fit

fitting = np.polyfit(P1, T1, 3)
fit = np.poly1d(fitting)

print """
HHHHHHHHHHHHHHHHHHHHHHHHHHHHH
HHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
fit = """, fit


# If we want the Regression coefficcient:

# Polynomial Regression
def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results

All_in_one_1st_degree = polyfit(P1, T1, 1)
All_in_one_2nd_degree = polyfit(P1, T1, 2)
All_in_one_3rd_degree = polyfit(P1, T1, 3)

print 'All_in_one_1st_degree = ',All_in_one_1st_degree 
print 'All_in_one_2nd_degree = ',All_in_one_2nd_degree 
print 'All_in_one_3rd_degree = ',All_in_one_3rd_degree 
#sys.exit()

# New figure for the matching: (= m)
fig = plt.figure()
x_vals_m = np.linspace(10.0, 2000.0, 100)

y_vals_sol_1_m = lam_y_sol_1(x_vals_m)
y_vals_sol_2_m = lam_y_sol_2(x_vals_m)
print 'y_vals_sol_1_m = ', y_vals_sol_1_m
print 'y_vals_sol_2_m = ', y_vals_sol_2_m

plt.plot(y_vals_sol_1_m, x_vals_m, "fuchsia")
plt.plot(y_vals_sol_2_m, x_vals_m, "black")

plt.ylabel('T (K)')
plt.xlabel('P (GPa)')

plt.scatter(P1, T1, color='r', marker='o', label='Calcite I (P,T) data \nfor which $G^{I}, T^{I}, P^{I} = G^{II}, T^{II}, P^{II}$')
plt.scatter(P2, T2, color='b', marker='o', label='Calcite II (P,T) data \nfor which $G^{I}, T^{I}, P^{I} = G^{II}, T^{II}, P^{II}$')
tics_shown =  [10, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250]
plt.grid()

plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)

fig.savefig("Matching_1_and_2_on_G_T_and_P_T_vs_P.pdf",  bbox_inches='tight', pad_inches=0.3)#, tight_layout() )#, bbox_inches=bbox)


fitting = np.polyfit(P1, T1, 1)
fit = np.poly1d(fitting)

# New figure for the matching: (= m)
fig = plt.figure()
x_vals_m = np.linspace(10.0, 2000.0, 100)

#xp = np.linspace(7.6478503874853, 10.7114925964187, 100)
xp = np.linspace(min(P1), max(P1), 100)

y_vals_sol_1_m = lam_y_sol_1(x_vals_m)
y_vals_sol_2_m = lam_y_sol_2(x_vals_m)

plt.plot(y_vals_sol_1_m, x_vals_m, "fuchsia") #, label='\nSolutions for the quadratic fit')
plt.plot(y_vals_sol_2_m, x_vals_m, "black") #, label='\n')

# The fit for the collisions:
plt.plot(xp, fit(xp), "green")

plt.ylabel('T (K)')
plt.xlabel('P (GPa)')

plt.scatter(P1, T1, color='r', marker='o', label='Calcite I (P,T) data \nfor which $G^{I}, T^{I}, P^{I} = G^{II}, T^{II}, P^{II}$')
plt.scatter(P2, T2, color='b', marker='o', label='Calcite II (P,T) data \nfor which $G^{I}, T^{I}, P^{I} = G^{II}, T^{II}, P^{II}$')
tics_shown =  [10, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250]
plt.grid()

plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)
#plt.ylim(, )  # If you want to limit the x-axis
fig.savefig("Matching_1_and_2_on_G_T_and_P_T_vs_P_fit_1st_degree.pdf",  bbox_inches='tight', pad_inches=0.3)#, tight_layout() )#, bbox_inches=bbox)


fitting = np.polyfit(P1, T1, 2)
fit = np.poly1d(fitting)

# New figure for the matching: (= m)
fig = plt.figure()
x_vals_m = np.linspace(10.0, 2000.0, 100)

#xp = np.linspace(7.6478503874853, 10.7114925964187, 100)
xp = np.linspace(min(P1), max(P1), 100)

y_vals_sol_1_m = lam_y_sol_1(x_vals_m)
y_vals_sol_2_m = lam_y_sol_2(x_vals_m)

plt.plot(y_vals_sol_1_m, x_vals_m, "fuchsia")
plt.plot(y_vals_sol_2_m, x_vals_m, "black")

# The fit for the collisions:
plt.plot(xp, fit(xp), "green")

plt.ylabel('T (K)')
plt.xlabel('P (GPa)')

plt.scatter(P1, T1, color='r', marker='o', label='Calcite I (P,T) data \nfor which $G^{I}, T^{I}, P^{I} = G^{II}, T^{II}, P^{II}$')
plt.scatter(P2, T2, color='b', marker='o', label='Calcite II (P,T) data \nfor which $G^{I}, T^{I}, P^{I} = G^{II}, T^{II}, P^{II}$')
tics_shown =  [10, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250]
plt.grid()

plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)

fig.savefig("Matching_1_and_2_on_G_T_and_P_T_vs_P_fit_2nd_degree.pdf",  bbox_inches='tight', pad_inches=0.3)#, tight_layout() )#, bbox_inches=bbox)


fitting = np.polyfit(P1, T1, 3)
fit = np.poly1d(fitting)

# New figure for the matching: (= m)
fig = plt.figure()
x_vals_m = np.linspace(10.0, 2000.0, 100)

#xp = np.linspace(7.6478503874853, 10.7114925964187, 100)
xp = np.linspace(min(P1), max(P1), 100)

y_vals_sol_1_m = lam_y_sol_1(x_vals_m)
y_vals_sol_2_m = lam_y_sol_2(x_vals_m)

plt.plot(y_vals_sol_1_m, x_vals_m, "fuchsia")
plt.plot(y_vals_sol_2_m, x_vals_m, "black")

# The fit for the collisions:
plt.plot(xp, fit(xp), "green")

plt.ylabel('T (K)')
plt.xlabel('P (GPa)')

plt.scatter(P1, T1, color='r', marker='o', label='Calcite I (P,T) data \nfor which $G^{I}, T^{I}, P^{I} = G^{II}, T^{II}, P^{II}$')
plt.scatter(P2, T2, color='b', marker='o', label='Calcite II (P,T) data \nfor which $G^{I}, T^{I}, P^{I} = G^{II}, T^{II}, P^{II}$')
tics_shown =  [10, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250]
plt.grid()

plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)

fig.savefig("Matching_1_and_2_on_G_T_and_P_T_vs_P_fit_3rd_degree.pdf",  bbox_inches='tight', pad_inches=0.3)#, tight_layout() )#, bbox_inches=bbox)

subprocess.call("./trimming.sh", shell=True)

plt.show()

