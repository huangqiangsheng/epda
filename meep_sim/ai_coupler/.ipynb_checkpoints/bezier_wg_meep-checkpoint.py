%matplotlib auto
import bezier
import numpy as np
import meep as mp
import matplotlib.pyplot as plt

point0 = mp.Vector3(-8.0, 2.0)
cpoint0 = mp.Vector3(2.0,2.0)
cpoint1 = mp.Vector3(-2.0,-2.0)
point1 = mp.Vector3(8.0,-2.0)
width = 1.0

factor = np.linspace(0,1,21)

nodes = np.asfortranarray([[point0.x, cpoint0.x, cpoint1.x, point1.x],[point0.y-width/2.0, cpoint0.y -width/2.0, cpoint1.y-width/2.0, point1.y-width/2.0]])
curve = bezier.Curve(nodes, degree=3)
points1 = curve.evaluate_multi(factor)
nodes = np.asfortranarray([[point0.x, cpoint0.x, cpoint1.x, point1.x],[point0.y+width/2.0, cpoint0.y +width/2.0, cpoint1.y+width/2.0, point1.y+width/2.0]])
curve = bezier.Curve(nodes, degree=3)
points2 = curve.evaluate_multi(factor)

plt.plot(points1[0,:],points1[1,:])
plt.plot(points2[0,:],points2[1,:])

vertices = [mp.Vector3(x,y) for x,y in zip(points1[0,:],points1[1,:])]
tmp_ver = [mp.Vector3(x,y) for x,y in zip(points2[0,:],points2[1,:])]
vertices.extend([tmp for tmp in reversed(tmp_ver)])

cell = mp.Vector3(16, 8, 0)
geometry = [mp.Prism(vertices, height=mp.inf, center=mp.Vector3(), material=mp.Medium(epsilon=12))]

sources = [mp.Source(mp.ContinuousSource(frequency=0.15),
                     component=mp.Ez,
                     center=mp.Vector3(-7,2))]
pml_layers = [mp.PML(1.0)]
resolution = 10
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)
sim.run(until=200)

import numpy as np
import matplotlib.pyplot as plt
eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
plt.figure(dpi=100)
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.axis('off')
plt.show()



ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez, cmplx=True)
plt.figure(dpi=100)
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.imshow(ez_data.real.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
plt.axis('off')
plt.show()

