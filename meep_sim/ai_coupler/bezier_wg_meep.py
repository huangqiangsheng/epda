import bezier
import numpy as np
import meep as mp
import matplotlib.pyplot as plt
import waveguide

point0 = mp.Vector3(-8.0, -2.0)
cpoint0 = mp.Vector3(2.0,-2.0)
cpoint1 = mp.Vector3(-2.0,2.0)
point1 = mp.Vector3(8.0,2.0)
width = 1.0

factor = np.linspace(0,1,41)


nodes = np.asfortranarray([[point0.x, cpoint0.x, cpoint1.x, point1.x],[point0.y, cpoint0.y, cpoint1.y, point1.y]])
curve = bezier.Curve(nodes, degree=3)
points1 = curve.evaluate_multi(factor)
new_points = points1.transpose()
wg = waveguide.Waveguide(new_points, width)
poly = wg.poly()
tmp_poly = np.asarray(poly)

plt.figure(1)
plt.plot(tmp_poly[:,0],tmp_poly[:,1],'-')
plt.plot(points1[0,:],points1[1,:],'-')

point0 = mp.Vector3(-8.0, 0.0)
cpoint0 = mp.Vector3(-8.0,0.0)
cpoint1 = mp.Vector3(8.0,0.0)
point1 = mp.Vector3(8.0,0.0)
width = 1.0
factor = np.linspace(0,1,41)
nodes = np.asfortranarray([[point0.x, cpoint0.x, cpoint1.x, point1.x],[point0.y, cpoint0.y, cpoint1.y, point1.y]])
curve = bezier.Curve(nodes, degree=3)
points1 = curve.evaluate_multi(factor)
new_points = points1.transpose()
wg = waveguide.Waveguide(new_points, width)
poly2 = wg.poly()
tmp_poly = np.asarray(poly)
plt.plot(tmp_poly[:,0],tmp_poly[:,1],'-')

if True:
    vertices1 = [mp.Vector3(tmp[0],tmp[1]) for tmp in poly]
    vertices2 = [mp.Vector3(tmp[0],tmp[1]) for tmp in poly2]
    #vertices.extend([tmp for tmp in reversed(tmp_ver)])

    cell = mp.Vector3(16, 16, 0)
    geometry = [mp.Prism(vertices1, height=mp.inf, center=mp.Vector3(0.0,2.0), material=mp.Medium(epsilon=12)),
                mp.Prism(vertices2, height=mp.inf, center=mp.Vector3(0.0,-2.0), material=mp.Medium(epsilon=12))]

    sources = [mp.Source(mp.ContinuousSource(frequency=0.15),
                        component=mp.Ez,
                        center=mp.Vector3(-7,0))]
    pml_layers = [mp.PML(1.0)]
    resolution = 10
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution)
    sim.run(until=200)

    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    plt.figure(dpi=100)
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    plt.axis('off')

    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez, cmplx=True)
    plt.figure(dpi=100)
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary', extent=[-8,8,8,-8])
    plt.imshow(ez_data.real.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9, extent=[-8,8,8,-8])
    plt.show()

