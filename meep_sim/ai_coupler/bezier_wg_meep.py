import bezier
import numpy as np
import meep as mp
import matplotlib.pyplot as plt
import waveguide

width = 1.0 #wg width
fcen = 0.15  # pulse center frequency
df = 0.2 * fcen     # pulse width (in frequency)
resolution = 10

sx = 16.0
sy = 16.0
cell = mp.Vector3(sx, sy, 0)

dpml = 1.0
pml_layers = [mp.PML(dpml)]

geometry = [mp.Block(size=mp.Vector3(mp.inf,width,mp.inf),
                     center=mp.Vector3(0,0,0),
                     material=mp.Medium(epsilon=12))]

# sources = [mp.Source(mp.GaussianSource(fcen,fwidth=df), component=mp.Ez,
#                      center=mp.Vector3(-0.5*sx + dmpl,0,0),size=mp.Vector3(0,width,0))]

src_pt = mp.Vector3(-0.5*sx + dpml,0)
sources = [mp.EigenModeSource(src=mp.GaussianSource(fcen,fwidth=df),
                                component=mp.Ez,
                                center= src_pt,
                                size=mp.Vector3(0,6,0),
                                eig_match_freq=True,
                                eig_parity=mp.ODD_Z)]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

nfreq = 3  # number of frequencies at which to compute flux

# reflected flux
refl_fr = mp.FluxRegion(center=mp.Vector3(-0.5*sx+dpml+0.5,0,0),size=mp.Vector3(0,2*width,0))
refl = sim.add_flux(fcen,df,nfreq,refl_fr)

# transmitted flux
tran_fr = mp.FluxRegion(center=mp.Vector3(0.5*sx-dpml,0,0),size=mp.Vector3(0,2*width,0))
tran = sim.add_flux(fcen,df,nfreq,tran_fr)

pt = mp.Vector3(0.5*sx-dpml-0.5,0.0)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50,mp.Ez,pt,1e-3))

# for normalization run, save flux fields data for reflection plane
straight_refl_data = sim.get_flux_data(refl)

# save incident power for transmission plane
straight_tran_flux = mp.get_fluxes(tran)

# bend
sim.reset_meep()

point0 = mp.Vector3(-8.0, -2.0)  # start point
cpoint0 = mp.Vector3(2.0,-2.0)   # control point1
cpoint1 = mp.Vector3(-2.0,2.0)   # control point2
point1 = mp.Vector3(8.0,2.0)    # end point

factor = np.linspace(0,1,41)  # sample number

nodes = np.asfortranarray([[point0.x, cpoint0.x, cpoint1.x, point1.x],[point0.y, cpoint0.y, cpoint1.y, point1.y]])
curve = bezier.Curve(nodes, degree=3)
points1 = curve.evaluate_multi(factor)
new_points = points1.transpose()
wg = waveguide.Waveguide(new_points, width)
poly = wg.poly()
tmp_poly = np.asarray(poly)

# plt.figure(1)
# plt.plot(tmp_poly[:,0],tmp_poly[:,1],'-')
# plt.plot(points1[0,:],points1[1,:],'-')

point0 = mp.Vector3(-6.0, 0.0)
cpoint0 = mp.Vector3(-6.0,0.0)
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
# plt.plot(tmp_poly[:,0],tmp_poly[:,1],'-')

vertices1 = [mp.Vector3(tmp[0],tmp[1]) for tmp in poly]
vertices2 = [mp.Vector3(tmp[0],tmp[1]) for tmp in poly2]
#vertices.extend([tmp for tmp in reversed(tmp_ver)])

geometry = [mp.Prism(vertices1, height=mp.inf, center=mp.Vector3(0.0,2.0), material=mp.Medium(epsilon=12)),
            mp.Prism(vertices2, height=mp.inf, center=mp.Vector3(1.0,-2.0), material=mp.Medium(epsilon=12))]

# sources = [mp.Source(mp.ContinuousSource(frequency=0.15),
#                     component=mp.Ez,
#                     center=mp.Vector3(-7,0))]

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

# reflected flux
refl = sim.add_flux(fcen, df, nfreq, refl_fr)

tran_fr = mp.FluxRegion(center=mp.Vector3(0.5 * sx - dpml - 0.5,4.0,0),size=mp.Vector3(0.0,2.0*width,0))
tran = sim.add_flux(fcen,df,nfreq,tran_fr)

# for normal run, load negated fields to subtract incident from refl. fields
sim.load_minus_flux_data(refl,straight_refl_data)


pt = mp.Vector3(4.0 , 0.5 * sx - dpml - 0.5)

sim.run(mp.at_time(200, mp.output_efield_z),until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))

bend_refl_flux = mp.get_fluxes(refl)
bend_tran_flux = mp.get_fluxes(tran)
flux_freqs = mp.get_flux_freqs(refl)

wl = []
Rs = []
Ts = []

for i in range(0,nfreq):
    wl = np.append(wl, 1/flux_freqs[i])
    Rs = np.append(Rs,-bend_refl_flux[i]/straight_tran_flux[i])
    Ts = np.append(Ts,bend_tran_flux[i]/straight_tran_flux[i])

plt.plot(wl,Rs,'bo-',label='reflectance')
plt.plot(wl,Ts,'ro-',label='transmittance')
plt.plot(wl,1-Rs-Ts,'go-',label='loss')
plt.xlabel(r'wavelength $\mu$ m')
plt.legend(loc="upper right")
plt.show()

eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
plt.figure(dpi=100)
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
plt.axis('off')

ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez, cmplx=True)
plt.figure(dpi=100)
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary', extent=[-8,8,8,-8])
plt.imshow(ez_data.real.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9, extent=[-8,8,8,-8])
plt.show()