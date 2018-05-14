from __future__ import print_function, division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.coordinates as coord
from astropy.constants import G

import interact

def scaling(seed=98):
    """"""
    
    # impact parameters
    M = 1e5*u.Msun
    B = 100*u.pc
    V = 100*u.km/u.s
    phi = coord.Angle(180*u.deg)
    theta=coord.Angle(45*u.deg)
    Tenc = 1*u.Gyr
    T = 10*u.Gyr
    dt = 0.1*u.Myr
    rs = 0*u.pc
    
    # setup tube
    Nstar = 500
    wx = 5*u.kpc
    wy = 2*u.pc
    wz = 0*u.pc
    sx = 0*u.km/u.s
    
    np.random.seed(seed)
    x = (np.random.rand(Nstar) - 0.5) * wx
    y = (np.random.randn(Nstar) - 0.5) * wy
    z = (np.random.randn(Nstar) - 0.5) * wz
    vx = np.zeros(Nstar)*u.km/u.s
    vy = np.zeros(Nstar)*u.km/u.s
    vz = np.zeros(Nstar)*u.km/u.s
    
    # limits
    print('dense:{:.2g} << 1'.format(rs/B))
    print('fast: {:.2g} << 1'.format((G*M/(V**2*B)).decompose()) )
    print('thin: {:.2g} << 1'.format((np.sqrt(wy**2 + wz**2)/B).decompose()) )
    print('long: {:.2g} >> 1'.format((wx/B).decompose()) )
    print('cold: {:.2g} << 1'.format((sx/V).decompose()) )
    
    x1, x2, x3, v1, v2, v3 = interact.interact(M.si.value, B.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, 
                                                    x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    f = 1.5
    finv = 1/f
    fsq = np.sqrt(f)
    
    x1, x2, x3, v1, v2, v3 = interact.interact(f*M.si.value, B.si.value, phi.rad, V.si.value, theta.rad, finv*Tenc.si.value, finv*T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)

    x1, x2, x3, v1, v2, v3 = interact.interact(f*M.si.value, B.si.value, phi.rad, f*V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)

    x1, x2, x3, v1, v2, v3 = interact.interact(f*M.si.value, f*B.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)

    #x1, x2, x3, v1, v2, v3 = interact.interact(M.si.value, finv*B.si.value, phi.rad, f*V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)

    print('fast: {:.2g} << 1'.format((G*M/(f**2*V**2*finv*B)).decompose()) )

    stream2 = {}
    stream2['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream2['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    dblue = mpl.cm.Blues(0.8)
    lblue = mpl.cm.Blues(0.5)
    ms = 2
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    
    plt.sca(ax[0])
    #plt.plot(x.to(u.pc), y.to(u.pc), 'o', color=lblue, ms=ms)
    plt.plot(stream['x'][0], stream['x'][1], 'o', color=lblue, ms=2*ms)
    plt.plot(stream2['x'][0], stream2['x'][1], 'o', color=dblue, ms=ms)
    plt.xlabel('x (pc)')
    plt.ylabel('y (pc)')
    
    plt.sca(ax[1])
    #plt.plot(x.to(u.pc), z.to(u.pc), 'o', color=lblue, ms=ms, label='Initial')
    plt.plot(stream['x'][0], stream['x'][2], 'o', color=lblue, ms=2*ms)
    plt.plot(stream2['x'][0], stream2['x'][2], 'o', color=dblue, ms=ms)
    plt.xlabel('x (pc)')
    plt.ylabel('z (pc)')
    
    plt.tight_layout()
