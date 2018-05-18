from __future__ import print_function, division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import astropy.units as u
import astropy.coordinates as coord
from astropy.constants import G

import interact

def atlas_freespace(th=90, seed=51835):
    """"""
    
    # impact parameters
    M = 1e5*u.Msun
    B = 100*u.pc
    V = 100*u.km/u.s
    phi = coord.Angle(180*u.deg)
    theta=coord.Angle(th*u.deg)
    Tenc = 1*u.Gyr
    T = 1*u.Gyr
    dt = 0.1*u.Myr
    rs = 0*u.pc
    
    # setup tube
    Nstar = 500
    wx = 5*u.kpc
    wy = 0*u.pc
    wz = 0*u.pc
    sx = 0*u.km/u.s
    
    np.random.seed(seed)
    x = (np.random.rand(Nstar) - 0.5) * wx
    y = (np.random.randn(Nstar) - 0.5) * wy
    z = (np.random.randn(Nstar) - 0.5) * wz
    vx = (np.random.randn(Nstar) - 0.5) * sx
    vy = (np.random.randn(Nstar) - 0.5) * sx
    vz = (np.random.randn(Nstar) - 0.5) * sx
    
    x1, x2, x3, v1, v2, v3 = interact.interact(M.si.value, B.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
    stream = {}
    stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    ms = 8
    alpha = 0.3
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(8,8), sharex=True)
    plt.sca(ax[0])
    plt.plot(stream['x'][0], stream['x'][1], 'o', ms=ms, alpha=alpha)
    plt.ylabel('y (pc)')
    plt.xlim(-3000,3000)
    plt.ylim(-1100,20)
    plt.title('$\\theta$ = {:3.0f}$^\circ$'.format(th), fontsize='medium')
    
    # circle
    phi_ = np.linspace(0,2*np.pi,100)
    r = 0.05
    x0 = 0.3
    y0 = 0.65
    x = r*np.cos(phi_) + x0
    y = r*np.sin(phi_) + y0
    
    xp = r*np.cos(2*np.pi-theta.rad) + x0
    yp = r*np.sin(2*np.pi-theta.rad) + y0
    
    Ns = 9
    xs = np.linspace(-1.5*r, 1.5*r, Ns) + x0
    ys = np.zeros(Ns) + y0
    
    plt.plot(x, y, '-', color='0.3', alpha=0.5, lw=2, transform=fig.transFigure)
    plt.plot(xp, yp, 'o', color='0.3', ms=10, transform=fig.transFigure)
    plt.plot(xs, ys, 'o', color='tab:blue', ms=5, alpha=0.5, transform=fig.transFigure)
    
    plt.sca(ax[1])
    plt.plot(stream['x'][0], stream['x'][2], 'o', ms=ms, alpha=alpha)
    plt.ylabel('z (pc)')
    plt.xlabel('x (pc)')
    plt.xlim(-3000,3000)
    plt.ylim(-30,30)
    
    plt.tight_layout()
    plt.savefig('../plots/animations/angles/angles_{:03.0f}.png'.format(th/5))
    
def phases(seed=8264):
    """"""
    # impact parameters
    M = 1e5*u.Msun
    B = 100*u.pc
    V = 100*u.km/u.s
    phi = coord.Angle(180*u.deg)
    #theta = coord.Angle(th*u.deg)
    Tenc = 1*u.Gyr
    T = 1*u.Gyr
    dt = 0.1*u.Myr
    rs = 0*u.pc
    
    # setup tube
    Nstar = 500
    wx = 5*u.kpc
    wy = 2*u.pc
    wz = 2*u.pc
    sx = 0*u.km/u.s
    
    np.random.seed(seed)
    x = (np.random.rand(Nstar) - 0.5) * wx
    y = (np.random.randn(Nstar) - 0.5) * wy
    z = (np.random.randn(Nstar) - 0.5) * wz
    vx = (np.random.randn(Nstar) - 0.5) * sx
    vy = (np.random.randn(Nstar) - 0.5) * sx
    vz = (np.random.randn(Nstar) - 0.5) * sx
    
    angles = [5, 18, 90]
    times = [0.01]
    
    for th in angles:
        theta = coord.Angle(th*u.deg)
        T = B**2*V*np.abs(np.sin(theta.rad))/(2*G*M)
        times += [T.to(u.Gyr).value]
    
    times += [4]
    times = np.array(times) * u.Gyr
    
    cmap_navy = mpl.colors.LinearSegmentedColormap.from_list('cmap_navy', [(0,'#78aadd'), (1,'#00187f')], N=256)
    
    plt.close()
    fig, ax = plt.subplots(5, 3, figsize=(10,10), sharex=True)
    
    for et, T in enumerate(times):
        for ea, th in enumerate(angles):
            theta = coord.Angle(th*u.deg)
            p = (G*M*T/(B**2*V*np.abs(np.sin(theta.rad)))).decompose()
            print(et, T, p)
            
            x1, x2, x3, v1, v2, v3 = interact.interact(M.si.value, B.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
            stream = {}
            stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
            stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
                    
            plt.sca(ax[et][ea])
            clog = np.log10(p.value)
            cmin = np.log10(5e-2)
            cmax = np.log10(20)
            if clog<cmin: clog = cmin
            if clog>cmax: clog = cmax
            cindex = (clog - cmin)/(cmax - cmin)
            plt.plot(stream['x'][0], stream['x'][1], 'o', color=cmap_navy(cindex), ms=1.5, alpha=0.6)

            #plt.plot(stream['x'][0], stream['x'][1], 'o', color=cmap_navy(min(1.,p.value/7)), ms=1.5, alpha=0.6)
            
            txt = plt.text(0.9, 0.15, '$\psi$={:.2f}'.format(p), ha='right', va='center', transform=plt.gca().transAxes, fontsize='small')
            txt.set_bbox(dict(facecolor='w', alpha=0.7, ec='none'))
            
            if et==0:
                plt.title('$\\theta$ = {:.0f}$^\circ$'.format(th), fontsize='medium')
            
            if et==np.size(times)-1:
                plt.xlabel('x [pc]')
            
            if ea==0:
                plt.ylabel('y [pc]')
            
            if ea==np.size(angles)-1:
                plt.ylabel('T = {:.2f}'.format(T), labelpad=20, fontsize='small', rotation=270)
                plt.gca().yaxis.set_label_position('right')
            
    
    plt.tight_layout(h_pad=0.1, w_pad=0.15)
    plt.savefig('../plots/freespace_phases.png')
    plt.savefig('../plots/freespace_phases.pdf')

def change_b(seed=7356, th=90, case=0):
    """"""
    # impact parameters
    M = 5e6*u.Msun
    B = 100*u.pc
    V = 100*u.km/u.s
    phi = coord.Angle(180*u.deg)
    theta = coord.Angle(th*u.deg)
    Tenc = 1*u.Gyr
    T = 1*u.Gyr
    dt = 0.05*u.Myr
    rs = 0*u.pc
    
    # setup tube
    Nstar = 1000
    wx = 20*u.kpc
    wy = 2*u.pc
    wz = 2*u.pc
    sx = 0*u.km/u.s
    
    np.random.seed(seed)
    x = (np.random.rand(Nstar) - 0.5) * wx
    y = (np.random.randn(Nstar) - 0.5) * wy
    z = (np.random.randn(Nstar) - 0.5) * wz
    vx = (np.random.randn(Nstar) - 0.5) * sx
    vy = (np.random.randn(Nstar) - 0.5) * sx
    vz = (np.random.randn(Nstar) - 0.5) * sx
    
    f_array = np.array([0.5,1,2])
    p_array = np.array([0.1,0.3,0.5,1.,2.])
    cmap_borange = mpl.colors.LinearSegmentedColormap.from_list('cmap_borange', [(0,'#ff9e00'), (1,'#e63f25')], N=256)
    
    title_main = ['B$_0$, M$_0$', 'B$_0$, V$_0$', 'M$_0$, V$_0$', 'M$_0$, T$_0$']
    title_less = ['$\sqrt{0.5}$ B$_0$, 0.5 M$_0$', '$\sqrt{0.5}$ B$_0$, V$_0$ / 0.5', '0.5 M$_0$, 0.5 V$_0$', '0.5 M$_0$, T$_0$ / 0.5']
    title_more = ['$\sqrt{2}$ B$_0$, 2 M$_0$', '$\sqrt{2}$ B$_0$, V$_0$ / 2', '2 M$_0$, 2 V$_0$', '2 M$_0$, T$_0$ / 2']
    titles = [title_less, title_main, title_more]
    
    plt.close()
    fig, ax = plt.subplots(5,3,figsize=(10,6), sharex=True, sharey='row')
    
    for ep, p in enumerate(p_array):
        #p = (G*M*T/(B**2*V*np.abs(np.sin(theta.rad)))).decompose()
        B_ = np.sqrt(G*M*T/(p*V*np.abs(np.sin(theta.rad)))).to(u.pc)
        print(ep, B_)
        
        clog = np.log10(p)
        cmin = np.log10(0.1)
        cmax = np.log10(5)
        if clog<cmin: clog = cmin
        if clog>cmax: clog = cmax
        cindex = (clog - cmin)/(cmax - cmin)
        color = cmap_borange(cindex)
        
        for ef, f in enumerate(f_array[:]):
            fsq = np.sqrt(f)
            finv = 1/f
            #fsqinv = np.sqrt(finv)
            
            if case==0:
                x1, x2, x3, v1, v2, v3 = interact.interact(f*M.si.value, fsq*B_.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, (x*0.5/np.sqrt(p)).si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
            elif case==1:
                x1, x2, x3, v1, v2, v3 = interact.interact(M.si.value, fsq*B_.si.value, phi.rad, finv*V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
            elif case==2:
                x1, x2, x3, v1, v2, v3 = interact.interact(f*M.si.value, B_.si.value, phi.rad, f*V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
            elif case==3:
                x1, x2, x3, v1, v2, v3 = interact.interact(f*M.si.value, B_.si.value, phi.rad, V.si.value, theta.rad, finv*Tenc.si.value, finv*T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
            
            stream = {}
            stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
            stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
            
            plt.sca(ax[ep][ef])
            plt.plot(stream['x'][0]/(fsq*B_), stream['x'][1]/(fsq*B_), '.', color=color, ms=3, alpha=0.02)
            plt.gca().set_aspect('equal')
            
            if ep==0:
                plt.title(titles[ef][case], fontsize='medium')

            if ep==np.size(p_array)-1:
                plt.xlabel('x / B')
            
            if ef==0:
                plt.ylabel('y / B')
            
            if ef==np.size(f_array)-1:
                plt.ylabel('$\psi$ = {:.1f}'.format(p), labelpad=20, fontsize='small', rotation=270)
                plt.gca().yaxis.set_label_position('right')
            
            if f==1:
                txt = plt.text(0.1, 0.15, '$B_0$={:.0f}'.format(B_.to(u.pc)), ha='left', va='center', transform=plt.gca().transAxes, fontsize='small')
                txt.set_bbox(dict(facecolor='w', alpha=0.7, ec='none'))
            
    plt.tight_layout(h_pad=0.1, w_pad=0.15)
    plt.savefig('../plots/change_bscaled_{}.png'.format(case))
    #plt.savefig('../plots/change_b.pdf')


def scaling(seed=98, f=2):
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
    
    x1, x2, x3, v1, v2, v3 = interact.interact(M.si.value, B.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
    stream1 = {}
    stream1['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream1['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    finv = 1/f
    fsq = np.sqrt(f)
    
    x1, x2, x3, v1, v2, v3 = interact.interact(f*M.si.value, B.si.value, phi.rad, V.si.value, theta.rad, finv*Tenc.si.value, finv*T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
    
    stream2 = {}
    stream2['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream2['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)

    x1, x2, x3, v1, v2, v3 = interact.interact(f*M.si.value, B.si.value, phi.rad, f*V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
    
    stream3 = {}
    stream3['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream3['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)

    x1, x2, x3, v1, v2, v3 = interact.interact(M.si.value, B.si.value, phi.rad, f*V.si.value, theta.rad, f*Tenc.si.value, f*T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)

    stream4 = {}
    stream4['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream4['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    #x1, x2, x3, v1, v2, v3 = interact.interact(f*M.si.value, f*B.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)

    #stream5 = {}
    #stream5['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    #stream5['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    dblue = mpl.cm.Blues(0.8)
    lblue = mpl.cm.Blues(0.5)
    ms = 2
    streams = [stream1, stream2, stream3, stream4]
    labels = ['M,T,V', '{0:.1f}M,T/{0:.1f},V'.format(f), '{0:.1f}M,T,{0:.1f}V'.format(f),'M,{0:.1f}T,{0:.1f}V'.format(f), '{0:.1f}M,sqrt{0:.1f}B'.format(f)]
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    
    for e, stream in enumerate(streams):
        color = mpl.cm.Blues(e/5+0.1)
        ms = 14 - 2*e
        
        plt.sca(ax[0])
        plt.plot(stream['x'][0], stream['x'][1], 'o', color=color, ms=ms)
    
        plt.sca(ax[1])
        plt.plot(stream['x'][0], stream['x'][2], 'o', color=color, ms=ms, label=labels[e])
    
    plt.sca(ax[0])
    plt.xlabel('x (pc)')
    plt.ylabel('y (pc)')
    
    plt.sca(ax[1])
    plt.xlabel('x (pc)')
    plt.ylabel('z (pc)')
    plt.legend(fontsize='small', loc=1)
    
    plt.tight_layout()
    plt.savefig('../plots/scaling_{:.1f}.png'.format(f))

def scaling_norm(seed=473):
    """"""
    
    # impact parameters
    M = 1e5*u.Msun
    B = 100*u.pc
    V = 100*u.km/u.s
    phi = coord.Angle(180*u.deg)
    theta=coord.Angle(45*u.deg)
    Tenc = 1*u.Gyr
    T = 1*u.Gyr
    dt = 0.05*u.Myr
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
    
    # nonscaled
    x1, x2, x3, v1, v2, v3 = interact.interact(M.si.value, B.si.value, phi.rad, V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
    stream0 = {}
    stream0['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
    stream0['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
    
    # scaling
    Nf = 10
    f1_array = np.logspace(np.log10(0.2),np.log10(10),Nf)
    f2_array = np.logspace(np.log10(0.1),np.log10(5),Nf)
    f3_array = np.logspace(np.log10(np.sqrt(0.1)),np.log10(np.sqrt(5)),Nf)
    sigma_x_mt = np.zeros(Nf)
    sigma_v_mt = np.zeros(Nf)
    sigma_x_mv = np.zeros(Nf)
    sigma_v_mv = np.zeros(Nf)
    sigma_x_vt = np.zeros(Nf)
    sigma_v_vt = np.zeros(Nf)
    
    for e, f in enumerate(f1_array):
        finv = 1/f
        x1, x2, x3, v1, v2, v3 = interact.interact(f*M.si.value, B.si.value, phi.rad, V.si.value, theta.rad, finv*Tenc.si.value, finv*T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
        stream = {}
        stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
        stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
        
        sigma_x_mt[e] = np.linalg.norm(stream['x']-stream0['x'])/Nstar
        sigma_v_mt[e] = np.linalg.norm(stream['v']-stream0['v'])/Nstar
        
        f = f2_array[e]
        x1, x2, x3, v1, v2, v3 = interact.interact(f*M.si.value, B.si.value, phi.rad, f*V.si.value, theta.rad, Tenc.si.value, T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
        stream = {}
        stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
        stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
        
        sigma_x_mv[e] = np.linalg.norm(stream['x']-stream0['x'])/Nstar
        sigma_v_mv[e] = np.linalg.norm(stream['v']-stream0['v'])/Nstar
        
        f = f3_array[e]
        x1, x2, x3, v1, v2, v3 = interact.interact(M.si.value, B.si.value, phi.rad, f*V.si.value, theta.rad, f*Tenc.si.value, f*T.si.value, dt.si.value, x.si.value, y.si.value, z.si.value, vx.si.value, vy.si.value, vz.si.value)
        stream = {}
        stream['x'] = (np.array([x1, x2, x3])*u.m).to(u.pc)
        stream['v'] = (np.array([v1, v2, v3])*u.m/u.s).to(u.km/u.s)
        
        sigma_x_vt[e] = np.linalg.norm(stream['x']-stream0['x'])/Nstar
        sigma_v_vt[e] = np.linalg.norm(stream['v']-stream0['v'])/Nstar
    
    print(sigma_v_mt)
    print(sigma_v_mv)
    print(sigma_v_vt)
    
    dblue = mpl.cm.Blues(0.9)
    mblue = mpl.cm.Blues(0.7)
    lblue = mpl.cm.Blues(0.5)
    
    plt.close()
    fig, ax = plt.subplots(2,1,figsize=(8,8), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(1/f1_array, sigma_x_mt, '-', color=lblue, lw=6)
    plt.plot(f2_array, sigma_x_mv, '-', color=mblue, lw=4)
    plt.plot(f3_array**2, sigma_x_vt, color=dblue, lw=2)
    
    plt.ylabel('$\left< \Sigma_x\\right>$ [pc]')
    
    plt.sca(ax[1])
    plt.plot(1/f1_array, sigma_v_mt, '-', color=lblue, lw=6)
    plt.plot(f2_array, sigma_v_mv, '-', color=mblue, lw=4)
    plt.plot(f3_array**2, sigma_v_vt, color=dblue, lw=2)
    
    #plt.xlabel('f')
    plt.gca().set_yscale('log')
    plt.xlabel('Fastness')
    plt.ylabel('$\left< \Sigma_v \\right>$ [km s$^{-1}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/scaling_norm.png')
    
    
    
