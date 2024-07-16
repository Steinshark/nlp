import numpy as np
import matplotlib.pyplot as plt

sigma=5.67e-8

def find_tau(tot_trans,num_layers):
    trans_layer=tot_trans**(1./num_layers)
    tau_layer= -1.*np.log(trans_layer)
    tau_layers=np.ones([num_layers])*tau_layer
    tau_levels=np.cumsum(tau_layers)
    tau_levels=np.concatenate(([0],tau_levels))
    return tau_levels

def find_heights(press_levels,rho_layers):
    Rd=287.
    g=9.8
    press_layers=(press_levels[1:] + press_levels[:-1])/2.
    del_press=(press_levels[1:] - press_levels[0:-1])
    rho_layers=press_layers/(Rd*Temp_layers)
    del_z= -1.*del_press/(rho_layers*g)
    level_heights=np.cumsum(del_z)
    level_heights=np.concatenate(([0],level_heights))
    return level_heights

def fluxes(tau_levels,Temp_layers,T_surf):
    up_rad=np.empty_like(tau_levels)
    down_rad=np.empty_like(tau_levels)
    sfc_rad=sigma*T_surf**4.
    up_rad[0]=sfc_rad
    tot_levs=len(tau_levels)
    for index in np.arange(1,tot_levs):
        upper_lev=index
        lower_lev=index - 1
        layer_num=index-1
        del_tau=tau_levels[upper_lev] - tau_levels[lower_lev]
        trans=np.exp(-1.666*del_tau)
        emiss=1 - trans
        layer_rad=sigma*Temp_layers[layer_num]**4.*emiss
        #
        # find the radiance at the next level
        #
        up_rad[upper_lev]=trans*up_rad[lower_lev] + layer_rad
    #
    # start at the top of the atmosphere
    # with zero downwelling flux
    #
    down_rad[tot_levs-1]=0
    #
    # go down a level at a time, adding up the radiances
    #
    for index in np.arange(1,tot_levs):
        upper_lev=tot_levs - index
        lower_lev=tot_levs - index -1
        layer_num=tot_levs - index - 1
        del_tau=tau_levels[upper_lev] - tau_levels[lower_lev]
        trans=np.exp(-1.666*del_tau)
        emiss=1 - trans
        layer_rad=sigma*Temp_layers[layer_num]**4.*emiss
        down_rad[lower_lev]=down_rad[upper_lev]*trans + layer_rad
    return (up_rad,down_rad)

def heating_rate(net_up,height_levels,rho_layers):
    cpd=1004.
    #
    # find the radiance divergence across the layer
    # by differencing the levels
    #
    dFn_dz= -1.*np.diff(net_up)/np.diff(height_levels)
    dT_dt=dFn_dz/(rho_layers*cpd)
    return dT_dt

if __name__=="__main__":
    
    tot_trans=0.2
    num_layers=100
    p_sfc=1000.*1.e2
    p_top=100.*1.e2
    g=9.8
    T_sfc=300.
    Rd=287. #J/kg/K
    num_levels=num_layers+1
    tau_levels=find_tau(tot_trans,num_layers)
    press_levels=np.linspace(p_top,p_sfc,num_levels)
    press_diff=np.diff(press_levels)[0]
    press_levels=press_levels[::-1]
    press_layers=(press_levels[1:] + press_levels[:-1])/2.
    Temp_levels=np.ones([num_levels])*T_sfc
    Temp_layers=(Temp_levels[1:] + Temp_levels[:-1])/2.
    rho_levels=press_levels/(Rd*Temp_levels)
    rho_layers=(rho_levels[1:] + rho_levels[:-1])/2.
    height_levels=find_heights(press_levels,rho_layers)
    up,down=fluxes(tau_levels,Temp_layers,T_sfc)
    dT_dt=heating_rate(up - down,height_levels,rho_layers)
    
    plt.close('all')
    fig1,axis1=plt.subplots(1,1)
    axis1.plot(up,height_levels*0.001,'b-',lw=5,label='upward flux')
    axis1.plot(down,height_levels*0.001,'g-',lw=5,label='downward lux')
    axis1.set_title('upward and downward fluxes')
    axis1.set_xlabel('flux $(W\,m^{-2})$')
    axis1.set_ylabel('height_levels (km)')
    axis1.legend(numpoints=1,loc='best')
    fig1.savefig('simple_flux_updn.png')

    
    fig2,axis2=plt.subplots(1,1)
    axis2.plot(up-down,height_levels*0.001,'b-',lw=5)
    axis2.set_title('net upward flux')
    axis2.set_xlabel('net upward flux $(W\,m^{-2})$')
    axis2.set_ylabel('heights (km)')
    fig2.savefig('simple_flux_net.png')

    dT_dt=heating_rate(up - down,height_levels,rho_layers)
    fig3,axis3=plt.subplots(1,1)
    #
    #find the height at mid-layer
    #
    layer_height=(height_levels[1:] + height_levels[:-1])/2.
    axis3.plot(dT_dt*3600.*24.,layer_height*0.001,'b-',lw=5)
    axis3.set_title('heating rate in K/day')
    axis3.set_xlabel('heating rate (K/day)')
    axis3.set_ylabel('height (km)')
    fig3.savefig('simple_heating.png')


    plt.show()

    
