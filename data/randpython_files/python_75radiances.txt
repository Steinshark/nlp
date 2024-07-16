import numpy as np
import matplotlib.pyplot as plt

sigma_pi=5.67e-8/np.pi

def hydrostat(T_surf,p_surf,dT_dz,delta_z,num_levels):
    """
       build a hydrostatic atmosphere by integrating the hydrostatic equation from the surface,
       using num_layers=num_levels-1 of constant thickness delta_z
       input:  T_surf -- surface temperature in K
              p_surf -- surface pressure in Pa
              dT_dz -- constant rate of temperature change with height in K/m
              delta_z  -- layer thickness in m
              num_levels -- number of levels in the atmosphere
       output:
              numpy arrays: Temp (K) , press (Pa), rho (kg/m^3), height (m)
    """
    Rd=287. #J/kg/K  -- gas constant for dry air
    g=9.8  #m/s^2
    #
    # all these variables are defined on levels
    #
    Temp=np.empty([num_levels])
    press=np.empty_like(Temp)
    rho=np.empty_like(Temp)
    height=np.empty_like(Temp)
    #
    # layer 0 sits directly above the surface, so start
    # with pressure, temp of air equal to ground temp, press
    # and get density from equaiton of state
    # 
    press[0]=p_surf
    Temp[0]=T_surf
    rho[0]=p_surf/(Rd*T_surf)
    height[0]=0
    num_layers=num_levels-1
    #now march up the atmosphere a layer at a time
    for i in range(num_layers):
        delP= -rho[i]*g*delta_z
        height[i+1] = height[i] + delta_z
        Temp[i+1] = Temp[i] + dT_dz*delta_z
        press[i+1]= press[i] + delP
        rho[i+1]=press[i+1]/(Rd*Temp[i+1])
    return (Temp,press,rho,height)

def find_tau(r_gas,k_lambda,rho,height):
    """
       input: r_gas -- gas mixing ratio in kg/kg
              k_lambda -- mass absorption coefficient in kg/m^2
              rho -- vector of air densities in kg/m^3
              height -- corresponding layer heights in m
       output:  tau -- vetical optical depth from the surface, same shape as rho
    """
    tau=np.empty_like(rho)
    tau[0]=0
    num_levels=len(rho)
    num_layers=num_levels-1
    for index in range(num_layers):
        delta_z=height[index+1] - height[index]
        delta_tau=r_gas*rho[index]*k_lambda*delta_z
        tau[index+1]=tau[index] + delta_tau
    return tau     

def radiances(tau,Temp,height,T_surf):
    up_rad=np.empty_like(height)
    down_rad=np.empty_like(height)
    sfc_rad=sigma_pi*T_surf**4.
    up_rad[0]=sfc_rad
    tot_levs=len(tau)
    for index in np.arange(1,tot_levs):
        upper_lev=index
        lower_lev=index - 1
        del_tau=tau[upper_lev] - tau[lower_lev]
        trans=np.exp(-del_tau)
        emiss=1 - trans
        layer_rad=sigma_pi*Temp[lower_lev]**4.*emiss
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
        del_tau=tau[upper_lev] - tau[lower_lev]
        trans=np.exp(-del_tau)
        emiss=1 - trans
        layer_rad=sigma_pi*Temp[upper_lev]**4.*emiss
        down_rad[lower_lev]=down_rad[upper_lev]*trans + layer_rad
    return (up_rad,down_rad)

if __name__=="__main__":
    r_gas=0.01  #kg/kg
    k_lambda=0.01  #m^2/kg
    T_surf=300 #K
    p_surf=100.e3 #Pa
    dT_dz = -7.e-3 #K/km
    delta_z=10
    num_levels=1500    
    Temp,press,rho,height=hydrostat(T_surf,p_surf,dT_dz,delta_z,num_levels)
    tau=find_tau(r_gas,k_lambda,rho,height)
    up,down=radiances(tau,Temp,height,T_surf)

    fig1,axis1=plt.subplots(1,1)
    axis1.plot(up,height*0.001,'b-',lw=5,label='upward radiance')
    axis1.plot(down,height*0.001,'g-',lw=5,label='downward radiance')
    axis1.set_title('upward and downward radiances')
    axis1.set_xlabel('radiance $(W\,m^{-2}\,sr^{-1})$')
    axis1.set_ylabel('height (km)')
    axis1.legend(numpoints=1,loc='best')
    fig1.savefig('rad_updn.png')

    fig2,axis2=plt.subplots(1,1)
    axis2.plot(up-down,height*0.001,'b-',lw=5)
    axis2.set_title('net upward radiance')
    axis2.set_xlabel('net upward radiance $(W\,m^{-2}\,sr^{-1})$')
    axis2.set_ylabel('height (km)')
    fig1.savefig('rad_net.png')

    plt.show()

    
