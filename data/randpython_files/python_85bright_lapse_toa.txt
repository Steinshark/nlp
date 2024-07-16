"""
   modify day22_radiance.py so that the lapse rate dT_dz is a vector
   Loop over the set of lapse rates and save the TOA radiances
   for our retrieval exercise
"""

import numpy as np
import matplotlib.pyplot as plt
from planck import planckwavelen,planckInvert
#
# OrderedDict keeps keys in order
#
from collections import OrderedDict
try:
    import seaborn
except:
    pass

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

def top_radiance(tau,Temp,height,T_surf,wavel,k_lambda):
    """Input:
           tau: vector of level optical depths
           Temp: vector of level temperatures (K)
           height: vector of level heights (m)
           T_surf: temperature of black surface (K)
           the_lambda: wavelength (m)
           k_lambda: mass absorption coefficient (m^2/kg)
       Output:
           top_rad: radiance at top of atmosphere (W/m^2/micron/sr)
    """       
    sfc_rad=planckwavelen(wavel,T_surf)
    up_rad=sfc_rad
    print "-"*60
    print "wavelength: %8.2f microns" % (wavel*1.e6)
    print "surface radiation: %8.2f W/m^2/micron/sr" % (up_rad*1.e-6)
    print "total tau: %8.2f" % tau[-1]
    print "-"*60
    tot_levs=len(tau)
    for index in np.arange(1,tot_levs):
        upper_lev=index
        lower_lev=index - 1
        del_tau=tau[upper_lev] - tau[lower_lev]
        trans=np.exp(-del_tau)
        emiss=1 - trans
        layer_rad=emiss*planckwavelen(wavel,Temp[lower_lev])
        #
        # find the radiance at the next level
        #
        up_rad=trans*up_rad + layer_rad
    return up_rad

if __name__=="__main__":
    r_gas=0.01  #kg/kg
    T_surf=300 #K
    p_surf=100.e3 #Pa
    dT_dz= np.arange(-9.e-3,-4.e-3,0.5e-3)
    delta_z=25000/7
    num_levels=7
    #
    # try to duplicate weighting functions for WH fig. 4.33
    #
    wavenums=np.linspace(666,766,7)
    wavelengths=(1/wavenums)*1.e4  #microns
    #
    # we want most absorbing channel at 15 microns, so reverse
    # the order of the wavelengths so that 15 microns is
    # at the end
    #
    wavelengths=wavelengths[::-1]
    rad_profs=[]
    bright_profs=[]
    for the_lapse_rate in dT_dz:
        print "looping: "
        Temp,press,rho,height=hydrostat(T_surf,p_surf,the_lapse_rate,delta_z,num_levels)
        #
        # I played around with the magnitude of k_lambda until the weighting functions
        # peaked at a range of heights
        #
        k_lambda=np.array([0.002,0.003,0.006,0.010,0.012,0.016,0.020])*5.  
        wavel_k_tup=zip(wavelengths,k_lambda)
        rad_dict=OrderedDict()
        bright_dict=OrderedDict()
        for wavel,k_lambda in wavel_k_tup:
            tau=find_tau(r_gas,k_lambda,rho,height)
            #convert wavel to meters
            rad_value=top_radiance(tau,Temp,height,T_surf,wavel*1.e-6,k_lambda)
            rad_dict[wavel]=rad_value
            bright_dict[wavel]=planckInvert(wavel*1.e-6,rad_value)
        rad_profs.append(rad_dict)
        bright_profs.append(bright_dict)

    plt.close('all')
    fig1,axis1=plt.subplots(1,1)
    for index,the_profile in enumerate(rad_profs):
        wavelengths=the_profile.keys()
        radiances=np.array(the_profile.values())
        radiances=radiances/radiances.mean()
        #
        # convert dT_dz to K/km
        #
        axis1.plot(wavelengths,radiances,label=str(dT_dz[index]*1.e3))
    axis1.set_title('normalized radiances at top of atmosphere for {} values of dT/dz (K/km)'.format(len(dT_dz)))
    axis1.set_ylabel('normalized radiances (no units)')
    axis1.set_xlabel('wavelength (microns)')
    axis1.legend(loc='best')
    fig1.savefig('normalized_radiances.png')

    fig2,axis2=plt.subplots(1,1)
    for index,the_profile in enumerate(bright_profs):
        wavelengths=the_profile.keys()
        brights=np.array(the_profile.values())
        #
        # convert dT_dz to K/km
        #
        axis2.plot(wavelengths,brights,label=str(dT_dz[index]*1.e3))
        
    axis2.set_title('brightness temperatures at top of atmosphere for {} values of dT/dz (K/km)'.format(len(dT_dz)))
    axis2.set_ylabel('Brightness temperature (K)')
    axis2.set_xlabel('wavelength (microns)')
    axis2.legend(loc='best')
    fig2.savefig('brightness.png')

    
    fig3,axis3=plt.subplots(1,1)
    lapse_rate=[]
    diff_list=[]
    for index,the_profile in enumerate(bright_profs):
        brights=np.array(the_profile.values())
        #
        # dT_dz in K/km
        #
        lapse_rate.append(dT_dz[index]*1.e3)
        diff_list.append(brights[0] - brights[-1])
        
    axis3.plot(lapse_rate,diff_list)
        
    axis3.set_title('13 $\mu m$ - 15 $\mu m$ brightness temperature difference (K)')
    axis3.set_ylabel('Brightness temperature differnce (K)')
    axis3.set_xlabel('lapse rate (K/km)')
    fig3.savefig('temp_diff.png')

    fig4,axis4=plt.subplots(1,1)
    lapse_rate=[]
    diff_list=[]
    for index,the_profile in enumerate(bright_profs):
        brights=np.array(the_profile.values())
        #
        # dT_dz in K/km
        #
        lapse_rate.append(dT_dz[index]*1.e3)
        diff_list.append(2.*(brights[0] - brights[-1])/(brights[0] + brights[-1]))
        
    axis4.plot(lapse_rate,diff_list)
        
    axis4.set_title('$(T_{13} - T_{15})/ ( \overline{T_{bright}}  )$')
    axis4.set_ylabel('normalized brightness temperature differnce (no units)')
    axis4.set_xlabel('lapse rate (K/km)')
    fig4.savefig('normalized_temp_diff.png')

        
    plt.show()

    
