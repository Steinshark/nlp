import numpy as np

#A10  beamwidth angle


def stull_8_13(the_wavel,dish_size):
    """
       input:  the_wavel : wavelength (cm)
               dish_size : antenna dish diameter (m)
       output: beamwidth : beamwidth angle  (degrees)
              
    """
    a = 71.6 #constant (degrees)
    the_wavel = the_wavel/100. #convert cm to m
    beamwidth = a*the_wavel/dish_size #beamwitdh in degrees
    return beamwidth


letters=['a','b','c','d','e','f','g','h','i','j'] #assignment letter
the_wavel=[20,20,10,10,10,5,5,5,5,3]  #wavelength (cm)
dish_size=[8,10,10,5,3,7,5,2,3,1]   #dishsize (meters)

input_vals=zip(the_wavel,dish_size)

beamwidth=[stull_8_13(wavel,dish_size) for wavel,dish_size in input_vals]

output_pairs=zip(letters,beamwidth)

print "\nQuestion A10\n" 
for letter,beamwidth in output_pairs:
    print "%s) %5.3f degrees" % (letter,beamwidth)

print "\nQuestion A11\n"
prob_tups=zip(letters,the_wavel)
band_vals=[('l_band',15,30),('s_band',7.5,15),('c_band',3.75,7.5),('x_band',2.5,3.75),('ku_band',1.67,2.5),('ka_band',0.75,1.11)]
for letter,wavel in prob_tups:
    for band_name,left_lim,right_lim in band_vals:
        if wavel >= left_lim and wavel < right_lim:
            print "%s) wavelength of %3d cm is %s" % (letter,int(wavel),band_name)
            break

## #A12 Find the range to a radar target, given the round-trip (return) travel times (us) t.
times=[2,5,10,25,50,75,100,150,200,300]

def stull_8_16(delT):
    """
       input:  delT : the round-trip travel times (us)
       output: radar range (km)
    """
    c = 3e8 #speed of light (m/s)
    delT = delT*(1.e-6) #convert  to s
    radar_range = c*delT/2
    return radar_range*1.e-3  #kilometers

radar_range=[stull_8_16(delT) for delT in times]

output_pairs=zip(letters,radar_range)

print "\nQuestion A12\n"

for letter,range in output_pairs:
    print "%s) %3.1f km" % (letter,range)

#A13 Find the radar max unambiguous range for pulse repetition frequencies (s^-1) PRF.

PRFs=[50,100,200,400,600,800,1000,1200,1400,1600]

def stull_8_17(PRF):
    """
       input:  PRF : pulse repetition frequencies (s^-1)
       output: Rmax : the rad max unambiguous range (km)
              
    """
    c = 3.e8 #speed of light (m/s)
    Rmax = c/(2*PRF)
    return Rmax*1.e-3  #range in km

radar_range=[stull_8_17(PRF) for PRF in PRFs]
output_vals=zip(letters,PRFs,radar_range)
     
print "\nQuestion A13\n"

for letter,PRF,range in output_vals:
    print "%s) PRF=%d, range=%3.1f km" % (letter,PRF,range)
     

#A14 Find the Doppler max unambiguous velocity for a radar with pulse repetition frequency (s^-1) 
#    as given in the previous exercise, for radar with wavelength of: (i) 10 cm (ii) 5 cm

def stull_8_35(the_lambda,PRF):
    """
       input:  the_lambda:  wavelength (cm)
               PRF : pulse repetition frequencies (s^-1)
       output: Mmax : the Doppler max unambiguous velocity (m/s)
              
    """
    the_lambda = the_lambda/100. #convert cm to m
    Mmax = the_lambda*PRF/4.
    return Mmax

vel_10cm=[stull_8_35(10,PRF) for PRF in PRFs]

vel_5cm=[stull_8_35(5,PRF) for PRF in PRFs]

output_quad=zip(letters,PRFs,vel_10cm,vel_5cm)

print "\nQuestion A14\n" 
for letter,PRF,vel10,vel5 in output_quad:
    print "%s) PRF=%d Hz, 10 cm velocity=%5.2f m/s, 5 cm velocity=%5.2f m/s" % (letter,PRF,vel10,vel5)


def stull_r7(delT):
    """
       input: delT: pulse duration (us)
       output: the_vol: the size of the radar sample volume (km^3)
    """
    delT=delT*1.e-6 #convert to seconds
    range = 30.e3 #convert km to m
    the_wavel = 10 #cm
    dish_size = 5 #m
    beam_width=stull_8_13(the_wavel,dish_size) #degrees       
    vol_radius = range*(beam_width*np.pi/180.)/2.
    volume=np.pi*vol_radius**2.*delT*3.e8 #m^3
    return volume*1.e-9  #km^3

times=[0.1,0.2,0.5,1.0,1.5,2,3,5]
letters=letters[:7]
volumes=[stull_r7(delT) for delT in times]
output_vals=zip(letters,times,volumes)

print "\nQuestion A15\n" 
for letter,the_time,volume in output_vals:
    print "%s) for pulse duration of %4.1f microseconds the sample volume is %5.2f km^3" % (letter,the_time,volume)



        
    
    







