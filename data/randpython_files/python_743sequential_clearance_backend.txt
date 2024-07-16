#!/usr/bin/python
from __future__ import division
import sys
import time
import math
import numpy as np
import scipy as scp
import cmath as cm

########################################################################

class CB_Relay(object):
    """This class attempts to define all the parameters surrounding an inverse time overcurrent relay"""

    #----------------------------------------------------------------------
    def __init__(self,TimeMultiplier=0.25,ctratio=400.0,pickup=1.0,curvetype='NI',percent_travel=0.0,\
                 cb_open_time=0.06, highsetpickup=10.0,highset=False):
        """Constructor"""
        self.TimeMultiplier=TimeMultiplier
        self.pickup=pickup
        self.ctratio=ctratio
        self.curvetype=curvetype
        self.percent_travel=percent_travel
        self.tripped=False
        self.pulse_counter=0 #how many pulses of fault current
        self.cb_open_time=cb_open_time  #time in milliseconds
        self.highset=highset
        self.highsetpickup=highsetpickup           
        self.current=[]
        self.current_fdr1_open=[]
        self.current_fdr2_open=[]
        self.time3=[]
        self.time_fdr1_open=[]
        self.time_fdr2_open=[]
        
        self.time_to_operate()


    def time_to_operate(self,I=2000):
        
        curvetypes=dict(NI=(0.00001399354,0.00010458841,0.99999866576,0.0,0.0),\
                        SI=(0.00001399354,0.00010458841,0.99999866576,0.0,0.0),\
                        VI=(1.0,13.5,1.0,0.0,0.0),\
                        EI=(2.0,80.0,1.0,0.0,0.0),\
                        )
        
        self.I=I
        
        a,b,c,d,e=curvetypes.get(unicode(self.curvetype))
        self.a,self.b,self.c,self.d,self.e=a,b,c,d,e

        self.multiple=(self.I/(self.pickup*self.ctratio))            

        if (self.highset==True and self.I>(self.highsetpickup*self.ctratio)):
            self.time=0.02
        elif self.I>(self.pickup*self.ctratio):
            self.time = self.TimeMultiplier * (d + (b / (math.pow(self.multiple,a) - c))) + e
        else:
            self.time=50
            
        return self.time
    
    def curve_plot(self):
        start = 1.2
        stop = 20.0
        samples = 100
        x=[]
        y=[]
        #self.time_to_operate(
        for i in np.linspace(start,stop,samples):
            
            q=(self.TimeMultiplier * (self.d + (self.b / (math.pow(i,self.a) - self.c))) + self.e)
            x.append((i*self.pickup*self.ctratio))
            y.append(q)
            #print i
        return x,y
    
    def setpercent(self,percent):
        self.percent_travel=min(percent,1)
        
    def marginleft(self,current):
        x=self.time_to_operate(current)
        y=(1-self.percent_travel)*x
        return y
    
    def time_to_open(self,I):
        x=self.time_to_operate(I)
        y=self.self.cb_open_time+x
        return y
    
    def percent_travel(self):
        return self.percent_travel
        
    
    
        

def main_seq(ratio2,mult0,mult1,mult2,pickup0,pickup1,pickup2,incct,feederct1,feederct2,\
             tximp,inc_cbtime,inc_highsetpickup,inc_checkBox,\
             fdr1_cbtime,fdr1_highsetpickup,fdr1_checkBox,\
             fdr2_cbtime,fdr2_highsetpickup,fdr2_checkBox,
             inc_curve,fdr1_curve,fdr2_curve,lineangle):

    resolution=0.01
    Tx_percentage_impedance=tximp
    Tx_impedance_angle=85
    Tx_impedance=complex(cm.rect(Tx_percentage_impedance,math.radians(Tx_impedance_angle)))
    V_base=11000
    MVA_base=20000000 #20MVA transformer
    Z_base=math.pow(V_base,2)/MVA_base

    Z_transformer=Z_base*Tx_impedance/100.0

    Z_loop_impedance=abs(Z_transformer)*ratio2

    Z_loop_impedance_angle=lineangle
    Z_loop=complex(cm.rect(Z_loop_impedance,math.radians(Z_loop_impedance_angle)))


    '''
    Initial conditions
    '''
    i=0
    m=0.5

    ratio=abs(Z_loop/Z_transformer)


    '''
    Declare lists and initialize
    '''
    m_store=[]
    del m_store[:]


    margin_store=[]
    margin_store2=[]
    margin_store3=[]
    margin_store4=[]
    margin_store5=[]
    margin_store6=[]

    incomer=CB_Relay(mult0,incct,pickup0,inc_curve,0,inc_cbtime,inc_highsetpickup,inc_checkBox)  #Instantiation of Incomer object
    feederone=CB_Relay(mult1,feederct1,pickup1,fdr1_curve,0,fdr1_cbtime,fdr1_highsetpickup,fdr1_checkBox)  #Instantiation of feederone object
    feedertwo=CB_Relay(mult2,feederct2,pickup2,fdr2_curve,0,fdr2_cbtime,fdr2_highsetpickup,fdr2_checkBox)  #Instantiation of feedertwo object 
    
    q1=incomer.curve_plot()
    q2=feederone.curve_plot()
    q3=feedertwo.curve_plot()
    
    #print q1

    '''Fault calculation functions'''

    def I_total_fault_current_loop(m):
        Z_parallel=1/((1/(Z_loop*m))+(1/((1-m)*Z_loop)))
        I=abs(V_base/math.sqrt(3)/(Z_transformer+Z_parallel))
        return(I)

    #I think we need to develop this to amend what is return based on whether feeder one or feeder two is open.

    def I_total_fault_current_radial(m):
        if feederone.tripped and feedertwo.tripped:
            Z_radial=Z_transformer        
        elif feederone.tripped:
            Z_radial=Z_loop*(1-m)
        elif feedertwo.tripped:
            Z_radial=Z_loop*m
        else:
            pass
        I=abs(V_base/math.sqrt(3)/(Z_transformer+Z_radial))
        return(I)

    #Calculate the currents for the incomer and both feeders 

    for m in np.arange(0.01,1.00,resolution):

        m_store.append(m)
    #First loop -stage 1
        incomer.percent_travel=(0)
        feederone.percent_travel=(0)
        feedertwo.percent_travel=(0)

    #Parallel    
        incomer.current.append(I_total_fault_current_loop(m)) # storage of current values for graphing.
        feederone.current.append(I_total_fault_current_loop(m)*(1-m)) # m is the relative distance from feeder one
        feedertwo.current.append(I_total_fault_current_loop(m)*(m)) # m is te the relative distance from feeder one


        incomer.time3.append(incomer.time_to_operate(I_total_fault_current_loop(m)))
        feederone.time3.append(feederone.time_to_operate(I_total_fault_current_loop(m)*(1-m)))     
        feedertwo.time3.append(feedertwo.time_to_operate(I_total_fault_current_loop(m)*(m))) 

        #Feeder One open
        feederone.tripped=True
        feedertwo.tripped=False

        incomer.current_fdr1_open.append(I_total_fault_current_radial(m))
        feedertwo.current_fdr1_open.append(I_total_fault_current_radial(m)) # m is te the relative distance from feeder one

        incomer.time_fdr1_open.append(incomer.time_to_operate(I_total_fault_current_radial(m))) 
        feedertwo.time_fdr1_open.append(feedertwo.time_to_operate(I_total_fault_current_radial(m))) 

        #Feeder Two Open
        feederone.tripped=False
        feedertwo.tripped=True

        incomer.current_fdr2_open.append(I_total_fault_current_radial(m))
        feederone.current_fdr2_open.append(I_total_fault_current_radial(m)) # m is the relative distance from feeder one

        incomer.time_fdr2_open.append(incomer.time_to_operate(I_total_fault_current_radial(m))) 
        feederone.time_fdr2_open.append(feederone.time_to_operate(I_total_fault_current_radial(m)))

    #    return(incomer_current,incomer_time,m_store)

    #Sequential clearance calc

    for i in xrange(len(m_store)):

        feederone.tripped=0
        feedertwo.tripped=0
        incomer.tripped=0
        feederone.setpercent(0)
        feedertwo.setpercent(0)
        incomer.setpercent(0)


        if (incomer.time3[i])<min(feederone.time3[i],feedertwo.time3[i]):
            case=1
        elif (feederone.time3[i]+feederone.cb_open_time)<feedertwo.time3[i]:
            case=2 #classic sequential clearance 
        elif (feedertwo.time3[i]+feedertwo.cb_open_time)<feederone.time3[i]:
            case=3 #classic sequential clearance 
        elif abs(feederone.time3[i]-feedertwo.time3[i])==0:
            case=4 #simple margin
        elif feederone.time3[i]<feedertwo.time3[i]:
            case=5 #feeder 1 starts to trip first but then its race to who open first
        elif feedertwo.time3[i]<feederone.time3[i]:
            case=6 # feeder 2 starts to trip first
            
        #print case
            
        if incomer.tripped==1:
            margin_store.append(0)
            margin_store2.append(100)
            margin_store3.append(1)
            margin_store4.append(0)
            margin_store5.append(0)
            margin_store6.append(0)
            
        elif (case==2): #classic sequential clearance 
            incomer.setpercent((feederone.time3[i]+feederone.cb_open_time)/incomer.time3[i])
            feedertwo.setpercent((feederone.time3[i]+feederone.cb_open_time)/feedertwo.time3[i])
            
            stage1=feederone.time3[i]+feederone.cb_open_time
            stage2=feedertwo.time_fdr1_open[i]*(1-feedertwo.percent_travel)+feedertwo.cb_open_time
            
            w=incomer.time3[i]*((incomer.percent_travel))
            x=incomer.time_fdr1_open[i]*(1-incomer.percent_travel)+w
            y=feedertwo.time_fdr1_open[i]*(1-feedertwo.percent_travel)
            
            y=stage1+stage2
            
            margin_store.append((x-y+0.06))
            margin_store2.append(incomer.percent_travel*100)
            margin_store3.append(2)
            
            margin_store4.append(incomer.time3[i]-feederone.time3[i])
            margin_store5.append(stage1)
            margin_store6.append((stage1+stage2))      
            #margin_store6.append((stage2))  
            
        elif (case==3): #classic sequential clearance 
            incomer.setpercent((feedertwo.time3[i]+feedertwo.cb_open_time)/incomer.time3[i])
            feederone.setpercent((feedertwo.time3[i]+feedertwo.cb_open_time)/feederone.time3[i])
            

            
            stage1=feedertwo.time3[i]+feedertwo.cb_open_time
            stage2=feederone.time_fdr2_open[i]*(1-feederone.percent_travel)+feederone.cb_open_time
            
            w=incomer.time3[i]*((incomer.percent_travel))
            x=incomer.time_fdr2_open[i]*(1-incomer.percent_travel)+w
            y=feederone.time_fdr2_open[i]*(1-feederone.percent_travel)            
            y=stage1+stage2
            
            margin_store.append((x-y+0.06))            
            margin_store2.append(incomer.percent_travel*100)                
            margin_store3.append(3)
            
            margin_store4.append(incomer.time3[i]-feedertwo.time3[i])
            margin_store5.append(stage1)
            margin_store6.append((stage1+stage2))      
            #margin_store6.append((stage2))  
            
        elif (case==4): # exactly the same time 
            incomer.setpercent((feedertwo.time3[i]+feedertwo.cb_open_time)/incomer.time3[i])
            feederone.setpercent(1)

            stage1= min(feederone.time3[i]+feederone.cb_open_time,feedertwo.time3[i]+feedertwo.cb_open_time)
            stage2=abs((feederone.time3[i]+feederone.cb_open_time)-(feedertwo.time3[i]+feedertwo.cb_open_time))
            
            x=incomer.time3[i]
            y=max(feederone.time_fdr2_open[i],feedertwo.time_fdr1_open[i])
            
            w=incomer.time3[i]*((incomer.percent_travel))
            x=incomer.time_fdr1_open[i]*(1-incomer.percent_travel)+w
            y=stage1+stage2 # total time to feeder tripping.            
            
            margin_store.append((x-y+0.06))   
            margin_store2.append(incomer.percent_travel*100)
            margin_store3.append(4)
            
            margin_store4.append(incomer.time3[i]-feedertwo.time3[i])
            margin_store5.append(stage1)
            margin_store6.append((stage1+stage2))      
            #margin_store6.append((stage2))  
            
        elif (case==5): #feeder 1 opens first
            #Stage 1 feeder one asserts to open first but if breaker open time is slow, feeder two may open first.
            incomer.setpercent((feederone.time3[i]+feederone.cb_open_time)/incomer.time3[i])
            feedertwo.setpercent((feederone.time3[i]+feederone.cb_open_time)/feedertwo.time3[i])
            
            #Stage 2 incomer continues to time while feeder 2 now opens
            
            stage1= min(feederone.time3[i]+feederone.cb_open_time,feedertwo.time3[i]+feedertwo.cb_open_time)
            stage2 = abs((feederone.time3[i]+feederone.cb_open_time)-(feedertwo.time3[i]+feedertwo.cb_open_time))
            
            
            w=incomer.time3[i]*((incomer.percent_travel))
            x=incomer.time_fdr1_open[i]*(1-incomer.percent_travel)+w
            y=stage1+stage2
            
            margin_store.append((x-y+0.06))
            margin_store2.append(incomer.percent_travel*100)            
            margin_store3.append(5)
            margin_store4.append(incomer.time3[i]-feederone.time3[i])
            margin_store5.append(stage1)
            margin_store6.append((stage1+stage2))      
            #margin_store6.append((stage2))  
            
        elif (case==6): #feeder 2 opens first
            #feeder two asserts to open first but if breaker open time is slow, feeder one may open first.
            incomer.setpercent((feedertwo.time3[i]+feedertwo.cb_open_time)/incomer.time3[i])
            feederone.setpercent((feedertwo.time3[i]+feedertwo.cb_open_time)/feederone.time3[i])
            
            #Stage 2 incomer continues to time while feeder 1 now opens
            stage1= min(feederone.time3[i]+feederone.cb_open_time,feedertwo.time3[i]+feedertwo.cb_open_time)
            stage2 = abs((feederone.time3[i]+feederone.cb_open_time)-(feedertwo.time3[i]+feedertwo.cb_open_time))

            w=incomer.time3[i]*((incomer.percent_travel))
            x=incomer.time_fdr2_open[i]*(1-incomer.percent_travel)+w
            y=stage1+stage2
            
            margin_store.append((x-y+0.06))            
            margin_store2.append(incomer.percent_travel*100)                
            margin_store3.append(6)
            margin_store4.append(incomer.time3[i]-feedertwo.time3[i])
            margin_store5.append(stage1)
            margin_store6.append((stage1+stage2))      
            #margin_store6.append((stage2))  
            
            
        else:
            print feederone.tripped,feedertwo.tripped,incomer.tripped
            margin_store.append(20)
            margin_store2.append(100)
            margin_store3.append(20)
            margin_store4.append(20)
            margin_store5.append(20)
            margin_store6.append(0)

    return (m_store,margin_store,margin_store2,margin_store3,margin_store4,margin_store5,margin_store6,ratio2,\
            incomer.time3,incomer.time_fdr1_open,incomer.time_fdr2_open,\
            feederone.time3,feederone.time_fdr1_open,feederone.time_fdr2_open,\
            feedertwo.time3,feedertwo.time_fdr1_open,feedertwo.time_fdr2_open,\
            incomer.current,incomer.current_fdr1_open,incomer.current_fdr2_open,
            feederone.current,feederone.current_fdr1_open,feederone.current_fdr2_open,
            feedertwo.current,feedertwo.current_fdr1_open,feedertwo.current_fdr2_open,q1,q2,q3\
            )

class sequentialClearance(object):
    def init(self,ratio2,mult0,mult1,mult2,pickup0,pickup1,pickup2,incct,feederct1,feederct2,\
                 tximp,inc_cbtime,inc_highsetpickup,inc_checkBox,\
                 fdr1_cbtime,fdr1_highsetpickup,fdr1_checkBox,\
                 fdr2_cbtime,fdr2_highsetpickup,fdr2_checkBox,
                 inc_curve,fdr1_curve,fdr2_curve,lineangle): # can I just say keyword arguments.
    
        resolution=0.01
        Tx_percentage_impedance=tximp
        Tx_impedance_angle=85
        Tx_impedance=complex(cm.rect(Tx_percentage_impedance,math.radians(Tx_impedance_angle)))
        V_base=11000
        MVA_base=20000000 #20MVA transformer
        Z_base=math.pow(V_base,2)/MVA_base
    
        Z_transformer=Z_base*Tx_impedance/100.0
    
        Z_loop_impedance=abs(Z_transformer)*ratio2
    
        Z_loop_impedance_angle=lineangle
        Z_loop=complex(cm.rect(Z_loop_impedance,math.radians(Z_loop_impedance_angle)))
    
    
        '''
        Initial conditions
        '''
        i=0
        m=0.5
    
        ratio=abs(Z_loop/Z_transformer)
    
    
        '''
        Declare lists and initialize
        '''
        m_store=[]
        del m_store[:]
    
    
        margin_store=[]
        margin_store2=[]
        margin_store3=[]
        margin_store4=[]
        margin_store5=[]
        margin_store6=[]
    
        incomer=CB_Relay(mult0,incct,pickup0,inc_curve,0,inc_cbtime,inc_highsetpickup,inc_checkBox)  #Instantiation of Incomer object
        feederone=CB_Relay(mult1,feederct1,pickup1,fdr1_curve,0,fdr1_cbtime,fdr1_highsetpickup,fdr1_checkBox)  #Instantiation of feederone object
        feedertwo=CB_Relay(mult2,feederct2,pickup2,fdr2_curve,0,fdr2_cbtime,fdr2_highsetpickup,fdr2_checkBox)  #Instantiation of feedertwo object 
        
        q1=incomer.curve_plot()
        q2=feederone.curve_plot()
        q3=feedertwo.curve_plot()
        
        #print q1
    
        '''Fault calculation functions'''
    
        def I_total_fault_current_loop(m):
            Z_parallel=1/((1/(Z_loop*m))+(1/((1-m)*Z_loop)))
            I=abs(V_base/math.sqrt(3)/(Z_transformer+Z_parallel))
            return(I)
    
        #I think we need to develop this to amend what is return based on whether feeder one or feeder two is open.
    
        def I_total_fault_current_radial(m):
            if feederone.tripped and feedertwo.tripped:
                Z_radial=Z_transformer        
            elif feederone.tripped:
                Z_radial=Z_loop*(1-m)
            elif feedertwo.tripped:
                Z_radial=Z_loop*m
            else:
                pass
            I=abs(V_base/math.sqrt(3)/(Z_transformer+Z_radial))
            return(I)
    
        #Calculate the currents for the incomer and both feeders 
    
        for m in np.arange(0.01,1.00,resolution):
    
            m_store.append(m)
        #First loop -stage 1
            incomer.percent_travel=(0)
            feederone.percent_travel=(0)
            feedertwo.percent_travel=(0)
    
        #Parallel    
            incomer.current.append(I_total_fault_current_loop(m)) # storage of current values for graphing.
            feederone.current.append(I_total_fault_current_loop(m)*(1-m)) # m is the relative distance from feeder one
            feedertwo.current.append(I_total_fault_current_loop(m)*(m)) # m is te the relative distance from feeder one
    
    
            incomer.time3.append(incomer.time_to_operate(I_total_fault_current_loop(m)))
            feederone.time3.append(feederone.time_to_operate(I_total_fault_current_loop(m)*(1-m)))     
            feedertwo.time3.append(feedertwo.time_to_operate(I_total_fault_current_loop(m)*(m))) 
    
            #Feeder One open
            feederone.tripped=True
            feedertwo.tripped=False
    
            incomer.current_fdr1_open.append(I_total_fault_current_radial(m))
            feedertwo.current_fdr1_open.append(I_total_fault_current_radial(m)) # m is te the relative distance from feeder one
    
            incomer.time_fdr1_open.append(incomer.time_to_operate(I_total_fault_current_radial(m))) 
            feedertwo.time_fdr1_open.append(feedertwo.time_to_operate(I_total_fault_current_radial(m))) 
    
            #Feeder Two Open
            feederone.tripped=False
            feedertwo.tripped=True
    
            incomer.current_fdr2_open.append(I_total_fault_current_radial(m))
            feederone.current_fdr2_open.append(I_total_fault_current_radial(m)) # m is the relative distance from feeder one
    
            incomer.time_fdr2_open.append(incomer.time_to_operate(I_total_fault_current_radial(m))) 
            feederone.time_fdr2_open.append(feederone.time_to_operate(I_total_fault_current_radial(m)))
    
        #    return(incomer_current,incomer_time,m_store)
    
        #Sequential clearance calc
    
        for i in xrange(len(m_store)):
    
            feederone.tripped=0
            feedertwo.tripped=0
            incomer.tripped=0
            feederone.setpercent(0)
            feedertwo.setpercent(0)
            incomer.setpercent(0)
    
    
            if (incomer.time3[i])<min(feederone.time3[i],feedertwo.time3[i]):
                case=1
            elif (feederone.time3[i]+feederone.cb_open_time)<feedertwo.time3[i]:
                case=2 #classic sequential clearance 
            elif (feedertwo.time3[i]+feedertwo.cb_open_time)<feederone.time3[i]:
                case=3 #classic sequential clearance 
            elif abs(feederone.time3[i]-feedertwo.time3[i])==0:
                case=4 #simple margin
            elif feederone.time3[i]<feedertwo.time3[i]:
                case=5 #feeder 1 starts to trip first but then its race to who open first
            elif feedertwo.time3[i]<feederone.time3[i]:
                case=6 # feeder 2 starts to trip first
                
            #print case
                
            if incomer.tripped==1:
                margin_store.append(0)
                margin_store2.append(100)
                margin_store3.append(1)
                margin_store4.append(0)
                margin_store5.append(0)
                margin_store6.append(0)
                
            elif (case==2): #classic sequential clearance 
                incomer.setpercent((feederone.time3[i]+feederone.cb_open_time)/incomer.time3[i])
                feedertwo.setpercent((feederone.time3[i]+feederone.cb_open_time)/feedertwo.time3[i])
                
                stage1=feederone.time3[i]+feederone.cb_open_time
                stage2=feedertwo.time_fdr1_open[i]*(1-feedertwo.percent_travel)+feedertwo.cb_open_time
                
                w=incomer.time3[i]*((incomer.percent_travel))
                x=incomer.time_fdr1_open[i]*(1-incomer.percent_travel)+w
                y=feedertwo.time_fdr1_open[i]*(1-feedertwo.percent_travel)
                
                y=stage1+stage2
                
                margin_store.append((x-y+0.06))
                margin_store2.append(incomer.percent_travel*100)
                margin_store3.append(2)
                
                margin_store4.append(incomer.time3[i]-feederone.time3[i])
                margin_store5.append(stage1)
                margin_store6.append((stage1+stage2))      
                #margin_store6.append((stage2))  
                
            elif (case==3): #classic sequential clearance 
                incomer.setpercent((feedertwo.time3[i]+feedertwo.cb_open_time)/incomer.time3[i])
                feederone.setpercent((feedertwo.time3[i]+feedertwo.cb_open_time)/feederone.time3[i])
                
    
                
                stage1=feedertwo.time3[i]+feedertwo.cb_open_time
                stage2=feederone.time_fdr2_open[i]*(1-feederone.percent_travel)+feederone.cb_open_time
                
                w=incomer.time3[i]*((incomer.percent_travel))
                x=incomer.time_fdr2_open[i]*(1-incomer.percent_travel)+w
                y=feederone.time_fdr2_open[i]*(1-feederone.percent_travel)            
                y=stage1+stage2
                
                margin_store.append((x-y+0.06))            
                margin_store2.append(incomer.percent_travel*100)                
                margin_store3.append(3)
                
                margin_store4.append(incomer.time3[i]-feedertwo.time3[i])
                margin_store5.append(stage1)
                margin_store6.append((stage1+stage2))      
                #margin_store6.append((stage2))  
                
            elif (case==4): # exactly the same time 
                incomer.setpercent((feedertwo.time3[i]+feedertwo.cb_open_time)/incomer.time3[i])
                feederone.setpercent(1)
    
                stage1= min(feederone.time3[i]+feederone.cb_open_time,feedertwo.time3[i]+feedertwo.cb_open_time)
                stage2=abs((feederone.time3[i]+feederone.cb_open_time)-(feedertwo.time3[i]+feedertwo.cb_open_time))
                
                x=incomer.time3[i]
                y=max(feederone.time_fdr2_open[i],feedertwo.time_fdr1_open[i])
                
                w=incomer.time3[i]*((incomer.percent_travel))
                x=incomer.time_fdr1_open[i]*(1-incomer.percent_travel)+w
                y=stage1+stage2 # total time to feeder tripping.            
                
                margin_store.append((x-y+0.06))   
                margin_store2.append(incomer.percent_travel*100)
                margin_store3.append(4)
                
                margin_store4.append(incomer.time3[i]-feedertwo.time3[i])
                margin_store5.append(stage1)
                margin_store6.append((stage1+stage2))      
                #margin_store6.append((stage2))  
                
            elif (case==5): #feeder 1 opens first
                #Stage 1 feeder one asserts to open first but if breaker open time is slow, feeder two may open first.
                incomer.setpercent((feederone.time3[i]+feederone.cb_open_time)/incomer.time3[i])
                feedertwo.setpercent((feederone.time3[i]+feederone.cb_open_time)/feedertwo.time3[i])
                
                #Stage 2 incomer continues to time while feeder 2 now opens
                
                stage1= min(feederone.time3[i]+feederone.cb_open_time,feedertwo.time3[i]+feedertwo.cb_open_time)
                stage2 = abs((feederone.time3[i]+feederone.cb_open_time)-(feedertwo.time3[i]+feedertwo.cb_open_time))
                
                
                w=incomer.time3[i]*((incomer.percent_travel))
                x=incomer.time_fdr1_open[i]*(1-incomer.percent_travel)+w
                y=stage1+stage2
                
                margin_store.append((x-y+0.06))
                margin_store2.append(incomer.percent_travel*100)            
                margin_store3.append(5)
                margin_store4.append(incomer.time3[i]-feederone.time3[i])
                margin_store5.append(stage1)
                margin_store6.append((stage1+stage2))      
                #margin_store6.append((stage2))  
                
            elif (case==6): #feeder 2 opens first
                #feeder two asserts to open first but if breaker open time is slow, feeder one may open first.
                incomer.setpercent((feedertwo.time3[i]+feedertwo.cb_open_time)/incomer.time3[i])
                feederone.setpercent((feedertwo.time3[i]+feedertwo.cb_open_time)/feederone.time3[i])
                
                #Stage 2 incomer continues to time while feeder 1 now opens
                stage1= min(feederone.time3[i]+feederone.cb_open_time,feedertwo.time3[i]+feedertwo.cb_open_time)
                stage2 = abs((feederone.time3[i]+feederone.cb_open_time)-(feedertwo.time3[i]+feedertwo.cb_open_time))
    
                w=incomer.time3[i]*((incomer.percent_travel))
                x=incomer.time_fdr2_open[i]*(1-incomer.percent_travel)+w
                y=stage1+stage2
                
                margin_store.append((x-y+0.06))            
                margin_store2.append(incomer.percent_travel*100)                
                margin_store3.append(6)
                margin_store4.append(incomer.time3[i]-feedertwo.time3[i])
                margin_store5.append(stage1)
                margin_store6.append((stage1+stage2))      
                #margin_store6.append((stage2))  
                
                
            else:
                print feederone.tripped,feedertwo.tripped,incomer.tripped
                margin_store.append(20)
                margin_store2.append(100)
                margin_store3.append(20)
                margin_store4.append(20)
                margin_store5.append(20)
                margin_store6.append(0)
    
        return (m_store,margin_store,margin_store2,margin_store3,margin_store4,margin_store5,margin_store6,ratio2,\
                incomer.time3,incomer.time_fdr1_open,incomer.time_fdr2_open,\
                feederone.time3,feederone.time_fdr1_open,feederone.time_fdr2_open,\
                feedertwo.time3,feedertwo.time_fdr1_open,feedertwo.time_fdr2_open,\
                incomer.current,incomer.current_fdr1_open,incomer.current_fdr2_open,
                feederone.current,feederone.current_fdr1_open,feederone.current_fdr2_open,
                feedertwo.current,feedertwo.current_fdr1_open,feedertwo.current_fdr2_open,q1,q2,q3\
                )
    
    def get_incomer_margin(self):
        #return self.margin
        pass
    
    def get_incomer_percentage(self):
        #return self.percent1st, self.percent2nd
        pass
    
    def get_feederone_opentime(self):
        pass
    
    def get_feedertwo_opentime(self):
        pass
    
    def set_resolution(self,resolution):
        #self.resolution=resolution
        pass
        

    
    
        
    


