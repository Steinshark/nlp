import tkinter as tk
import json 


if __name__ == "__main__":

    #APP DEFAULT SETTINGS 
    app_settings        = { 
                            'window_name'               :"ML Viewer 0.0",
                            'window_w'                  :1920,
                            'window_h'                  :1080,

                            'control_frame_bg'          :"#97c1d1",
                            'top_frame_bg'              :"#677075",
                            'view_frame_bg'             :"#c5e2e6",

                            'control_item_bg'           :"#062936"

                           }

    #TRAINING DEFAULT SETTINGS
    train_settings      = {}





    #####################################################################################################################
    #                                                    CREATE APP FRAMEWORK                                           #
    #####################################################################################################################
    window                              = tk.Tk()
    window.title                        = window.title(app_settings['window_name'])
    window                              .geometry(f'{app_settings["window_w"]}x{app_settings["window_h"]}')

    window                              .columnconfigure(0,weight=1)
    window                              .columnconfigure(1,weight=4)

    window                              .rowconfigure(0,weight=5)
    window                              .rowconfigure(1,weight=11)
    
    window.grid()






    #####################################################################################################################
    #                                                    CREATE APP LAYOUT                                              #
    #####################################################################################################################
    control_frame                       = tk.Frame(window)
    top_frame                           = tk.Frame(window)
    view_frame                          = tk.Frame(window)

    #Configure Control Frame
    control_frame                       .configure(background=app_settings['control_frame_bg'])
    control_frame                       .columnconfigure(0)

    #Configure Top Frame
    top_frame                           .configure(background=app_settings['top_frame_bg'])
    top_frame                           .columnconfigure(0,weight=20)
    top_frame                           .columnconfigure(1,weight=20)
    top_frame                           .columnconfigure(2,weight=1)
    top_frame                           .columnconfigure(3,weight=20)
    top_frame                           .columnconfigure(4,weight=20)

    #Configure Control Frame
    control_frame                       .configure(background=app_settings['control_frame_bg'])
    control_frame                       .columnconfigure(0)

    #Place 
    control_frame                       .grid(row=0,column=0,rowspan=2,sticky=tk.NSEW)
    top_frame                           .grid(row=0,column=1,sticky=tk.NSEW)
    view_frame                          .grid(row=1,column=1,sticky=tk.NSEW)





    #####################################################################################################################
    #                                                    LAYOUT CONTROL FRAME                                           #
    #####################################################################################################################
    control_frames_count                = 5

    session_name_frame                  = tk.Frame(control_frame,height=5,width=100)
    model_optim_frame                   = tk.Frame(control_frame,height=5,width=100)
    activation_fn_frame                 = tk.Frame(control_frame,height=5,width=100)
    optim_kwarg_frame                   = tk.Frame(control_frame,height=5,width=100)
    train_settings_frame                = tk.Frame(control_frame,height=5,width=100)                                                       # 5


    control_frame_items                 = [session_name_frame,model_optim_frame,activation_fn_frame,optim_kwarg_frame,train_settings_frame] 


    #Rowconfig for control frame
    for frame_num in range(control_frames_count):
        control_frame                   .rowconfigure(frame_num,weight=1) 

    #Configure all control frames 
    for i,frame in enumerate(control_frame_items):
        
        #Color 
        frame                           .configure(background=app_settings['control_item_bg']) 
        #Row
        frame.rowconfigure(0,weight=1)
        #Column
        for col in range(4):
            frame                       .columnconfigure(col,weight=1)
        #Grid
        frame                           .grid(row=i,column=1,sticky=tk.EW)



    #####################################################################################################################
    #                                                  LAYOUT TOP FRAME                                                 #
    #####################################################################################################################

    top_frame                           .rowconfigure(0,weight=1)
    top_frame                           .rowconfigure(1,weight=5)
    top_frame                           .rowconfigure(2,weight=1)

    #Create divider 
    top_frame_divider_frame             = tk.Frame(top_frame)
    top_frame_divider_frame             .configure(background='#000000')
    top_frame_divider_frame             .grid(row=0,column=2,rowspan=3)



    window.mainloop()







