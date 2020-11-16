def latexify(fig_width=None, fig_height=None):
    # setup Matplotlib to create paper ready figures
    # call this function before plotting figure

   
    # import packages
    import matplotlib    
    
    # figure sizing        
    if fig_width is None:
        fig_width = 9.0
        
    if fig_height is None:
#        fig_height = fig_width*0.75         # height in inches
        fig_height = 6.0
        
    # define all the necessary figure properties   
    params = {'backend': 'ps',
              'text.latex.preamble': ['\usepackage{gensymb}'],
              'text.usetex': False,         # set to True for Latex text including Latex math
              'text.fontsize': 17,
              'text.antialiased': True,     # render text in antialised 
              
              'figure.figsize': [fig_width,fig_height],
              'figure.facecolor': 'white',
              'figure.frameon': True,
              
              'font.family': 'serif',
              'font.serif' : 'Times New Roman',
              'font.variant': 'normal',
              'font.weight': 600,
              
              'axes.labelsize': 17,         # fontsize for x and y labels (was 10)
              'axes.titlesize': 18.5,
              'axes.labelpad': 10.0,         # space between label and axis
              'axes.hold': True,             # whether to clear the axes by default on
              'axes.facecolor': 'white',     # axes background color
              'axes.edgecolor': 'black',     # axes edge color
              'axes.linewidth': 1.0,         # edge linewidth
              'axes.grid': False,            # display grid or not
              'axes.labelweight': 600,       # weight of the x and y labels
              'axes.labelcolor': 'black',
              'axes.axisbelow': False,       # whether axis gridlines and ticks are below
                                             # the axes elements (lines, text, etc)
#              
              'xtick.labelsize': 17,         # size of numbers or letters on x axis 
              'ytick.labelsize': 17,         # size of numbers or letters on x axis 
              'xtick.major.pad': 6.0,        # distance to major tick label in points
              'ytick.major.pad': 4.0,        # distance to major tick label in points
              'xtick.major.size': 5.0,       # major tick size in points
              'ytick.major.size': 5.0,       # major tick size in points
              'xtick.major.width' : 0.75,    # major tick width in points
              'ytick.major.width' : 0.75,    # major tick width in points
              
              'lines.linewidth': 3.0,        # line width in points
              'lines.linestyle': '-',        # solid line
              'lines.markeredgewidth': 0.25, # the line width around the marker symbol
              'lines.markersize': 15.0,      # markersize, in points
              'lines.antialiased': True,     # render lines in antialised 
              
              'errorbar.capsize' : 7.0,      # length of end cap on error bars in pixels
              
              'legend.fontsize': 16.0,       # was 10                         
              'legend.fancybox': False,      # if True, use a rounded box for the legend, else a rectangle                                                       
              'legend.shadow': False,        # shadow for legend frame
              'legend.frameon':False,        # whether or not to draw a frame around legend
              'legend.numpoints': 1,         # the number of points in the legend line
              'legend.borderpad': 0.10,      # border whitespace in fontsize units
              'legend.columnspacing' :0.5  , # the border between the axes and legend edge in fraction of fontsize

              'savefig.bbox': 'standard',        # 'tight' or 'standard'.                                
#              'savefig.pad_inches': 0.2      # Padding to be used when bbox is set to 'tight'
             
    }

    # update matplotlib figure properties
    matplotlib.rcParams.update(params)
    