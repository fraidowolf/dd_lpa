import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os


scale = 1
colors = ['#ff4747ff','#58c0cbfc','#7acc8aff','#3492daff']


# PRE figsize
mm_to_inch = 0.039370079
figsize = { 'mm': {'column_width': 86*scale, #(PRE), #89, #(nature) # 86, #(PRE)
                   'double_column_width': 183*scale, #(PRE) # 183, # (nature) # 178, #(PRE)
                   'gutter_width': 5*scale,
                   'max_height': 247*scale,
                   'units': 'mm',
                   'golden_ratio': 0.618},
            'inch': {'column_width': mm_to_inch*86*scale, #89, # (nature) #86, #(PRE)
                     'double_column_width': mm_to_inch*183*scale, # 183, # (nature) #178, #(PRE)
                     'gutter_width': mm_to_inch*5*scale,
                     'max_height': mm_to_inch*247*scale,
                     'units': 'inch',
                     'golden_ratio': 0.618} }



def load_preset(scale = 12/8,font_path='.'):
    
    font_manager._get_fontconfig_fonts.cache_clear()
    
    current_dir_path = '../'
    
    
    font_dirs = [font_path, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    SMALL_SIZE = 28
    MEDIUM_SIZE = 28
    BIGGER_SIZE = 32
    MS = 10
    
        
    plt.rcParams['font.family'] = 'Helvetica Neue'
    plt.rcParams['font.style'] = 'normal'
    plt.rcParams['font.size'] = 8*scale
    plt.rcParams['font.weight'] = 'normal' 
    plt.rcParams['axes.unicode_minus'] = False
   
    #plt.rcParams['text.latex.preamble'] = [
    #       r'\usepackage{siunitx}',
    #       r'\sisetup{detect-all}',
    #       #r'\usepackage{helvet}',
    #       r'\usepackage{sansmath}',
    #       r'\sansmath'
    #]

    plt.rcParams['font.size'] = 8*scale
    plt.rcParams['axes.linewidth'] = 0.75*scale
    plt.rcParams['axes.labelweight'] = 'normal'#'light'
    plt.rcParams['axes.titleweight'] = 'normal'#'light'
    plt.rcParams['axes.labelsize'] = 8*scale
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = [figsize['inch']['column_width'], figsize['inch']['column_width']*figsize['inch']['golden_ratio']]
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['figure.constrained_layout.h_pad'] = 0.01*scale
    plt.rcParams['figure.constrained_layout.w_pad'] = 0.01*scale
    plt.rcParams['image.aspect'] = 'auto'
    plt.rcParams['image.interpolation'] = 'None'
    plt.rcParams['image.origin'] = 'lower'
    plt.rcParams['lines.dash_capstyle'] = 'round'
    plt.rcParams['lines.solid_capstyle'] = 'round'
    plt.rcParams['lines.dash_joinstyle'] = 'bevel'
    plt.rcParams['lines.linewidth'] = 1.0*scale
    plt.rcParams['hatch.linewidth'] = 1.0*scale

    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Helvetica Neue:normal' #'Helvetica Neue'
    plt.rcParams['mathtext.sf'] = 'Helvetica Neue:normal' #'Helvetica Neue'
    plt.rcParams['mathtext.it'] = 'Helvetica Neue:normal:italic' #'Helvetica Neue:italic'
    plt.rcParams['mathtext.cal'] = 'Helvetica Neue:regular:italic' #'Helvetica Neue:regular:italic'
    plt.rcParams['mathtext.bf'] =  'Helvetica Neue:regular' #'Helvetica Neue:regular'

    plt.rcParams['xtick.major.width'] = 0.75*scale
    plt.rcParams['xtick.major.pad'] = 2.5*scale
    plt.rcParams['xtick.major.size'] =  3.5*scale
    plt.rcParams['xtick.labelsize'] =  8*scale

    plt.rcParams['ytick.major.width'] = 0.75*scale
    plt.rcParams['ytick.major.pad'] = 2.5*scale
    plt.rcParams['ytick.major.size'] =  3.5*scale
    plt.rcParams['ytick.labelsize'] =  8*scale

    plt.rcParams['savefig.transparent'] = True
    print('matplotlib preset loaded')
    
    
def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

# Custom colormap
cmap = matplotlib.cm.get_cmap('YlGnBu_r')
my_cmap_rgba = cmap(np.arange(cmap.N))

# Set alpha
my_cmap_rgba[:,-1][:cmap.N//5] = np.sin(np.linspace(0, np.pi/2, cmap.N//5))
my_cmap_rgb = my_cmap_rgba.copy()

my_cmap_rgb[:,0] = (1-my_cmap_rgba[:,-1])*1 + my_cmap_rgba[:,-1]*my_cmap_rgba[:,0]
my_cmap_rgb[:,1] = (1-my_cmap_rgba[:,-1])*1 + my_cmap_rgba[:,-1]*my_cmap_rgba[:,1]
my_cmap_rgb[:,2] = (1-my_cmap_rgba[:,-1])*1 + my_cmap_rgba[:,-1]*my_cmap_rgba[:,2]

my_cmap_rgb[:,-1] *= 0
my_cmap_rgb[:,-1] += 1

# Create new colormap
cmap = matplotlib.colors.ListedColormap(my_cmap_rgb)
cmap_trans = matplotlib.colors.ListedColormap(my_cmap_rgba)


# Custom colormap
cmap2 = matplotlib.cm.get_cmap('YlGnBu_r')
my_cmap_rgba2 = cmap2(np.arange(cmap2.N))

# Set alpha
my_cmap_rgba2[:,-1][:cmap2.N//4] = np.sin(np.linspace(0, np.pi/2, cmap2.N//4))
my_cmap_rgb2 = my_cmap_rgba2.copy()

my_cmap_rgb2[:,0] = (1-my_cmap_rgba2[:,-1])*1 + my_cmap_rgba2[:,-1]*my_cmap_rgba2[:,0]
my_cmap_rgb2[:,1] = (1-my_cmap_rgba2[:,-1])*1 + my_cmap_rgba2[:,-1]*my_cmap_rgba2[:,1]
my_cmap_rgb2[:,2] = (1-my_cmap_rgba2[:,-1])*1 + my_cmap_rgba2[:,-1]*my_cmap_rgba2[:,2]

my_cmap_rgb2[:,-1] *= 0
my_cmap_rgb2[:,-1] += 1

# Create new colormap
cmap2 = matplotlib.colors.ListedColormap(my_cmap_rgb2)
cmap2_trans = matplotlib.colors.ListedColormap(my_cmap_rgba2)

# 29.11.2020 --------------
# Custom colormap
cmap3 = matplotlib.cm.get_cmap('YlGnBu_r')
my_cmap_rgba3 = cmap3(np.arange(cmap3.N))

# Set alpha
my_cmap_rgba3[:,-1][:int(cmap3.N//3.5)] = np.sin(np.linspace(0, np.pi/2, int(cmap3.N//3.5)))
my_cmap_rgb3 = my_cmap_rgba3.copy()

my_cmap_rgb3[:,0] = (1-my_cmap_rgba3[:,-1])*1 + my_cmap_rgba3[:,-1]*my_cmap_rgba3[:,0]
my_cmap_rgb3[:,1] = (1-my_cmap_rgba3[:,-1])*1 + my_cmap_rgba3[:,-1]*my_cmap_rgba3[:,1]
my_cmap_rgb3[:,2] = (1-my_cmap_rgba3[:,-1])*1 + my_cmap_rgba3[:,-1]*my_cmap_rgba3[:,2]

my_cmap_rgb3[:,-1] *= 0
my_cmap_rgb3[:,-1] += 1

# Create new colormap
cmap3 = matplotlib.colors.ListedColormap(my_cmap_rgb3)
cmap3_trans = matplotlib.colors.ListedColormap(my_cmap_rgba3)


