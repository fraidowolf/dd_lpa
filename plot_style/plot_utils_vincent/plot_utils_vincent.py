import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib._cm import cubehelix as chelix

def plot_zoom_inset(ax, xy1, xy2, loc=1, scale = (1.,1.), offset = (0.05,0.05),
                    shadow_offset = (0.02,0.02), color = 'grey', alpha=0.5, edges=[1,2,3,4]):
    """
    Add an inset to a plot with a zoom on selected data.
    The data has to be replotted, but the limits of the plot are already set.
    If used within a subplot, the offsets have to be adjusted manually to get the same spacing vertically and horizontally.
    
    Example
    -------
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.arange(100)
    y = np.sin(x*2*np.pi/20)*x
    plt.plot(x, y)
    axz = plot_zoom_inset(ax, (0,-10), (20,10), loc=2)
    axz.plot(x, y)
    axz2 = plot_zoom_inset(ax, (70,-20), (80,-80), loc=3)
    axz2.plot(x, y)
    
    Parameters
    ----------
    
    ax: matplotlib.axes
        Axis containing the plot
    
    xy1: tuple
        bottom left corner of the data to zoom on
        
    xy2: tuple
        top right corner of the data to zoom on
    
    loc: int, in {1,2,3,4}, optional
        defines the location of the inset:
        1: top right
        2: top left
        3: bottom left
        4: bottom right
    
    scale: tuple or float, optional
        Scale the size of the inset
    
    offset: tuple or float, optional
        offset of the inset from the edge of the parent axis
    
    shadow_offset: tuple or float, optional
        offset of the shadow below the inset. Set to 0 for no shadow
    
    color: str, optional
        Color of the edges and the shadow.
    
    alpha: float, optional
        alpha value of the box, edges, and shadow
    
    edges: list, optional
        Define which edge connecting the data box to the inset is drawn. If empty list, no edge is drawn
    """
    
    # Default axis size
    def_size = 0.25
    
    # check arguments
    if loc not in [1,2,3,4]:
        loc=1
    try:
        scalex = scale[0]
        scaley = scale[1]
    except TypeError:
        scalex = scale
        scaley = scale
    try:
        offsetx = offset[0]
        offsety = offset[1]
    except TypeError:
        offsetx = offset
        offsety = offset
    try:
        shadow_offsetx = shadow_offset[0]
        shadow_offsety = shadow_offset[1]
    except TypeError:
        shadow_offsetx = shadow_offset
        shadow_offsety = shadow_offset
    
    # retrieve figure ratio
    fig_ratio = ax.figure.bbox.bounds
    fig_ratio = fig_ratio[3]/fig_ratio[2]
    
    # Get data box position
    xa = min([xy1[0], xy2[0]])
    xb = max([xy1[0], xy2[0]])
    ya = min([xy1[1], xy2[1]])
    yb = max([xy1[1], xy2[1]])
    dx, dy = xb - xa, yb - ya
    data_ratio = dy/dx
    
    # Get the plot limits
    limx = ax.get_xlim()
    limy = ax.get_ylim()
    
    # Adjust the offsets if the aspect ratio is not auto
    if not ax.get_aspect() == 'auto':
        if ax.get_aspect() == 'equal':
            asp = 1
        else:
            asp = ax.get_aspect()
        if data_ratio >= scaley/scalex:
            x_red = (scalex/scaley - 1/data_ratio)*def_size*scaley/2
            y_red = 0
        else:
            y_red = (scaley/scalex - data_ratio)*def_size*scalex/2
            x_red = 0
        offsetx *= asp
    else:
        x_red = 0
        y_red = 0
        offsetx *= fig_ratio
        asp = fig_ratio    
    
    # Add black rectangle around selected data
    zmax = max([_.zorder for _ in ax.get_children()]) # Get max z order to place the rectangle on top
    r_data = mpl.patches.Rectangle((xa, ya), dx, dy, fill=False, ec=color, zorder=zmax, lw=0.5) # Build rectangle
    ax.add_patch(r_data) # Add it to the axis
    
    # Geet position of inset in data coordinates
    ll, lr, ul, ur = relative_data_position(def_size, loc, (scalex, scaley),
                                            (offsetx, offsety), limx, limy, ax.get_aspect(), data_ratio)
    
    # Add connecting lines
    if 3 in edges:
        ax.plot([xa, ll[0]], [ya, ll[1]], c = color, alpha = alpha, lw=0.5, zorder=zmax+1)
    if 4 in edges:
        ax.plot([xb, lr[0]], [ya, lr[1]], c = color, alpha = alpha, lw=0.5, zorder=zmax+1)
    if 2 in edges:
        ax.plot([xa, ul[0]], [yb, ul[1]], c = color, alpha = alpha, lw=0.5, zorder=zmax+1)
    if 1 in edges:
        ax.plot([xb, ur[0]], [yb, ur[1]], c = color, alpha = alpha, lw=0.5, zorder=zmax+1)
    # Reset limits
    ax.set_xlim(limx)
    ax.set_ylim(limy)
    
    # Add shadow below inset
    sx, sy = shadow_offsetx, shadow_offsety/asp
    if loc in [1,4]:
        sx = -sx
    if loc in [3,4]:
        sy = -sy
    of,_,_,_ = relative_data_position(def_size, loc, (scalex, scaley),
                                      (offsetx+sx, offsety+sy), limx, limy, ax.get_aspect(), data_ratio)
    # Build the shadow patch
    shadow = mpl.patches.Rectangle(of, lr[0]-ll[0], ul[1]-ll[1], color=color, ec=None, alpha=alpha, zorder=zmax)
    ax.add_patch(shadow)
    
    # Add axis inset
    nap = relative_inset_position(def_size, loc, (scalex, scaley), (offsetx-x_red, offsety-y_red)) # new axis position
    oap = ax.get_position() # old axis position
    axz = plt.axes([oap.x0 + nap[0]*(oap.x1-oap.x0), oap.y0 + nap[1]*(oap.y1-oap.y0), nap[2]*(oap.x1-oap.x0), nap[3]*(oap.y1-oap.y0)])

    # Set the limits
    axz.set_xlim(xa, xb)
    axz.set_ylim(ya, yb)
    # Remove ticks
    axz.set_xticks([])
    axz.set_yticks([])
    # change color
    for key in axz.spines.keys():
        axz.spines[key].set_color(color)
    return axz

def relative_inset_position(def_size, loc, scale, offset):
    "Computes the inset position from the parameters"
    if loc==4:
        position = [1-offset[0]-scale[0]*def_size,
                    offset[1], scale[0]*def_size, scale[1]*def_size]
    elif loc==2:
        position = [offset[0],
                    1-offset[1]-scale[1]*def_size, scale[0]*def_size, scale[1]*def_size]
    elif loc==3:
        position = [offset[0], offset[1], scale[0]*def_size,
                    scale[1]*def_size]
    else:
        position = [1-offset[0]-scale[0]*def_size, 1-offset[1]-scale[1]*def_size,
                    scale[0]*def_size, scale[1]*def_size]
    return position

def relative_data_position(def_size, loc, scale, offset, limx, limy, aspect, ratio):
    "Computes the inset position in the data coordinates"
    xmin = limx[0]
    ymin = limy[0]
    dx = limx[1]-xmin
    dy = limy[1]-ymin
    if not aspect == 'auto':
        if ratio >= scale[1]/scale[0]:
            scale = (scale[1]/ratio, scale[1])
        else:
            scale = (scale[0], scale[0]*ratio)
    pos = relative_inset_position(def_size, loc, scale, offset)
        
    ll = (xmin + dx*pos[0], ymin + dy*pos[1])
    ur = (xmin + dx*(pos[0]+pos[2]), ymin + dy*(pos[1]+pos[3]))
    lr = (ur[0], ll[1])
    ul = (ll[0], ur[1])
    return ll, lr, ul, ur

def remove_ticks():
    """Remove ticks on current axis"""
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    
def cmap_nicify(cmap, transparent=False, idx_white = 0, size_white = None):
    """
    Make the bottom of the colormap white
    """
    register = False
    if type(cmap) == str:
        cmap = mpl.cm.get_cmap(cmap)
        register = True
    if size_white is None:
        size_white = cmap.N//5
        
    index_white = np.arange(2*size_white-1) - size_white + idx_white + 1
    curve = np.sin(np.linspace(-np.pi/2, np.pi/2, 2*size_white-1))**2
    clip = (index_white>=0)*(index_white<=(cmap.N-1))
    my_cmap_rgba = cmap(np.arange(cmap.N))
    # Set alpha
    my_cmap_rgba[:,-1][index_white[clip][0]:index_white[clip][-1]+1] = curve[clip]
    my_cmap_rgb = my_cmap_rgba.copy()
    
    if not transparent:
        # Transform alpha to color
        my_cmap_rgb[:,0] = (1-my_cmap_rgba[:,-1])*1 + my_cmap_rgba[:,-1]*my_cmap_rgba[:,0]
        my_cmap_rgb[:,1] = (1-my_cmap_rgba[:,-1])*1 + my_cmap_rgba[:,-1]*my_cmap_rgba[:,1]
        my_cmap_rgb[:,2] = (1-my_cmap_rgba[:,-1])*1 + my_cmap_rgba[:,-1]*my_cmap_rgba[:,2]

        my_cmap_rgb[:,-1] *= 0
        my_cmap_rgb[:,-1] += 1
    if register and not idx_white:
        mpl.cm.register_cmap(name=cmap.name + '_w', cmap=ListedColormap(my_cmap_rgb))
    else:
        return ListedColormap(my_cmap_rgb)

def cmap_nicify_all():
    """
    Make the bottom of all colormaps white, and add them to the list of known maps with the flag '_w'
    """
    deprecated_maps = ['Vega10','Vega10_r','Vega20','Vega20_r',
    'Vega20b','Vega20b_r','Vega20c','Vega20c_r','spectral','spectral_r']
    for cmap in plt.colormaps():
        if cmap[-2:] != '_w':
            if cmap not in deprecated_maps:
                cmap_nicify(cmap, transparent=False)

def custom_cubehelix(gamma=1.0, start=0.0, rotation=-0.5, hue=1.0):
    """
    Custom cubehelix colormap, can be called using 'custom_ch'. The reversed version is also accessible using '_r'
    """
    mpl.cm.register_cmap(name='custom_ch', data=chelix(gamma=gamma, s=start, r=rotation, h=hue))
    mpl.cm.register_cmap(name='custom_ch_r', cmap=plt.get_cmap('custom_ch').reversed())

def truncate_cmap(cmap, minval=0., maxval=1., numcol=256):
    """
    Truncate a colormap and register it with '_t' at the end of the name
    """
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, numcol)))
    mpl.cm.register_cmap(name=cmap.name + '_t', cmap=new_cmap)