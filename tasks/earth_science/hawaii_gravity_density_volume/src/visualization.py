"""Scientific map helpers; imports never alter a backend."""
def station_map(axis,locations,values,title,limits=None):
    image=axis.scatter(locations[:,0]/1000,locations[:,1]/1000,c=values,s=10,cmap='RdBu_r')
    if limits is not None:image.set_clim(*limits)
    axis.set(xlabel='East (km)',ylabel='North (km)',title=title);axis.set_aspect('equal')
    return image
def density_slice(axis,volume,index,title,limits):
    image=axis.imshow(volume[index],origin='lower',extent=[-24,24,-24,24],cmap='RdBu_r',vmin=limits[0],vmax=limits[1])
    axis.set(xlabel='East (km)',ylabel='North (km)',title=title)
    return image
