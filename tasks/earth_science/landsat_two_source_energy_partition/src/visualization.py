"""Spatial reviews; no plotting backend is selected on import."""
from pathlib import Path
import numpy as np
def image_from_pixels(values,pixel_index,shape=(336,335)):
    image=np.full(np.prod(shape),np.nan);image[np.asarray(pixel_index,dtype=int)]=values
    return image.reshape(shape)
def save_review_figure(figure,path):
    destination=Path(path);destination.parent.mkdir(parents=True,exist_ok=True);figure.savefig(destination,dpi=150)
    return destination
