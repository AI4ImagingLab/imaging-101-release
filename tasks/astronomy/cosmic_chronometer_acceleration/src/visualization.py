"""Review figure export without selecting a plotting backend."""
from pathlib import Path
def save_review_figure(figure,path):
    """Write a fully computed Matplotlib figure at150dpi; return its Path."""
    destination=Path(path);destination.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(destination,dpi=150)
    return destination
