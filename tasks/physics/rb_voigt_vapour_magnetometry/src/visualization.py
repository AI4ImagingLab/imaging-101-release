"""Optional figure export helper; does not select a plotting backend."""
from pathlib import Path
def save_review_figure(figure,path):
    destination=Path(path);destination.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(destination,dpi=150)
    return destination
