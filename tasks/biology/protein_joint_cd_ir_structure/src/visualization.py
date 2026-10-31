import numpy as np
import matplotlib.pyplot as plt
NAMES=['Myoglobin','Lysozyme','Ubiquitin','Concanavalin A']
STRUCTURES=['Helix','Sheet','Turns','Other']
def rms_error(a,b):return float(np.sqrt(np.mean((a-b)**2)))
def fractions_plot(groups,title):
    fig,axes=plt.subplots(1,4,figsize=(17,4),layout='constrained')
    width=.8/len(groups)
    for i,ax in enumerate(axes):
        for j,(name,array) in enumerate(groups.items()):ax.bar(np.arange(4)-.4+width*(j+.5),array[i],width,label=name)
        ax.set_xticks(range(4),STRUCTURES,rotation=30);ax.set(title=NAMES[i],ylabel='Fraction');ax.axhline(0,color='black',lw=.5)
    axes[-1].legend(fontsize=7);fig.suptitle(title);return fig
