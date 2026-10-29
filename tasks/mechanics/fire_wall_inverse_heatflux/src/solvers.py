"""Explicit closed-form inverse heat balance; no artificial iteration."""
from .physics_model import energy_terms,EMISSIVITY

def reconstruct(data,**parameters):
    """Return incident-radiative field (25,64,48) kW/m² and actual energy terms.

    All25 measured frames participate. The measured spatial collar supplies the
    neighbors for each interior target pixel. No historical estimate is read.
    """
    terms=energy_terms(data['temperature'],**parameters)
    eps=parameters.get('emissivity',EMISSIVITY)
    q=(terms['storage']+terms['convection']+terms['radiation']-terms['conduction'])/(1000*eps)
    return q,terms
