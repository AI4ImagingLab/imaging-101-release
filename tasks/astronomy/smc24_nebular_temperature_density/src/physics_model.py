"""Explicit multi-level collisional-radiative statistical equilibrium.
Equations adapted from Luridiana, Morisset and Shaw, PyNeb1.1.32.
GPL-3.0-or-later; source attribution and license in task LICENSE.
No PyNeb or other domain library imports.
"""
import numpy as np

KCOLLRATE=8.629132419846745e-6
BOLTZMANN_ANGK=6.95034760642295e-9
HPLANCK=6.62606957e-27
CLIGHT=29979245800.

class AtomicIon:
    def __init__(self, energy,weight,A,temperature,collision):
        self.energy=np.asarray(energy);self.weight=np.asarray(weight)
        self.A=np.asarray(A);self.temperature=np.asarray(temperature)
        self.collision=np.asarray(collision);self.n=len(energy)

    def omega(self,temperature):
        temperature=np.asarray(temperature)
        result=np.zeros((self.n,self.n)+temperature.shape)
        for lower in range(self.n-1):
            for upper in range(lower+1,self.n):
                result[upper,lower]=np.interp(np.log10(temperature),self.temperature,self.collision[lower,upper])
        return result

    def rates(self,temperature):
        temperature=np.asarray(temperature);om=self.omega(temperature)
        result=np.zeros_like(om)
        for lower in range(self.n-1):
            for upper in range(lower+1,self.n):
                down=KCOLLRATE/np.sqrt(temperature)/self.weight[upper]*om[upper,lower]
                result[upper,lower]=down
                result[lower,upper]=(self.weight[upper]/self.weight[lower])*np.exp((self.energy[lower]-self.energy[upper])/(BOLTZMANN_ANGK*temperature))*down
        return result

    def populations(self,temperature,density):
        ts=np.atleast_1d(temperature);ns=np.atleast_1d(density)
        rates=self.rates(ts);answer=np.zeros((self.n,len(ts),len(ns)))
        for i,t in enumerate(ts):
            for j,n in enumerate(ns):
                transition=n*rates[:,:,i]+self.A
                matrix=transition.T.copy()
                np.fill_diagonal(matrix,-transition.sum(axis=1))
                matrix[0]=1.
                rhs=np.zeros(self.n);rhs[0]=1.
                answer[:,i,j]=np.linalg.solve(matrix,rhs)
        return np.squeeze(answer)

    def emissivities(self,temperature,density):
        ts=np.atleast_1d(temperature);ns=np.atleast_1d(density)
        pop=self.populations(ts,ns).reshape(self.n,len(ts),len(ns))
        result=np.zeros((self.n,self.n,len(ts),len(ns)))
        for upper in range(1,self.n):
            for lower in range(upper):
                # Energy tables are inverse Angstrom; h*c uses centimetres.
                result[upper,lower]=pop[upper]*self.A[upper,lower]*HPLANCK*CLIGHT*1e8*(self.energy[upper]-self.energy[lower])/ns
        return np.squeeze(result)

def load_ions(path='data/raw_data.npz'):
    with np.load(path,allow_pickle=False) as a:
        return {key:AtomicIon(a[key+'_energy'][0],a[key+'_weight'][0],a[key+'_A'][0],a[key+'_collision_temperature'][0],a[key+'_collision_strength'][0]) for key in ['N2','O2','O3','S2','S3','Ne3']}
