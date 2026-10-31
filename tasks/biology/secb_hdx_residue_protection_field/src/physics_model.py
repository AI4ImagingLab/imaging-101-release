import torch
GAS_CONSTANT=8.31446261815324
TEMPERATURE=303.15
TV_WEIGHT=2e-4
LABEL_FRACTION=.9

def tensors(raw,corrected):
    return tuple(torch.as_tensor(x,dtype=torch.float64) for x in [raw['membership'],raw['intrinsic_rates'][:,None],raw['time_s'][None],corrected])

def forward(energy,membership,intrinsic_rates,time_s):
    protection=torch.exp(energy.reshape(-1,1)/(GAS_CONSTANT*TEMPERATURE))
    uptake=1-torch.exp(-(intrinsic_rates/(1+protection))@time_s)
    return LABEL_FRACTION*(membership@uptake)

def losses(energy,membership,intrinsic_rates,time_s,observed,weight=TV_WEIGHT):
    predicted=forward(energy,membership,intrinsic_rates,time_s);mse=torch.mean((predicted-observed)**2);tv=weight*torch.mean(torch.abs(energy[:-1]-energy[1:]));return mse,tv,predicted
