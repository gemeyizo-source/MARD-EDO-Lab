"""PINN para PVI escalares precargados; no implementa todavía MARD-EDO."""
import time
import numpy as np
import pandas as pd
import torch
from torch import nn

class RedPINN(nn.Module):
    def __init__(self, ancho=32, capas=3):
        super().__init__()
        layers=[]; d=1
        for _ in range(capas):
            layers.extend([nn.Linear(d,ancho),nn.Tanh()]);d=ancho
        layers.append(nn.Linear(d,1))
        self.net=nn.Sequential(*layers)
    def forward(self, s):
        return self.net(s)

def entrenar_pinn(tipo='Decaimiento exponencial',t0=0.,tf=3.,y0=1.,kappa=50.,epocas=1500,puntos=100,tasa=0.001,semilla=42,ancho=32):
    """Entrena con residuo L2; la condición inicial se cumple exactamente.

    Usa tiempo normalizado s=(t-t0)/(tf-t0), y u(t)=y0+s*N(s).
    """
    if tipo not in ('Decaimiento exponencial','Crecimiento logístico','Forzado rígido'):
        raise ValueError('La PINN actual solo admite tres PVI escalares precargados.')
    if tf<=t0 or puntos<3 or epocas<1:raise ValueError('Intervalo, puntos o épocas inválidos')
    torch.manual_seed(int(semilla));torch.set_num_threads(1)
    modelo=RedPINN(ancho=ancho)
    opt=torch.optim.Adam(modelo.parameters(),lr=tasa)
    dur=float(tf-t0)
    s=torch.linspace(0,1,puntos,dtype=torch.float32).reshape(-1,1).requires_grad_(True)
    history=[];inicio=time.perf_counter()
    for ep in range(1,epocas+1):
        opt.zero_grad()
        t=t0+dur*s
        u=y0+s*modelo(s)
        du_ds=torch.autograd.grad(u,s,grad_outputs=torch.ones_like(u),create_graph=True)[0]
        du_dt=du_ds/dur
        if tipo=='Decaimiento exponencial':rhs=-2*u
        elif tipo=='Crecimiento logístico':rhs=1.2*u*(1-u/10)
        else:rhs=-kappa*(u-torch.cos(t))-torch.sin(t)
        residual=du_dt-rhs
        loss=(residual**2).mean()
        if not torch.isfinite(loss):raise ValueError('Pérdida no finita; modifica los parámetros.')
        loss.backward();opt.step()
        if ep==1 or ep%10==0 or ep==epocas:
            history.append({'epoca':ep,'perdida':float(loss.detach())})
    duration=time.perf_counter()-inicio
    s_test=torch.linspace(0,1,301,dtype=torch.float32).reshape(-1,1).requires_grad_(True)
    t_test=t0+dur*s_test
    u_test=y0+s_test*modelo(s_test)
    du_ds=torch.autograd.grad(u_test,s_test,grad_outputs=torch.ones_like(u_test),create_graph=False)[0]
    with torch.no_grad():
        tt=t_test.detach().flatten().numpy().astype(float)
        yy=u_test.detach().flatten().numpy().astype(float)
        derivative=(du_ds/dur).detach().flatten().numpy().astype(float)
    if tipo=='Decaimiento exponencial':
        exact=y0*np.exp(-2*(tt-t0));rhs_np=-2*yy
    elif tipo=='Crecimiento logístico':
        exact=10/(1+(10/y0-1)*np.exp(-1.2*(tt-t0))) if y0>0 else None
        rhs_np=1.2*yy*(1-yy/10)
    else:
        exact=np.cos(tt)+(y0-np.cos(t0))*np.exp(-kappa*(tt-t0))
        rhs_np=-kappa*(yy-np.cos(tt))-np.sin(tt)
    residual=derivative-rhs_np
    data={'t':tt,'PINN':yy,'Residuo PINN':residual,'Derivada PINN':derivative}
    metrics={'Tiempo PINN (s)':duration,'RMS residuo PINN':float(np.sqrt(np.mean(residual**2))), 'Semilla':int(semilla)}
    if exact is not None and np.all(np.isfinite(exact)):
        err=yy-exact
        data.update({'Exacta':exact,'Error absoluto PINN':np.abs(err)})
        metrics.update({'MAE':float(np.mean(np.abs(err))),'RMSE':float(np.sqrt(np.mean(err**2))),'Error máximo':float(np.max(np.abs(err))),'L2 relativo':float(np.linalg.norm(err)/max(np.linalg.norm(exact),1e-12))})
    return pd.DataFrame(data),pd.DataFrame(history),metrics
