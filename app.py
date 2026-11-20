"""MARD-EDO Lab - Solucionador de EDO. V1: integradores clásicos."""
from __future__ import annotations

import io
import math
import time
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from scipy.integrate import solve_ivp

st.set_page_config(page_title="MARD-EDO Lab", page_icon="📈", layout="wide")
st.markdown("""
<style>
.stApp {background: linear-gradient(145deg, #080e1e, #142038 68%, #0b1428);}
h1,h2,h3 {color:#edf7ff}
[data-testid="stMetric"] {background-color: #162a45; border: 1px solid #315778; padding:13px;border-radius:11px;}
</style>
""", unsafe_allow_html=True)

T, Y, Z = sp.symbols("t y z", real=True)
ALLOWED_FUNCS = {"sin":sp.sin,"cos":sp.cos,"tan":sp.tan,"exp":sp.exp,
                 "log":sp.log,"sqrt":sp.sqrt,"abs":sp.Abs,"pi":sp.pi,
                 "sinh":sp.sinh,"cosh":sp.cosh}
ALLOWED_SYMBOLS = {"t": T, "y": Y}


def safe_scalar_rhs(expression: str) -> Callable:
    """Parse symbolic expression with explicit symbols/functions and no eval."""
    if len(expression)>250 or not expression.strip():
        raise ValueError("La expresión debe tener entre 1 y 250 caracteres.")
    # Strict character whitelist prevents attributes, containers and arbitrary code.
    import re
    if not re.fullmatch(r"[0-9a-zA-Z_+*/()., \t\-^]+", expression):
        raise ValueError("La expresión contiene caracteres no permitidos.")
    # Require each named token to be an approved function, symbol, or constant.
    tokens=set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", expression))
    if not tokens.issubset(set(ALLOWED_FUNCS)|set(ALLOWED_SYMBOLS)):
        raise ValueError("Símbolos no permitidos: "+", ".join(sorted(tokens-set(ALLOWED_FUNCS)-set(ALLOWED_SYMBOLS))))
    # sympy parse_expr evaluates Python grammar: avoid parsing untrusted code in server.
    # Reject all non-mathematical grammar by accepting only a restricted mathematical AST.
    import ast
    normalized=expression.replace("^","**")
    node=ast.parse(normalized,mode="eval")
    allowed_bin=(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)
    def validate(n):
        if isinstance(n,ast.Expression): return validate(n.body)
        if isinstance(n,ast.Constant):
            if not isinstance(n.value,(int,float)) or isinstance(n.value,bool): raise ValueError("Número inválido")
            return
        if isinstance(n,ast.Name):
            if n.id not in ALLOWED_SYMBOLS and n.id!="pi": raise ValueError("Símbolo inválido")
            return
        if isinstance(n,ast.BinOp) and isinstance(n.op,allowed_bin):
            validate(n.left);validate(n.right);return
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.USub,ast.UAdd)):
            validate(n.operand);return
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ALLOWED_FUNCS and n.func.id!="pi" and not n.keywords and len(n.args)==1:
            validate(n.args[0]);return
        raise ValueError("Solo se aceptan operaciones matemáticas y funciones permitidas")
    validate(node)
    # Build SymPy recursively; never evaluate Python code.
    def build(n):
        if isinstance(n,ast.Expression): return build(n.body)
        if isinstance(n,ast.Constant): return sp.sympify(n.value)
        if isinstance(n,ast.Name): return ALLOWED_SYMBOLS.get(n.id,sp.pi)
        if isinstance(n,ast.UnaryOp):return -build(n.operand) if isinstance(n.op,ast.USub) else build(n.operand)
        if isinstance(n,ast.BinOp):
            a,b=build(n.left),build(n.right)
            if isinstance(n.op,ast.Add):return a+b
            if isinstance(n.op,ast.Sub):return a-b
            if isinstance(n.op,ast.Mult):return a*b
            if isinstance(n.op,ast.Div):return a/b
            if isinstance(n.op,ast.Pow):
                if b.is_number and (abs(float(b))>20):raise ValueError("Exponente demasiado grande")
                return a**b
        if isinstance(n,ast.Call):return ALLOWED_FUNCS[n.func.id](build(n.args[0]))
        raise ValueError("Expresión no admitida")
    symbolic=build(node)
    fun=sp.lambdify((T,Y),symbolic,modules=["numpy"])
    def rhs(t,y):
        value=np.asarray(fun(t,float(y[0])),dtype=float)
        if value.size!=1 or not np.all(np.isfinite(value)):
            raise ValueError("La función produjo un valor no real o no finito")
        return np.array([float(value.item())])
    return rhs


def model_catalog(choice,kappa, damping,alpha,beta,delta,gamma):
    if choice=="Decaimiento exponencial":
        return (lambda t,s:np.array([-2*s[0]]), ["y"], [1.0],(0.,3.),
                lambda t,ic,tstart:np.exp(-2*(t-tstart))*ic[0],"y' = -2y")
    if choice=="Crecimiento logístico":
        return (lambda t,s:np.array([1.2*s[0]*(1-s[0]/10)]),["y"],[0.5],(0.,8.),
                lambda t,ic,tstart:10/(1+(10/ic[0]-1)*np.exp(-1.2*(t-tstart))) if ic[0]>0 else np.zeros_like(t),
                "y' = 1.2y(1-y/10)")
    if choice=="Forzado rígido":
        return (lambda t,s:np.array([-kappa*(s[0]-np.cos(t))-np.sin(t)]),["y"],[1.],(0.,10.),
                lambda t,ic,tstart:np.cos(t)+(ic[0]-np.cos(tstart))*np.exp(-kappa*(t-tstart)),
                "y' = -κ(y-cos(t))-sin(t)")
    if choice=="Oscilador amortiguado (orden 2)":
        def osc(t,s):return np.array([s[1],-2*damping*s[1]-s[0]])
        return (osc,["y","v = y'"],[1.,0.],(0.,15.),None,"y'' + 2ζy' + y = 0")
    if choice=="Lotka-Volterra (sistema)":
        def lotka(t,s):return np.array([alpha*s[0]-beta*s[0]*s[1],delta*s[0]*s[1]-gamma*s[1]])
        return (lotka,["presa","depredador"],[10.,5.],(0.,20.),None,
                "x'=αx-βxy; z'=δxy-γz")
    raise ValueError("Modelo no reconocido")


def fixed_integrate(fun,tgrid,y0,method):
    states=np.empty((len(tgrid),len(y0)),dtype=float)
    states[0]=y0
    calls=0
    first=None
    for idx in range(len(tgrid)-1):
        t=float(tgrid[idx]);h=float(tgrid[idx+1]-t)
        y=states[idx]
        k1=np.asarray(fun(t,y),dtype=float);calls+=1
        if method=="Euler":
            next_y=y+h*k1
        elif method=="Heun":
            k2=np.asarray(fun(t+h,y+h*k1),dtype=float);calls+=1
            next_y=y+h*(k1+k2)/2
            if first is None:first={"k1":k1,"k2":k2,"resultado":next_y}
        else:
            k2=np.asarray(fun(t+h/2,y+h*k1/2),dtype=float)
            k3=np.asarray(fun(t+h/2,y+h*k2/2),dtype=float)
            k4=np.asarray(fun(t+h,y+h*k3),dtype=float);calls+=3
            next_y=y+h*(k1+2*k2+2*k3+k4)/6
            if first is None:first={"k1":k1,"k2":k2,"k3":k3,"k4":k4,"resultado":next_y}
        if not np.all(np.isfinite(next_y)) or np.max(np.abs(next_y))>1e100:
            raise ValueError(f"{method} diverge o desborda en t={t:.5g}; disminuye el paso o usa BDF/Radau.")
        states[idx+1]=next_y
    return states,calls,first


def run_case(fun,init,t0,tend,n_steps,methods,rtol,atol):
    grid=np.linspace(t0,tend,n_steps+1)
    results={};rows=[];stages={};fail=[]
    for method in methods:
        start=time.perf_counter()
        try:
            if method in ("Euler","Heun","RK4"):
                values,nfev,first=fixed_integrate(fun,grid,init,method)
                if first is not None:stages[method]=first
            else:
                sol=solve_ivp(fun,(t0,tend),init,method=method,t_eval=grid,rtol=rtol,atol=atol)
                if not sol.success or sol.y.shape[1]!=len(grid):raise ValueError(sol.message)
                values=sol.y.T;nfev=sol.nfev
            duration=time.perf_counter()-start
            results[method]=values
            rows.append({"Método":method,"Evaluaciones f":nfev,"Tiempo (s)":duration})
        except Exception as ex:
            fail.append(f"{method}: {ex}")
    return grid,results,pd.DataFrame(rows),stages,fail

st.title("📈 MARD-EDO Lab")
st.caption("Laboratorio de ecuaciones diferenciales | Versión 1: métodos numéricos clásicos")
st.info("Esta versión todavía no entrena PINN ni ejecuta el método experimental MARD-EDO; prepara sus referencias de comparación.")

with st.sidebar:
    st.header("1 · Seleccionar problema")
    cases=["Decaimiento exponencial","Crecimiento logístico","Forzado rígido",
           "Oscilador amortiguado (orden 2)","Lotka-Volterra (sistema)","Personalizada escalar"]
    selected=st.selectbox("Modelo",cases)
    kappa=st.number_input("Rigidez κ",min_value=0.1,max_value=1000.0,value=50.0) if selected=="Forzado rígido" else 50.
    damping=st.number_input("Amortiguamiento ζ",min_value=0.0,max_value=10.0,value=0.2) if selected=="Oscilador amortiguado (orden 2)" else 0.2
    if selected=="Lotka-Volterra (sistema)":
        alpha=st.number_input("α",min_value=0.01,value=1.1)
        beta=st.number_input("β",min_value=0.001,value=0.4)
        delta=st.number_input("δ",min_value=0.001,value=0.1)
        gamma=st.number_input("γ",min_value=0.01,value=0.4)
    else:alpha,beta,delta,gamma=1.1,0.4,0.1,0.4
    expression=st.text_input("f(t,y) personalizada",value="-2*y",help="Usa t, y; sin, cos, exp, log, sqrt, abs, pi. Ejemplo: cos(t)-y") if selected=="Personalizada escalar" else ""
    st.header("2 · Parámetros")
    if selected=="Personalizada escalar":
        default_ic=[1.];default_span=(0.,3.);names=["y"];model_text="y' = "+expression
    else:
        fun0,names,default_ic,default_span,exact,model_text=model_catalog(selected,kappa,damping,alpha,beta,delta,gamma)
    t0=st.number_input("Tiempo inicial t₀",value=float(default_span[0]),step=0.5)
    tf=st.number_input("Tiempo final T",value=float(default_span[1]),step=0.5)
    init=[st.number_input(f"Condición inicial: {name}(t₀)",value=float(v),key=f"ic_{selected}_{i}") for i,(name,v) in enumerate(zip(names,default_ic))]
    steps=st.slider("Número de pasos fijos (N)",min_value=20,max_value=2500,value=300,step=20)
    methods=st.multiselect("Métodos de comparación",["Euler","Heun","RK4","RK45","Radau","BDF"],default=["Euler","RK4","RK45","BDF"])
    rtol=st.select_slider("Tolerancia relativa (RK45/Radau/BDF)",options=[1e-3,1e-5,1e-7,1e-9],value=1e-7)
    run=st.button("▶ Resolver y comparar",type="primary",use_container_width=True)

st.markdown(f"**Ecuación:** `{model_text}`")
if selected=="Forzado rígido":
    h=(tf-t0)/steps if tf>t0 else 0
    if kappa*h>2.78:
        st.warning(f"Advertencia de estabilidad: κ·h = {kappa*h:.3g} > 2.78. RK4 explícito puede ser inestable. Aumenta N o utiliza BDF/Radau.")
if run:
    if tf<=t0:st.error("El tiempo final debe superar al inicial.");st.stop()
    if not methods:st.error("Selecciona por lo menos un método.");st.stop()
    try:
        if selected=="Personalizada escalar":
            fun=safe_scalar_rhs(expression);exact=None
        else:fun=fun0
        grid,res,benchmark,stages,fail=run_case(fun,np.asarray(init,dtype=float),t0,tf,steps,methods,rtol,rtol/100)
        if not res:raise ValueError("Ningún método convergió con la configuración indicada.")
        exact_vals=None
        if exact is not None:
            if selected=="Crecimiento logístico" and init[0]<0:
                st.warning("La condición inicial negativa puede producir una singularidad; se omite la referencia exacta.")
            else:
                exact_vals=np.asarray(exact(grid,init,t0),dtype=float)
                if not np.all(np.isfinite(exact_vals)):exact_vals=None
        st.session_state["last_run"]={"t":grid,"res":res,"bench":benchmark,"stages":stages,"fail":fail,"names":names,"exact":exact_vals,"model":model_text,"init":init}
    except Exception as ex:st.error(f"No fue posible resolver el problema: {ex}")
if "last_run" in st.session_state:
    d=st.session_state["last_run"]
    for msg in d["fail"]:st.warning(msg)
    fig=go.Figure()
    if d["exact"] is not None:
        fig.add_trace(go.Scatter(x=d["t"],y=d["exact"],name="Exacta (primera variable)",line={"color":"white","dash":"dash","width":3}))
    for meth,vals in d["res"].items():
        for col,name in enumerate(d["names"]):
            fig.add_trace(go.Scatter(x=d["t"],y=vals[:,col],name=f"{meth}: {name}",mode="lines"))
    fig.update_layout(title="Soluciones comparadas",xaxis_title="t",yaxis_title="Estado",template="plotly_dark",height=480,legend={"orientation":"h","y":-0.2})
    st.plotly_chart(fig,use_container_width=True)
    report=d["bench"].copy()
    if d["exact"] is not None:
        for metric in ["MAE","RMSE","Error máximo","L2 relativo"]:report[metric]=np.nan
        for i,row in report.iterrows():
            err=d["res"][row["Método"]][:,0]-d["exact"]
            report.loc[i,"MAE"]=np.mean(np.abs(err))
            report.loc[i,"RMSE"]=np.sqrt(np.mean(err*err))
            report.loc[i,"Error máximo"]=np.max(np.abs(err))
            report.loc[i,"L2 relativo"]=np.linalg.norm(err)/max(np.linalg.norm(d["exact"]),1e-12)
    st.subheader("Tabla de precisión y costo")
    st.dataframe(report,use_container_width=True,hide_index=True)
    if d["exact"] is not None:
        efig=go.Figure()
        for meth,vals in d["res"].items():
            efig.add_trace(go.Scatter(x=d["t"],y=np.abs(vals[:,0]-d["exact"]),name=meth))
        efig.update_layout(title="Error absoluto |aproximada - exacta|",template="plotly_dark",xaxis_title="t",yaxis_title="Error absoluto")
        st.plotly_chart(efig,use_container_width=True)
    st.subheader("Primera iteración de Runge–Kutta (RK4)")
    if "RK4" in d["stages"]:
        one=d["stages"]["RK4"]
        st.latex(r"y_1=y_0+\frac{h}{6}(k_1+2k_2+2k_3+k_4)")
        st.dataframe(pd.DataFrame({"Componente":d["names"],**{k:np.asarray(v) for k,v in one.items()}}),hide_index=True,use_container_width=True)
    else:st.caption("Selecciona RK4 para visualizar sus cuatro pendientes.")
    st.subheader("Descargar resultados")
    data={"t":d["t"]}
    if d["exact"] is not None:data["exacta_variable_1"]=d["exact"]
    for meth,vals in d["res"].items():
        for j,name in enumerate(d["names"]):data[f"{meth}_{name}"]=vals[:,j]
    output=pd.DataFrame(data)
    st.download_button("📥 Descargar soluciones CSV",data=output.to_csv(index=False).encode("utf-8-sig"),file_name="mard_edo_soluciones.csv",mime="text/csv")
    st.download_button("📥 Descargar métricas CSV",data=report.to_csv(index=False).encode("utf-8-sig"),file_name="mard_edo_metricas.csv",mime="text/csv")
    st.caption("Las métricas se calculan sobre la malla de salida. El tiempo depende del equipo, carga y entorno. La solución exacta solo se muestra cuando el modelo proporciona una fórmula válida para la condición inicial.")
else:
    st.subheader("Cómo comenzar")
    st.write("Selecciona una EDO en el panel izquierdo, ajusta sus datos y pulsa **Resolver y comparar**. La aplicación dibujará las soluciones y exportará las tablas.")
