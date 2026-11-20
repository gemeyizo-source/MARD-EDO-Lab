# MARD-EDO Lab — v1

Laboratorio de ecuaciones diferenciales en Streamlit.

## Instalación en Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Funciones actuales

- EDO escalar de primer orden: modelos precargados y función personalizada con sintaxis restringida.
- Sistemas: oscilador amortiguado de orden 2 y Lotka–Volterra.
- Euler, Heun y RK4 de paso fijo; RK45, Radau y BDF de SciPy.
- Gráficas, métricas cuando existe solución analítica y exportación CSV.

**MARD-EDO/PINN todavía no está implementado.** No usar esta versión para atribuir resultados experimentales a MARD-EDO.
