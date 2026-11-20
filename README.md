# MARD-EDO Lab — Métodos numéricos y PINN

Aplicación educativa de EDO con Euler, Heun, RK4, RK45, Radau, BDF y una PINN de residuo L² para tres problemas escalares precargados.

## Iniciar en Windows (CMD)

```cmd
cd /d "C:\Users\HP\Desktop\InfO\Python\MARD-EDO-Lab"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Gráficas

- Soluciones numéricas frente a solución exacta cuando está disponible.
- Error absoluto, error logarítmico, número de evaluaciones y tiempos.
- Retrato de fase para modelos con dos variables.
- Predicción PINN frente a solución exacta y a integradores clásicos compatibles.
- Pérdida de entrenamiento por época.
- Residuo diferencial y error absoluto de PINN.

## Alcance científico

PINN disponible solo para decaimiento exponencial, crecimiento logístico y ecuación rígida forzada. Entrena con pérdida L² y condición inicial exacta; **no implementa todavía la formulación H¹ ni el muestreo adaptativo MARD-EDO**. Las métricas se calculan al ejecutar el programa, no son resultados prefijados.

## Despliegue

Publicar el repositorio en GitHub y desplegar `app.py` desde Streamlit Community Cloud. `torch` puede aumentar el consumo de memoria y el tiempo de instalación en la nube.
