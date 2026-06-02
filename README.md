# UE6 IA — Modelo Predictivo de Riesgo Académico

Módulo de Machine Learning del *Sistema Web para el Control Pedagógico y Análisis
Predictivo del Riesgo Académico* — U.E. "6 de Junio".

Clasifica a cada estudiante en **Sobresaliente / SinRiesgo / EnRiesgo / RiesgoCrítico**
a partir de sus calificaciones por dimensión (Ser, Saber, Hacer, Autoevaluación),
el rendimiento por área y el porcentaje de asistencia. Cumple la
RM 0001/2026 (ponderación 10/45/40/5, nota de aprobación 51).

---

## 1. Decisiones de arquitectura (léelo antes de instalar)

| Tema | Decisión | Por qué |
|------|----------|---------|
| **Modelo** | TensorFlow **Decision Forests** (Gradient Boosted Trees) | El doc especifica "Árboles de Decisión". Es el mejor algoritmo para datos tabulares pequeños (~cientos de estudiantes) y es **interpretable** (importancia de variables, defendible en la tesis). |
| **Python** | **3.11** (no 3.14) | TensorFlow y TF-DF soportan solo 3.9–3.12. Tu venv actual con 3.14 **no sirve** para TF. |
| **Sistema** | **WSL2 (Ubuntu)**, no Windows nativo | TF-DF **no tiene wheel para Windows** (solo se distribuye como código fuente, no compilable en Windows). WSL2 corre dentro de Windows e integra con PyCharm. |
| **Cómputo** | **CPU** (sin GPU) | Los árboles de decisión entrenan en CPU; la GPU no los acelera. Con este volumen de datos (~cientos de estudiantes) el entrenamiento en CPU es de segundos. |

---

## 2. Estructura del proyecto

```
ue6dejunio-ia/
├── config/config.yaml          # ← TODO lo configurable (hojas, columnas, umbrales)
├── requirements.txt            # deps de runtime (instalar en WSL2/Py3.11)
├── requirements-dev.txt
├── scripts/
│   ├── setup_wsl.sh            # instala Python 3.11 + venv + deps en WSL
│   ├── check_env.py            # verifica TF/TF-DF (Python 3.11, no Windows)
│   └── inspect_sheet.py        # vuelca hojas Excel para mapear layout
├── src/ue6_ia/
│   ├── config.py               # carga config.yaml + .env
│   ├── ingestion/              # lectura .xlsb/.xlsx, descubrimiento de cursos
│   ├── preprocessing/          # limpieza de nombres + matriz de features
│   ├── labeling.py             # reglas de categoría de riesgo
│   ├── pipeline.py             # Excel → dataset parquet
│   ├── training/               # entrenamiento TF-DF (CPU)
│   ├── evaluation/             # métricas + importancia de variables
│   ├── serving/api.py          # FastAPI (lo consume Spring Boot)
│   └── cli.py                  # comandos del pipeline
├── data/                       # (gitignored) raw/interim/processed
├── models/                     # (gitignored) modelo entrenado
└── tests/
```

Los datos crudos viven en `F:\DATA PARA MODELO IA PROYECTO DE GRADO`
(en WSL: `/mnt/f/...`). **Nunca se versionan** (datos de menores; ver `.gitignore`).

---

## 3. Puesta en marcha (WSL2)

### 3.1 Una sola vez: instalación de WSL
En **Windows** (PowerShell como administrador):
```
wsl --install -d Ubuntu
```
Reinicia si lo pide. (No se necesita driver NVIDIA: el entrenamiento es en CPU.)

### 3.2 Entorno Python (dentro de WSL)
```bash
# Abre Ubuntu y ve al proyecto (montado desde D:)
cd "/mnt/d/Unidad Educativa 6 de Junio - Sistema/ue6dejunio-ia"

bash scripts/setup_wsl.sh        # instala Python 3.11, crea ~/.venv-ue6, instala deps
source ~/.venv-ue6/bin/activate
python scripts/check_env.py      # debe decir LISTO
```

> El repo está en `D:`. Trabajar sobre `/mnt/d` desde WSL funciona; si el I/O de
> Excel resulta lento, copia `data/` a una ruta nativa de WSL (`~/ue6/data`).

### 3.3 Configura `.env`
```bash
cp .env.example .env
# edita UE6_DATA_ROOT=/mnt/f/DATA PARA MODELO IA PROYECTO DE GRADO
```

---

## 4. Editor con intérprete WSL

### VS Code (gratis) — recomendado aquí
1. Extensiones: **WSL** + **Python**.
2. `Ctrl+Shift+P` → **WSL: Connect to WSL using Distro** → **Ubuntu**.
3. **File → Open Folder** → `/mnt/d/Unidad Educativa 6 de Junio - Sistema/ue6dejunio-ia`.
4. `Ctrl+Shift+P` → **Python: Select Interpreter** → `~/.venv-ue6/bin/python`.

### PyCharm Professional
*Settings → Python Interpreter → Add → On WSL → Ubuntu →* `~/.venv-ue6/bin/python`.

> PyCharm **Community** no soporta intérprete WSL. Usa VS Code, o edita en
> PyCharm (Windows) y entrena desde una terminal WSL.

---

## 5. Uso del pipeline

El proyecto vive en `/mnt/d` (DrvFs), donde `pip install -e .` falla por permisos.
Por eso se ejecuta con `PYTHONPATH=src` (sin instalar el paquete):

```bash
PYTHONPATH=src python -m ue6_ia.cli build-dataset   # Excel → data/processed/dataset_entrenamiento.parquet
PYTHONPATH=src python -m ue6_ia.cli train           # entrena TF-DF → models/tfdf_riesgo/
PYTHONPATH=src python -m ue6_ia.cli evaluate        # matriz de confusión, F1, importancia de variables
PYTHONPATH=src python -m ue6_ia.cli all             # las tres en secuencia
```

> Para no repetir `PYTHONPATH=src`: `export PYTHONPATH=src` al inicio de la
> sesión (estando en la carpeta del proyecto), o configúralo en VS Code
> (`.vscode/settings.json` → `"terminal.integrated.env.linux"`).

Variables de entrada del modelo (doc HU-019) — exactamente las que envía el
sistema por estudiante: `ser`, `saber`, `hacer`, `auto`, `attendance_pct`.
El entrenamiento usa esas mismas 5 variables con datos de gestiones pasadas.

---

## 6. Servicio de inferencia ↔ Spring Boot

```bash
uvicorn ue6_ia.serving.api:app --host 0.0.0.0 --port 8001
```

La API `ue6dejunio-api` (Spring Boot) llama por HTTP y persiste el resultado en
`RiskPredictions`:

```
POST http://<host-wsl>:8001/predict
Authorization: Bearer <UE6_API_TOKEN>
{ "ser": 8.5, "saber": 30, "hacer": 28, "auto": 4, "attendance_pct": 82 }
→ { "risk_level": "EnRiesgo", "probability_score": 0.78, "probabilidades": {...} }
```

`GET /health` indica si el modelo está cargado.

---

## 7. Datos: estado y pendientes

- **Etiquetas (labels):** salen de `PROMEDIOS POR TRIMESTRE` del centralizador de
  **gestiones pasadas (2023–2025)**, que sí tienen ciclo completo y situación final
  ("perdió el año"). La gestión **2026 está en curso** (registros aún vacíos) → se
  usa para inferir, no para entrenar.
- **`.xlsb` vs `.xlsx`:** 2026 es binario (`pyxlsb`), años pasados es `.xlsx`
  (`openpyxl`). El lector unifica ambos.
- **Asistencia:** `attendance_pct` se calcula contando marcas en las hojas
  mensuales (`.`=presente, `F`=falta, `R`=retraso, `L`=licencia):
  `(presente+retraso+licencia)/dias_habiles`. Meses por trimestre y marcas son
  configurables en `config.yaml → asistencia`. Validado contra gestión 2025.
- **Pendiente de confirmar con docentes** (ya preparado en `config.yaml`):
  - **Control de lectura / tareas:** desactivados por defecto
    (`features_opcionales`). Si se confirma que inciden en alguna dimensión,
    actívalos sin tocar el código del modelo.
  - **Índices de columnas por dimensión** (`layout_area`): verificados en 2026;
    revisar si cambian entre gestiones.

---

## 8. Calidad

```bash
ruff check src tests      # lint
black src tests           # formato
pytest                    # pruebas de lógica (no requieren TF)
```
