# UE6 IA — Modelo Predictivo de Riesgo Académico

Módulo de Machine Learning del *Sistema Web para el Control Pedagógico y Análisis
Predictivo del Riesgo Académico* — U.E. "6 de Junio".

Clasifica a cada estudiante **en cada materia** como
**Sobresaliente / SinRiesgo / EnRiesgo / RiesgoCrítico**, a partir de las notas por
criterio de sus cuatro dimensiones (Ser, Saber, Hacer, Decidir) y el porcentaje de
asistencia. Cumple la RM 0001/2026 (ponderación 10/45/40/5, nota de aprobación 51).

La predicción mira hacia adelante: se entrena contra el resultado del **trimestre
siguiente** en esa misma materia, para que el docente pueda actuar antes de que el
estudiante repruebe.

---

## 1. Decisiones de arquitectura (léelo antes de instalar)

| Tema | Decisión | Por qué |
|------|----------|---------|
| **Modelo** | TensorFlow **Decision Forests** (Gradient Boosted Trees) | El doc especifica "Árboles de Decisión". Es el mejor algoritmo para datos tabulares pequeños (~cientos de estudiantes) y es **interpretable** (importancia de variables, defendible en la tesis). |
| **Python** | **3.11** (no 3.14) | TensorFlow y TF-DF soportan solo 3.9–3.12. Tu venv actual con 3.14 **no sirve** para TF. |
| **Sistema** | **Linux** — nativo (Mint, Ubuntu) o WSL2 | TF-DF **no tiene wheel para Windows**. En Linux se instala directo; en Windows hace falta WSL2, que corre adentro e integra con VS Code y PyCharm. |
| **Cómputo** | **CPU** (sin GPU) | Los árboles de decisión entrenan en CPU; la GPU no los acelera. Con este volumen de datos (~cientos de estudiantes) el entrenamiento en CPU es de segundos. |

---

## 2. Estructura del proyecto

```
ue6dejunio-ia/
├── config/config.yaml          # ← TODO lo configurable (hojas, columnas, umbrales)
├── requirements.txt            # deps de runtime (instalar en WSL2/Py3.11)
├── requirements-dev.txt
├── scripts/
│   ├── setup_linux.sh          # instala Python 3.11 + venv + deps en Linux nativo
│   ├── setup_wsl.sh            # lo mismo, dentro de WSL2
│   ├── check_env.py            # verifica Python, TF/TF-DF y la carpeta de datos
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

Los datos crudos —los registros Excel del colegio— viven fuera del repo, en la
carpeta que apunta `UE6_DATA_ROOT`. **Nunca se versionan**: son calificaciones de
menores de edad, y `.gitignore` cubre `data/`, `models/` y `.env`.

El modelo entrenado tampoco viaja por git: se reconstruye con `cli all` en cada
máquina, que es más barato y más honesto que arrastrar un binario.

---

## 3. Puesta en marcha

Hay dos caminos según el sistema. **En Linux nativo no hace falta WSL**: TF
Decision Forests publica wheel para Linux, así que se instala y corre directo.

### 3.0 Linux (Mint, Ubuntu) — el camino corto

```bash
git clone <repo> && cd ue6dejunio-ia
bash scripts/setup_linux.sh
```

El script instala Python 3.11, crea `~/.venv-ue6`, instala las dependencias y
copia `.env.example` a `.env`. Después:

```bash
source ~/.venv-ue6/bin/activate
export PYTHONPATH=src
python -m ue6_ia.cli all
```

**La ruta de los registros va en `.env`, y en Linux es una ruta normal.** El
prefijo `/mnt/` es exclusivo de WSL, donde así se ven los discos de Windows:

```bash
# Linux Mint — el disco es local:
UE6_DATA_ROOT=/home/tuusuario/ue6/DOCS docentes

# WSL2 — el mismo material, en un disco de Windows:
UE6_DATA_ROOT=/mnt/d/Unidad Educativa 6 de Junio - Sistema/Documentos/DOCS docentes
```

Adentro de esa carpeta tienen que estar `REGISTROS DE AÑOS PASADOS` y
`Registros 2026`, con los nombres tal cual: los busca `config/config.yaml`. Sin
comillas ni escapes en el `.env` aunque la ruta lleve espacios.

Para comprobarlo antes de entrenar:

```bash
python scripts/check_env.py
```

Verifica Python, TensorFlow, TF-DF, las librerías de Excel **y que la carpeta de
datos exista con las dos subcarpetas dentro**. Ese es el error más fácil al mover
el proyecto de máquina, y el que peor avisa por su cuenta: sin las carpetas el
dataset sale vacío con una sola línea de log.

> Una nota sobre Mint: `deadsnakes` sólo publica para los nombres de Ubuntu
> (`jammy`, `noble`), no para los de Mint (`vanessa`, `wilma`…). Agregar el PPA
> sin traducir deja `apt` apuntando a un repositorio inexistente. El script lee
> `UBUNTU_CODENAME` de `/etc/os-release` y lo resuelve solo.

### 3.1 Windows — vía WSL2

#### Una sola vez: instalación de WSL
En **Windows** (PowerShell como administrador):
```
wsl --install -d Ubuntu
```
Reinicia si lo pide. (No se necesita driver NVIDIA: el entrenamiento es en CPU.)

#### Entorno Python (dentro de WSL)
```bash
# Abre Ubuntu y ve al proyecto (montado desde D:)
cd "/mnt/d/Unidad Educativa 6 de Junio - Sistema/ue6dejunio-ia"

bash scripts/setup_wsl.sh        # instala Python 3.11, crea ~/.venv-ue6, instala deps
source ~/.venv-ue6/bin/activate
python scripts/check_env.py      # debe decir LISTO
```

> El repo está en `D:`. Trabajar sobre `/mnt/d` desde WSL funciona; si el I/O de
> Excel resulta lento, copia `data/` a una ruta nativa de WSL (`~/ue6/data`).

#### Configura `.env`
```bash
cp .env.example .env
# edita UE6_DATA_ROOT con la carpeta que contiene 'REGISTROS DE AÑOS PASADOS'
# y 'Registros 2026'. En esta máquina:
#   UE6_DATA_ROOT=/mnt/d/Unidad Educativa 6 de Junio - Sistema/Documentos/DOCS docentes
```

> `.env` **pisa** a `config/config.yaml` (`env.data_root or raw[...]`). Si el
> pipeline no encuentra las carpetas, revisa el `.env` antes que el YAML.

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
PYTHONPATH=src python -m ue6_ia.cli serve           # levanta el servicio de inferencia
```

> Para no repetir `PYTHONPATH=src`: `export PYTHONPATH=src` al inicio de la
> sesión (estando en la carpeta del proyecto), o configúralo en VS Code
> (`.vscode/settings.json` → `"terminal.integrated.env.linux"`).

Variables de entrada del modelo — **cinco, por estudiante y por materia**:

| variable | qué es |
|---|---|
| `being`, `knowing`, `doing`, `deciding` | las notas de esa dimensión, **una lista**, ordenadas por fecha |
| `attendance_pct` | porcentaje de asistencia del trimestre, ya calculado |

Son listas y no promedios porque el promedio esconde que `90/90/20` y `67/67/66`
son cosas distintas. Adentro, `contract.py` las convierte en 30 columnas —siete
estadísticos por dimensión más asistencia y progreso—, y esa expansión es la misma
función que usa el servicio de inferencia: calcularla dos veces sería darle al
modelo, en producción, variables que nunca vio.

La etiqueta sale del **trimestre siguiente** en la misma materia. Etiquetar con el
promedio del mismo trimestre del que salen las notas le pediría al modelo que
aprenda una suma: el promedio trimestral **es** la suma de las cuatro dimensiones.

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
