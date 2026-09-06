# Code Review Rules — ue6dejunio-ia

Modelo predictivo de riesgo académico. Python 3.11, TensorFlow Decision Forests
(Gradient Boosted Trees), FastAPI para inferencia. Paquete: `src/ue6_ia`.

Consume registros Excel históricos (2023-2025) y datos del sistema `ue6dejunio-api`
(2026+). Sirve predicciones que la API Spring Boot persiste en `risk_predictions`.

**Es un prototipo declarado**, limitado por la poca data disponible. No exijas
garantías de modelo en producción, pero sí honestidad en lo que se reporta.

## Datos faltantes — la regla que más importa acá

- **Un dato que falta es `None`/`NaN`, nunca `0`.** TF-DF trata el faltante como
  tal; un cero es una nota mala inventada.
- **Prohibido `valor or 0.0`, `valor or 0` y equivalentes** sobre lecturas de
  planilla o de base. Ese patrón ya causó el bug vivo más caro del proyecto: la
  detección de la columna SER fallaba y `_to_float(None) or 0.0` la convertía en
  cero, dejando 107 de 181 filas con una dimensión nula que parecía una nota real.
  Un fallo de detección tiene que ser distinguible de un cero legítimo.
- Si una columna esperada no aparece, hay que registrarlo (`logger.warning`) o
  fallar, no rellenar en silencio.

## Contrato de features — fuente única

- `src/ue6_ia/contract.py` es **la única** construcción del vector de features.
  La usan el pipeline de entrenamiento y `serving/api.py`.
- Marcá cualquier cálculo de features duplicado fuera de ahí: dos rutas distintas
  producen training/serving skew, y el modelo se degrada **sin un solo error**,
  solo prediciendo peor.
- `FEATURE_COLS` define nombres y orden. Cualquier cambio en el vector se hace ahí
  y tiene que quedar cubierto por el test que compara `expandir()` contra
  `FEATURE_COLS`.
- Las notas se normalizan a porcentaje del tope de **su gestión** (`Escala`), no
  contra los topes de hoy. La escuela ponderó distinto: 2023 reparte Saber 35 y
  Hacer 35; 2024 en adelante, 45 y 40.
- **La ponderación se mide en la carpeta que tiene notas, no en cualquiera.** Hay
  dos carpetas del mismo curso 2024 con templates distintos: la vacía conserva
  `SABER - 35` y la que tiene 3304 notas dice `SABER - 45`. Leer la escala de la
  equivocada hace rebotar notas legítimas contra un tope que no era el suyo.

## Honestidad de las métricas

- **El split no puede partir filas cuando el mismo estudiante aparece muchas
  veces.** El dataset tiene ~34 estudiantes con ~5 filas cada uno; un
  `train_test_split` por fila deja al mismo chico en train y en test e infla el
  accuracy. Usá `GroupShuffleSplit` u `groups=` por estudiante.
- No reportes accuracy sin la línea base de la clase mayoritaria. Con clases
  desbalanceadas (RiesgoCritico ≈ 6% de las filas) el accuracy solo no dice nada.
- `model.compile(metrics=[...])` sin `loss` hace que la loss reportada sea `0.0` y
  no signifique nada. No guardes esa métrica como si fuera real.
- Marcá cualquier feature que sea función determinística del target, o al revés:
  el promedio trimestral **es** la suma de las cuatro dimensiones
  (`total_score` es columna generada en la BDD), así que etiquetar con él y
  entrenar con ellas es aprender aritmética, no riesgo.

## Ingesta de Excel

- Las columnas se detectan **por encabezado**, nunca por índice fijo: el número de
  criterios varía por área y el template cambia entre gestiones.
- El match de nombres de hoja y de encabezado es tolerante a acentos, espacios y
  mayúsculas (`_nb()`). Las letras espaciadas (`P R O M E D I O`) son normales.
- Un template nuevo no puede romper los anteriores en silencio: si la detección no
  encuentra un bloque, tiene que decirlo.

## Privacidad

- **Son datos de menores de edad.** `data/`, `models/` y `.env` están en
  `.gitignore` y ahí se quedan. Marcá cualquier cosa que versione datasets,
  planillas, parquet o modelos entrenados.
- Nunca loguees nombres de estudiantes, y menos en `logger.info`. Contá filas,
  no personas.
- No hardcodees rutas de datos ni tokens: van por `config.yaml` o `UE6_*`.
  Ojo que `.env` **pisa** a `config.yaml` (`env.data_root or raw[...]`).

## Servicio de inferencia

- `serving/api.py` es una función pura: sin base de datos, sin saber qué es un
  PDC. La dirección de la llamada es **Spring → FastAPI**, nunca al revés.
- El token se compara en tiempo constante (`secrets.compare_digest`), no con `!=`.
- `probability_score` tiene que decir qué probabilidad es. Devolver `max(proba)`
  guarda la confianza en la clase ganadora, y un "Sobresaliente 0.95" termina
  leyéndose como 95% de riesgo en el panel.
- El servicio no se expone a internet: localhost o red interna.

## Python / estilo

- Ruff con `line-length = 100`, reglas `E, F, I, UP, B, SIM`. Sin imports ni
  variables sin usar.
- Type hints en toda función pública, con `from __future__ import annotations`.
- Sin `print()` para diagnóstico: `logging`. **Excepción: `scripts/`**, que son
  reportes de consola para una persona — ahí `print` ES la salida, no un rastro de
  depuración, y `logging.basicConfig` la ensuciaría con timestamps y niveles.
- Excepciones concretas, no `except Exception` desnudo salvo con `# noqa: BLE001`
  y un motivo escrito.
- TF y TF-DF se importan **dentro** de la función que los usa: importarlos arriba
  hace que la lógica pura no se pueda testear sin TensorFlow.

## Testing

- Los tests (`tests/test_*.py`) están **fuera** del alcance de esta revisión
  (excluidos en `.gga`). Existen y los corre pytest. **No marques "faltan tests"
  ni "no hay tests en el changeset"**: no los estás viendo.
- TDD estricto. Toda lógica pura (estadísticos, escalas, etiquetado, parseo de
  nombres) se testea sin TensorFlow ni pandas.

## Idioma de los artefactos

- **Identificadores y docstrings en español**, que es la convención del repo
  (`construir_features`, `clasificar_riesgo`, `entrenar`, `expandir`). No lo
  marques como violación.
- **Excepción deliberada**: los nombres de las cuatro dimensiones en el vector de
  features (`being`, `knowing`, `doing`, `deciding`) están en inglés porque son
  los valores exactos que la BDD acepta en `evaluation_criteria.dimension`.
- Mensajes de commit en inglés.
