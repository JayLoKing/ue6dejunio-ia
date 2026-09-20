# Las laminas del modelo predictivo, y que dice cada una

Material de apoyo para el documento oficial del sistema. Las seis figuras se
regeneran con:

```bash
PYTHONPATH=src python -m ue6_ia.cli figuras            # escribe reports/figures/*.png
PYTHONPATH=src python -m ue6_ia.cli figuras --resumen  # ademas imprime los numeros citados aqui
```

El paquete no se instala (`pip install -e .` falla en `/mnt/d`), asi que el
`PYTHONPATH=src` no es opcional: sin el, el comando muere con
`No module named 'ue6_ia'`. En PowerShell: `$env:PYTHONPATH="src"`.

El comando **no carga el modelo**: lee los JSON que dejo el entrenamiento en
`models/tfdf_riesgo/`. Por eso corre en Windows sin WSL, que es donde se escribe
el documento. TF-DF solo tiene wheel para Linux.

Las figuras no se versionan (`.gitignore` reserva `reports/figures/`). Son
derivadas: se vuelven a dibujar desde los reportes cuando se las necesita.

---

## Dos mediciones distintas, y cual va en el documento

El modelo tiene **dos numeros de acierto**, y confundirlos invalida el capitulo:

| | Que mide | Valor |
|---|---|---|
| **En entrenamiento** | Cuanto memorizo de las filas que ya vio | **0.924** |
| **Fuera de muestra** | Como le va con un estudiante que **nunca vio** | **0.627** |

El segundo es el honesto. Sale de validacion cruzada en 5 pliegues **agrupada por
estudiante**: cada chico cae entero de un solo lado del corte, asi que el modelo
no puede reconocerlo en vez de predecirlo. Con ~33 estudiantes cada uno aparece
unas cinco veces (una por trimestre y gestion); un corte por fila dejaria al mismo
chico de los dos lados.

**Al documento va el 0.627.** El 0.924 solo sirve citado junto al otro, para
mostrar la distancia — que es exactamente cuanto memorizo.

---

## Lamina 1 — `matriz_confusion.png`

**Que es.** Las filas son la clase real, las columnas la que el modelo predijo.
La diagonal son los aciertos. Es la suma de los cinco pliegues, es decir **fuera
de muestra**: cada pliegue evalua estudiantes que el modelo no vio, y juntos
cubren el dataset entero.

**Por que va.** El accuracy dice *cuanto* se equivoca; esta lamina dice **hacia
donde**, y en este sistema los errores no valen igual. Confundir `EnRiesgo` con
`RiesgoCritico` deja igual al docente avisado. Confundirlo con `Sobresaliente`
es un chico que se cae sin que nadie mire.

**Que muestra en esta corrida.** De los **50 casos reales de `RiesgoCritico`**:

- **11** se detectan como tales,
- **26** se llaman `EnRiesgo` — error, pero el docente igual recibe un aviso,
- **6** salen `SinRiesgo` y **7** `Sobresaliente`.

Esos ultimos **13 son el dato duro del capitulo de limitaciones**: son chicos en
riesgo critico que el sistema presenta como si estuvieran bien. No es un error
estadistico neutro, es el unico error que el sistema existe para evitar.

---

## Lamina 2 — `distribucion_clases.png`

**Que es.** Cuantas filas del dataset caen en cada clase.

**Por que va.** Es la lamina que **justifica toda la eleccion de metricas**. Sin
ella, el lector no puede saber si un accuracy es bueno.

**Que muestra.** `EnRiesgo` 359 (32.9%), `Sobresaliente` 349 (32.0%), `SinRiesgo`
334 (30.6%) y **`RiesgoCritico` apenas 50 (4.6%)**.

La clase que mas importa es la mas rara. Un modelo que **jamas** prediga
`RiesgoCritico` acierta el 95.4% de las veces en esa clase y no sirve para nada.
Por eso el documento no puede reportar accuracy a secas.

---

## Lamina 3 — `accuracy_vs_linea_base.png`

**Que es.** Pliegue por pliegue (P1..P5): en azul el acierto del modelo, en gris
lo que sacaria un modelo tonto que **siempre responde la clase mas frecuente**.
La linea roja punteada es el acierto en entrenamiento.

**Por que va.** Un 0.63 aislado no se puede leer. Contra su linea base si:
demuestra que el modelo **aprendio algo** y no solo a repetir.

**Que muestra.** Accuracy media **0.629** (desvio 0.024) contra linea base media
**0.379**. El modelo supera al tonto por ~25 puntos en los cinco pliegues, de
forma consistente.

La linea roja en **0.924** es la otra lectura: el hueco entre ella y las barras
azules es sobreajuste. El modelo memoriza bastante mejor de lo que generaliza —
esperable con 33 estudiantes, y hay que decirlo.

---

## Lamina 4 — `metricas_por_clase.png`

**Que es.** Precision, recall y F1 de cada clase, fuera de muestra. El rotulo de
`RiesgoCritico` va en rojo porque es la que manda.

**Como se leen los dos numeros.**
- **Recall** — de los que estaban en riesgo, a cuantos detecto. Un falso negativo
  es un chico que reprueba sin aviso.
- **Precision** — de los que marco, cuantos lo estaban. Un falso positivo cuesta
  una revision de mas.

**No cuestan lo mismo**, y por eso el recall de `RiesgoCritico` es la metrica
principal del sistema.

**Que muestra.**

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| Sobresaliente | 0.746 | 0.734 | 0.740 | 349 |
| EnRiesgo | 0.638 | 0.688 | 0.662 | 359 |
| SinRiesgo | 0.548 | 0.512 | 0.529 | 334 |
| **RiesgoCritico** | **0.220** | **0.220** | **0.220** | **50** |

El modelo anda bien en las tres clases abundantes y **mal justo en la que
importa**: detecta poco mas de uno de cada cinco casos criticos.

---

## Lamina 5 — `dispersion_pliegues.png`

**Que es.** Un boxplot por metrica con los valores de los cinco pliegues, y los
puntos individuales encima. El triangulo verde es la media.

**Por que va.** Con 33 estudiantes, **quien quede afuera cambia el resultado**.
Una media sin su dispersion es una anecdota disfrazada de medicion. El desvio es
parte del resultado, no un adorno.

**Que muestra.** `accuracy` y `linea_base` son cajas angostas: el modelo se
comporta parecido caiga quien caiga. **`recall_riesgo_critico` es una caja
enorme**: media 0.286 con desvio 0.287, con pliegues que van de 0.0 a 0.75. Y
**solo 4 de los 5 pliegues tienen valor**: uno no trajo ni un caso critico, asi
que no midio — y se reporta como hueco, no como cero, porque un cero ahi
castigaria al modelo por un reparto que no eligio.

**Un desvio del tamano de la media significa que ese 0.286 no es predecible.**
Es la limitacion mas seria del modelo y va dicha con estas palabras.

---

## Lamina 6 — `importancia_variables.png`

**Que es.** Las variables ordenadas por cuanto pesan en la decision, segun
`INV_MEAN_MIN_DEPTH` — que tan arriba del arbol aparece cada una. Cuanto mas
arriba, a mas casos afecta.

**Por que va.** Es la ventaja de los arboles de decision sobre una red neuronal:
se puede **responder por que**. Un sistema que le dice a un docente que un chico
esta en riesgo tiene que poder explicar en que se baso; si no, la escuela no
tiene motivo para creerle.

**Que muestra.** `knowing_mean` (0.275), `doing_mean` (0.266) y
**`attendance_pct` (0.221)** encabezan.

Los dos primeros son esperables: SABER y HACER pesan 45 y 40 sobre 100 en la
ponderacion oficial RM 0001/2026. El tercero **no estaba dado**: la asistencia
aparece arriba por si sola, sin que nadie se lo indicara. Tambien entran
`doing_trend` y `knowing_trend`, que son la *pendiente* — el modelo mira si el
chico viene cayendo, no solo donde esta parado.

---

## Lo que esta lamina no existe, y por que

**No hay curva de entrenamiento** (logloss contra numero de arboles). Vive en
`make_inspector().training_logs()` y el entrenamiento **no la persiste**.
Extraerla obligaria a cargar el modelo con TF-DF, y entonces las figuras del
documento solo se podrian regenerar desde WSL.

La decision fue mantener el generador libre de TensorFlow. Para recuperar esa
lamina hay que guardar los logs en `training/train.py` y reentrenar.

---

## Resumen citable

- Modelo elegido: **gradient_boosted_trees** (gana por macro F1 sobre estudiantes
  no vistos).
- Dataset: **1092 filas de 33 estudiantes**, gestiones 2023-2025.
- Validacion: **5 pliegues, `StratifiedGroupKFold` agrupado por estudiante**.
- Accuracy fuera de muestra **0.627** contra linea base **0.329**.
- Macro F1 fuera de muestra **0.538**.
- **Recall de `RiesgoCritico` fuera de muestra: 0.220.** Media entre pliegues
  0.286 con desvio 0.287.
- Acierto en entrenamiento 0.924 (memorizacion, no estimacion).

> El numero de la media entre pliegues (0.286) y el de la matriz acumulada
> (0.220) no coinciden, y los dos estan bien: el primero promedia el recall de
> cada pliegue dandole el mismo peso, el segundo cuenta los 50 casos juntos. El
> documento deberia citar **0.220** como la deteccion efectiva y 0.286 ± 0.287
> como la medicion por pliegues con su dispersion.
