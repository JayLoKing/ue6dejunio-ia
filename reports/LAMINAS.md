# Las laminas del modelo predictivo, y que dice cada una

Material de apoyo para el documento oficial del sistema. Las once figuras se
regeneran con:

```bash
PYTHONPATH=src python -m ue6_ia.cli figuras            # escribe reports/figures/*.png
PYTHONPATH=src python -m ue6_ia.cli figuras --resumen  # ademas imprime los numeros citados aqui
PYTHONPATH=src python -m ue6_ia.cli metricas           # las tablas del entrenamiento, sin reentrenar
```

> **Corrida citada:** 2026-10-24, cuatro candidatos (GBT y Random Forest, con y
> sin ponderacion de clases), parada temprana del GBT agrupada por estudiante.
> Gano **`random_forest_ponderado`** (Random Forest con ponderacion `balanceada`).
> Todos los numeros de abajo son de esa corrida; `figuras --resumen` los reimprime.

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
| **En entrenamiento** | Cuanto memorizo de las filas que ya vio | **0.929** |
| **Fuera de muestra** | Como le va con un estudiante que **nunca vio** | **0.634** |

El segundo es el honesto. Sale de validacion cruzada en 5 pliegues **agrupada por
estudiante**: cada chico cae entero de un solo lado del corte, asi que el modelo
no puede reconocerlo en vez de predecirlo. Con ~33 estudiantes cada uno aparece
unas cinco veces (una por trimestre y gestion); un corte por fila dejaria al mismo
chico de los dos lados.

**Al documento va el 0.634.** El 0.929 solo sirve citado junto al otro, para
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

- **10** se detectan como tales,
- **23** se llaman `EnRiesgo` — error, pero el docente igual recibe un aviso,
- **11** salen `SinRiesgo` y **6** `Sobresaliente`.

Esos ultimos **17 son el dato duro del capitulo de limitaciones**: son chicos en
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

**Que muestra.** Accuracy media **0.639** (desvio 0.056) contra linea base media
**0.379** (desvio 0.024). Pliegue por pliegue: 0.585, 0.598, 0.743, 0.650 y
0.619. El modelo supera al tonto por ~26 puntos de media, en los cinco pliegues.

La linea roja en **0.929** es la otra lectura: el hueco entre ella y las barras
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
| **RiesgoCritico** | **0.154** | **0.200** | **0.174** | **50** |
| EnRiesgo | 0.663 | 0.752 | 0.705 | 359 |
| SinRiesgo | 0.560 | 0.488 | 0.522 | 334 |
| Sobresaliente | 0.757 | 0.713 | 0.735 | 349 |
| Promedio macro | 0.534 | 0.538 | 0.534 | 1092 |
| Promedio ponderado | 0.638 | 0.634 | 0.634 | 1092 |

El modelo anda bien en las tres clases abundantes y **mal justo en la que
importa**: detecta uno de cada cinco casos criticos, y de cada 13 avisos de
riesgo critico solo 2 lo son. La ponderacion de clases (cada fila critica pesa
5.46, las demas ~0.8) es lo que sube ese recall respecto del Random Forest sin
ponderar (0.100 sobre los 50 casos), a costa de precision.

---

## Lamina 5 — `dispersion_pliegues.png`

**Que es.** Un boxplot por metrica con los valores de los cinco pliegues, y los
puntos individuales encima. El triangulo verde es la media.

**Por que va.** Con 33 estudiantes, **quien quede afuera cambia el resultado**.
Una media sin su dispersion es una anecdota disfrazada de medicion. El desvio es
parte del resultado, no un adorno.

**Que muestra.** `accuracy` y `linea_base` son cajas angostas: el modelo se
comporta parecido caiga quien caiga. **`recall_riesgo_critico` es una caja
enorme**: media 0.223 con desvio 0.293, con pliegues que van de 0.0 a 0.714. Y
**solo 4 de los 5 pliegues tienen valor**: uno no trajo ni un caso critico, asi
que no midio — y se reporta como hueco, no como cero, porque un cero ahi
castigaria al modelo por un reparto que no eligio.

**Un desvio mayor que la media significa que ese 0.223 no es predecible.**
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

**Que muestra.** `knowing_mean` (0.186), `doing_mean` (0.185), `being_mean`
(0.157), `doing_min` (0.151) y `knowing_min` (0.139) encabezan.

Los dos primeros son esperables: SABER y HACER pesan 45 y 40 sobre 100 en la
ponderacion oficial RM 0001/2026. Los minimos importan casi tanto como las
medias: el modelo mira la peor nota del trimestre, no solo el promedio. Entre
las diez primeras entra `doing_trend` (0.122), la *pendiente*: si el chico viene
cayendo. `attendance_pct`, que encabezaba con el GBT de la corrida anterior, ya
no esta entre las diez primeras de este Random Forest.

---

## Por que gano este modelo — `comparacion_candidatos.png` (lamina 11)

Se compararon **cuatro candidatos** con los mismos 5 pliegues agrupados por
estudiante y la misma semilla:

| Candidato | Accuracy | Macro F1 | Recall critico (pliegues) |
|---|---|---|---|
| `gradient_boosted_trees` | 0.637 ± 0.029 | 0.534 ± 0.062 | 0.062 ± 0.064 (4) |
| `random_forest` | 0.660 ± 0.062 | **0.563** ± **0.095** | 0.125 ± 0.178 (4) |
| `gradient_boosted_trees_ponderado` | 0.635 ± 0.043 | 0.536 ± 0.069 | 0.161 ± 0.239 (4) |
| **`random_forest_ponderado`** | 0.639 ± 0.056 | 0.555 ± 0.079 | **0.223** ± 0.293 (4) |

**La regla** (`training/train.py::_elegir`, no cambio): gana el mejor macro F1;
si otros quedan a menos de la **banda de empate** —el mayor desvio de macro F1
entre pliegues de todos los candidatos—, decide entre ellos el mayor recall
medio de `RiesgoCritico`.

**Esta corrida.** El mejor macro F1 es 0.563 (`random_forest`), y la banda es
0.095, el desvio de ese mismo candidato. Los cuatro caben en
`[0.468, 0.563]`: ninguna diferencia de macro F1 entre ellos es mayor que lo que
la propia metrica se mueve al cambiar de pliegue. Decide el recall critico, y
gana **`random_forest_ponderado`** con 0.223. Ningun candidato quedo excluido.

**Ponderacion balanceada.** Cada fila pesa `n / (k * n_clase)`, calculado solo
con las filas con que se ajusta (nunca con la validacion del pliegue), y la
parada temprana del GBT usa los mismos pesos. En el modelo final: `RiesgoCritico`
5.46, `EnRiesgo` 0.76, `SinRiesgo` 0.82, `Sobresaliente` 0.78. El peso viaja en
una columna aparte; no es una feature y el servicio no la ve.

**La lamina.** Dos paneles con media ± desvio por candidato: macro F1 con la
banda sombreada, y recall critico. El ganador va en rojo y en rombo. Que el
desvio del recall sea mayor que la media en todos es la misma advertencia de la
lamina 5: la ventaja del ganador es real en la media y fragil pliegue a pliegue.

---

## De donde salen las laminas 7 a 10

El generador sigue sin cargar TensorFlow. Para estas laminas `train` deja dos
archivos mas en `models/tfdf_riesgo/`:

- `registro_entrenamiento.json` — por modelo y por pliegue, la perdida y el
  accuracy de cada iteracion del GBT (en el conjunto de ajuste y en el de parada
  temprana), el tamano real de cada particion y las iteraciones del modelo final.
- `predicciones_fuera_de_pliegue.parquet` — una fila por observacion: pliegue,
  clase real, clase predicha y la probabilidad de cada una de las cuatro clases,
  predicha por el pliegue que **no** vio a ese estudiante. Sin nombres: `fila` es
  la posicion en el dataset.

Un modelo entrenado antes de que existieran no los trae. Entonces las laminas 8
a 10 se **omiten con un aviso** y las otras siete salen igual.

### La parada temprana tambien se separa por estudiante

El GBT decide cuantas iteraciones conservar mirando una perdida de validacion.
TF-DF, por defecto, aparta para eso el 10% de las **filas**, y el mismo chico
quedaba en el ajuste y en la parada: la curva de validacion salia optimista.
Ahora, dentro de cada pliegue, se aparta ~10% de los **estudiantes** de la
porcion de entrenamiento (`StratifiedGroupKFold` de 10 cortes, primer corte) y
se le pasa al GBT como `validation_data`. Hay tres conjuntos sin un estudiante
en comun —ajuste, parada temprana y validacion del pliegue— y el entrenamiento
**falla** si alguno se repite. El modelo final hace lo mismo sobre todo el
dataset (90/10 por estudiante). El Random Forest no para temprano y entrena con
la porcion entera.

El modelo que se guarda conserva la iteracion de **mejor perdida de validacion**,
no la ultima: TF-DF entrena unas iteraciones de mas para confirmar que la perdida
ya no baja y despues recorta.

---

## Lamina 7 — `matriz_confusion_normalizada.png`

**Que es.** La misma matriz de la lamina 1, con cada fila convertida en
porcentaje de su clase real. Cada celda muestra el porcentaje y, abajo, el
conteo. Las clases van de la mas grave a la mejor.

**Como se lee.** La diagonal es el recall de cada clase. Una fila que reparte su
porcentaje fuera de la diagonal es una clase que el modelo confunde, y **hacia
que columna** se va dice si la confusion es tolerable (`Riesgo critico` →
`En riesgo`: el docente igual recibe aviso) o grave (`Riesgo critico` →
`Sobresaliente`).

**Por que va ademas de la lamina 1.** Con 50 filas de `RiesgoCritico` contra
~350 de cada otra clase, en conteos la fila critica se ve palida aunque el modelo
la falle casi entera. En porcentaje se ve tal cual.

---

## Lamina 8 — `resultados_entrenamiento.png`

**Que es.** Cuatro paneles contra el numero de iteraciones (en un GBT de cuatro
clases, cada iteracion agrega un arbol por clase): perdida en entrenamiento,
perdida en el conjunto de parada temprana, accuracy en entrenamiento y accuracy
en el conjunto de parada temprana. La linea gruesa es el modelo final; las finas
y grises, cada pliegue de la validacion cruzada. La recta roja discontinua marca
donde se detuvo el entrenamiento; la punteada, la iteracion que el modelo
conserva (mejor perdida de validacion).

**Como se lee.** La perdida de entrenamiento baja siempre: el modelo se ajusta
cada vez mejor a lo que ya vio. La de parada temprana baja al principio y despues
se aplana o sube: desde ahi, cada arbol nuevo memoriza en vez de generalizar. La
distancia entre los dos paneles de accuracy es la misma memorizacion que muestra
la lamina 3.

**Si gana un Random Forest (esta corrida).** Un RF no tiene iteraciones ni
parada temprana; lo que registra TF-DF es su evaluacion **fuera de bolsa (OOB)**
en ~30 puntos entre 1 y 300 arboles: accuracy y log loss sobre las filas que el
bootstrap de cada arbol no tomo. La lamina muestra entonces dos paneles —perdida
OOB y accuracy OOB contra el numero de arboles—, modelo final grueso y pliegues
finos. En el modelo final la perdida OOB baja de 12.9 (1 arbol) a 0.79 (300) y
el accuracy OOB llega a 0.689.

**Ojo: OOB muestrea filas, no estudiantes.** El mismo chico aparece en varias
filas, y una puede quedar dentro de la bolsa y otra fuera: el OOB lo ve
"parecido" y sale optimista, igual que el corte por fila. Por eso el titulo lo
dice y el numero honesto sigue siendo el de la validacion agrupada (0.634). Con
ponderacion, no se verifico si TF-DF pondera tambien la evaluacion OOB.

**Cuando no sale.** Si el modelo es anterior a `registro_entrenamiento.json`, o
no registro ni iteraciones ni OOB, la lamina se omite con un aviso.

---

## Lamina 9 — `curva_precision_confianza.png`

**Que es.** Para cada clase, uno contra el resto: si el sistema solo dijera "esta
clase" cuando la probabilidad que le asigna es al menos `t`, que proporcion de
esos avisos seria correcta. Se recorren 101 umbrales de 0 a 1 sobre las
predicciones fuera de pliegue. La linea negra gruesa es la media de las clases.

**Como se lee.** Subir el umbral hace avisar menos y, en general, acertar mas en
lo que se avisa. Una curva que **se corta** no cayo a cero: a partir de ahi
ninguna fila llega a esa confianza, asi que no hay precision que medir (se
dibuja como hueco, nunca como cero). Un tramo **en cero** si es medicion: hubo
filas con esa confianza y ninguna era de la clase.

**La media solo donde miden todas.** La linea negra se dibuja solo en los
umbrales donde las cuatro clases tienen precision, y se corta donde se corta la
primera. Promediar las que quedaban hacia saltar la linea justo cuando salia
`RiesgoCritico`, y ese salto se leia como una mejora. En esta corrida la media
llega a 0.76 con confianza 0.69 y se corta en 0.82.

---

## Lamina 10 — `curva_recall_confianza.png`

**Que es.** Lo mismo, con el recall: de todos los casos reales de la clase, que
proporcion alcanza al menos la confianza `t`.

**Como se lee.** En `t = 0` todo pasa y el recall es 1. A medida que se exige mas
confianza se pierden casos. **Que tan rapido cae la curva de `Riesgo critico`**
es la lectura que importa: si cae enseguida, el modelo casi nunca esta seguro de
que un chico esta en riesgo critico, y un umbral alto lo dejaria sin avisar.
Esta lamina y la anterior juntas son el compromiso entre avisar de mas y avisar
de menos.

---

## Resumen citable

- Modelo elegido: **Random Forest con ponderacion de clases balanceada**
  (`random_forest_ponderado`), entre cuatro candidatos: los cuatro empatan en
  macro F1 dentro de la banda de 0.095 y gana por recall critico.
- Dataset: **1092 filas de 33 estudiantes**, gestiones 2023-2025.
- Validacion: **5 pliegues, `StratifiedGroupKFold` agrupado por estudiante**.
- Accuracy fuera de muestra **0.634** contra linea base **0.329**.
- Macro F1 fuera de muestra **0.534** (media entre pliegues 0.555 ± 0.079).
- **Recall de `RiesgoCritico` fuera de muestra: 0.200** (10 de 50). Media entre
  pliegues 0.223 con desvio 0.293, medida en 4 de 5 pliegues.
- Acierto en entrenamiento 0.929 (memorizacion, no estimacion).
- Modelo final: 300 arboles, entrenado con las 1092 filas.

> El numero de la media entre pliegues (0.223) y el de la matriz acumulada
> (0.200) no coinciden, y los dos estan bien: el primero promedia el recall de
> cada pliegue dandole el mismo peso, el segundo cuenta los 50 casos juntos. El
> documento deberia citar **0.200** como la deteccion efectiva y 0.223 ± 0.293
> como la medicion por pliegues con su dispersion.
