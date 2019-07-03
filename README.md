# Clusterizacion de Imagenes en funcion de la deteccion de objetos

## Preámbulo
Repo con el proyecto para los ramos CC5509-1 y CASDYTI-1.

## Pasos para reproducir el código


### Detectar objetos con YOLO

Correr YOLO9000 sobre un dataset propio:

Clonar repo:
`git clone --recursive https://github.com/philipperemy/yolo-9000.git`

Cambiar a la carpeta yolo:
`cd yolo-9000`

Copiar pesos de yolo
`cat yolo9000-weights/x* > yolo9000-weights/yolo9000.weights`

Verificar que el valor sea d74ee8d5909f3b7446e9b350b4dd0f44 para ver si se creó bien el archivo
`md5sum yolo9000-weights/yolo9000.weights` 

Actualizar darknet
`git submodule foreach git pull origin master`

Cambiar a carpeta darknet
`cd darknet`

Borrar archivos compilados
`make clean` 

Cambiar las lineas `GPU=1` y `CUDNN=1` en Makefile para activar uso de GPU
`vim Makefile`

Compilar los archivos
`make`

### Correr ejemplo para probarlo
`./darknet detector test cfg/combine9k.data cfg/yolo9000.cfg ../yolo9000-weights/yolo9000.weights data/dog.jpg`

Debería arrojar lo siguiente:
	`Loading weights from ../yolo9000-weights/yolo9000.weights...Done!
	data/dog.jpg: Predicted in 0.035112 seconds.
	car: 70%
	canine: 56%
	bicycle: 57%
	Not compiled with OpenCV, saving to predictions.png instead`


Modificar el archivo para correr el detector en multiples imagenes (se adjunta archivo)
`vim python/darknet.py`

Correr el detector:
`python2 python/darknet.py` (si tira seg fault correr el ejemplo del demo y correr el script de nuevo)



### Clusterización
Ejecutar el script `clusterize.py` que recibe el archivo generado en el paso anterior para posteriormente retornar un archivo de texto plano que contiene el nombre de cada imagen junto con el cluster al cual pertenece.
Asegúrese de haber descargado los arhivos del modelo de `Word2Vec` en el preámbulo antes de ejecutar este paso.

### División de las Imágenes
TODO

## Autores

* **Daniela Quenti**
* **Fabián Villena**
* **Raúl Rayo** :zap:
