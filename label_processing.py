import csv
from pathlib import Path
import shutil
import sys

detecciones = 'images_labeled.csv'
data = "/home/rrayo/train2014"
out_dir = "."

try:
    detecciones = sys.argv[1]
    data = sys.argv[2]
    out_dir = sys.argv[3]
except:
    pass

# funcion para copiar las imagenes con objetos detectados en carpetas segun su clasificación en clusters
label_dict = dict()

# obtener un diccionario a partir del archivo con los 
# resultados de la clusterización sobre los objetos detectados en las imagenes
with open(detecciones) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            if row[1] not in label_dict:
                label_dict[row[1]] = []
            label_dict[row[1]].append(row[0])
            line_count += 1
    print(f'Processed {line_count} lines.')

# path a la carpeta con las imagenes donde se detecto objetos
coco_path = Path(data)
path = Path(".")

# copia las imagenes en la carpeta correspondiente a la clase asignada
for l in label_dict.keys():
    p = path / l
    p.mkdir(exist_ok=True)
    print(l, len(label_dict[l]))

    for f in label_dict[l]:
        c = coco_path / f
        shutil.copy(c, p)

