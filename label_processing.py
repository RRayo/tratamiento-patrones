import csv
from pathlib import Path
import shutil

# funcion para copiar las imagenes con objetos detectados en carpetas segun su clasificación en clusters
label_dict = dict()

# obtener un diccionario a partir del archivo con los 
# resultados de la clusterización sobre los objetos detectados en las imagenes
with open('images_labeled.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            if row[2] not in label_dict:
                label_dict[row[2]] = []
            label_dict[row[2]].append(row[1])
            line_count += 1
    print(f'Processed {line_count} lines.')

# path a la carpeta con las imagenes donde se detecto objetos
coco_path = Path("/home/rrayo/train2014")
path = Path(".")

# copia las imagenes en la carpeta correspondiente a la clase asignada
for l in label_dict.keys():
    p = path / l
    p.mkdir(exist_ok=True)
    print(l, len(label_dict[l]))

    for f in label_dict[l]:
        c = coco_path / f
        shutil.copy(c, p)

