# El Intersection over Union (IoU) nos permite medir que tan bueno es nuestro detector prediciendo bounding boxes
# Calcula la razon entre la interseccion de areas sobre la union de areas
# IoU = Interseccion areas / Union areas


def computeIoU(boxA, boxB):
	# Determinar las coordenadas (x, y) del rectangulo de interseccion 
    # Obtenemos las esquinas: superior derecha e inferior izquierda
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# Calculamos el area de interseccion
	intersectionArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# Calculamos el area de ambos rectangulos, para poder luego calcular la union
    # Para la union tomamos la suma de ambas areas - la interseccion
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	unionArea = float(boxAArea + boxBArea - intersectionArea)

    # Calculamos la metrica IoU
	iou = intersectionArea / unionArea

	return iou