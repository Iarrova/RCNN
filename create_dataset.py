from utils.iou import computeIoU
from utils import config
from bs4 import BeautifulSoup
from imutils import paths
import cv2
import os

# Buscamos los directorios donde guardaremos las imagenes
# En caso de que no existan, los creamos
for dirPath in (config.POSITVE_PATH, config.NEGATIVE_PATH):
	if not os.path.exists(dirPath):
		os.makedirs(dirPath)

# Obtenemos todos los paths de las imagenes
imagePaths = list(paths.list_images(config.ORIG_IMAGES))

# Inicializamos la cantidad de imagenes guardadas hasta ahora
# Nos permite llevar una cuenta y tener los nombres de los archivos (ej: 1.jpg)
totalPositive = 0
totalNegative = 0

# Iteramos sobre las imagenes
for (i, imagePath) in enumerate(imagePaths):
	print("[INFO] processing image {}/{}...".format(i + 1, len(imagePaths)))

	# Obtenemos el nombre del archivo y lo utilizamos para obtener el .xml correspondiente
	filename = imagePath.split(os.path.sep)[-1]
	filename = filename[:filename.rfind(".")]
	annotPath = os.path.sep.join([config.ORIG_ANNOTATIONS, "{}.xml".format(filename)])

    # Cargamos el archivo .xml e inicializamos la lista de bounding boxes reales
	contents = open(annotPath).read()
	soup = BeautifulSoup(contents, "html.parser")
	gtBoxes = []

	# Utilizando BeautifulSoup, procedemos a buscar tags en el archivo .xml
	w = int(soup.find("width").string)
	h = int(soup.find("height").string)

	# Iteramos sobre los 'object'
	for o in soup.find_all("object"):
		# Extraemos la etiqueta y bounding box correspondiente
		label = o.find("name").string
		xMin = int(o.find("xmin").string)
		yMin = int(o.find("ymin").string)
		xMax = int(o.find("xmax").string)
		yMax = int(o.find("ymax").string)

		# Truncamos cualquier bounding box que quede fuera de la imagen
		xMin = max(0, xMin)
		yMin = max(0, yMin)
		xMax = min(w, xMax)
		yMax = min(h, yMax)

		# Lo agregamos a nuestra lista de bounding boxes reales
		gtBoxes.append((xMin, yMin, xMax, yMax))

	# Cargamos la imagen
	image = cv2.imread(imagePath)




	# Utilizamos selective search para nuestros bounding box propuestos
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(image)
	ss.switchToSelectiveSearchFast()
	rects = ss.process()
	proposedRects= []

	# Iteramos sobre los rectangulos propuestos por selective search
	for (x, y, w, h) in rects:
		# convert our bounding boxes from (x, y, w, h) to (startX,
		# startY, startX, endY)
		proposedRects.append((x, y, x + w, y + h))

	# Inicializamos contadores para guardar la cantidad de RoI positivos y negativos
    # Numero de RoI para la imagen que: tienen un IoU aceptable, son guardados en el Positive Path
	positiveROIs = 0
    # Numero de Roi para la imagen que: no cumplen con IoU sobre 70%, son guardados en el Negative Path
	negativeROIs = 0

	# Iteramos sobre la cantidad maxima de regiones propuestas
	for proposedRect in proposedRects[:config.MAX_PROPOSALS]:
		# Obtenemos el bounding box propuesto
		(propStartX, propStartY, propEndX, propEndY) = proposedRect

		# Iteramos sobre los bounding box reales
		for gtBox in gtBoxes:
			# Calculamos la IoU y obtenemos el bounding box real
			iou = computeIoU(gtBox, proposedRect)
			(gtStartX, gtStartY, gtEndX, gtEndY) = gtBox

			# Inicializamos la RoI y el path de salida
			roi = None
			outputPath = None

			# Verificamos que el IoU cumpla sobre el 70% y que no hayamos alcanzado la cantidad maxima de positivos
			if iou > 0.7 and positiveROIs <= config.MAX_POSITIVE:
                # Extraemos el RoI y obtenemos el path correspondiente para el caso positivo
                # Cortamos la imagen al bounding box propuesto
				roi = image[propStartY:propEndY, propStartX:propEndX]
				filename = "{}.png".format(totalPositive)
				outputPath = os.path.sep.join([config.POSITVE_PATH, filename])

				# Actualizamos los contadores
				positiveROIs += 1
				totalPositive += 1

			# Antes de determinar si es un negativo, verificamos que no se encuentren uno encima de otro
			fullOverlap = propStartX >= gtStartX
			fullOverlap = fullOverlap and propStartY >= gtStartY
			fullOverlap = fullOverlap and propEndX <= gtEndX
			fullOverlap = fullOverlap and propEndY <= gtEndY

			# Verificamos si corresponde a un negative RoI
            # No son un fullOverlap
            # No cumple con IoU sobre 5%
            # No hemos llegado al maximo definido de negativos
			if not fullOverlap and iou < 0.05 and negativeROIs <= config.MAX_NEGATIVE:
				# Extraemos el RoI y obtenemos el path correspondiente para el caso negativo
				roi = image[propStartY:propEndY, propStartX:propEndX]
				filename = "{}.png".format(totalNegative)
				outputPath = os.path.sep.join([config.NEGATIVE_PATH, filename])

				# Actualizamos los contadores
				negativeROIs += 1
				totalNegative += 1

			if roi is not None and outputPath is not None:
				# Modificamos el tamano al que debe tener la red
                # Guardamos la imagen al disco
				roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
				cv2.imwrite(outputPath, roi)