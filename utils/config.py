import os

# Definimos el directorio base del cual obtendremos tanto imagenes como anotaciones
ORIG_BASE_PATH = "raccoons"
ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "images"])
ORIG_ANNOTATIONS = os.path.sep.join([ORIG_BASE_PATH, "annotations"])

# Definimos el directorio para el nuevo dataset por crear mediante scripts
# Definimos las clases que tendra este, en este caso raccoon y no_raccoon
BASE_PATH = "dataset"
POSITVE_PATH = os.path.sep.join([BASE_PATH, "raccoon"])
NEGATIVE_PATH = os.path.sep.join([BASE_PATH, "no_raccoon"])

# Definimos el numero maximo de propuestas cuando usamos selective search
# Primero para cuando utilicemos el conjunto de training
# Segundo para cuando estemos realizando inferencia
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

# Definimos la cantidad maxima de regiones positivas y negativas cuando creemos el dataset
MAX_POSITIVE = 30
MAX_NEGATIVE = 10

# Definimos las dimensiones de la red a utilizar
# (MobileNet, preentrenada en ImageNet)
INPUT_DIMS = (224, 224)

# Path donde guardaremos el modelo y label encoder
MODEL_PATH = "Model/raccoon_detector.h5"
ENCODER_PATH = "Model/label_encoder.pickle"

# Definimos la probabilidad minima requerida para una deteccion positiva (evita falsos positivos)
MIN_PROBA = 0.99