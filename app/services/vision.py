# app/services/vision.py
import numpy as np

modelo_tatuagem = None

def carregar_modelo_tatuagem():
    global modelo_tatuagem
    if modelo_tatuagem is None:
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import GlobalAveragePooling2D
        
        base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base.output)
        modelo_tatuagem = Model(inputs=base.input, outputs=x)
    return modelo_tatuagem

def extrair_vetor_tatuagem(caminho_imagem: str):
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.applications.efficientnet import preprocess_input

    model = carregar_modelo_tatuagem()
    img = keras_image.load_img(caminho_imagem, target_size=(224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    embedding = model.predict(x, verbose=0)[0]
    
    norma = np.linalg.norm(embedding)
    vetor_normalizado = embedding / norma if norma > 0 else embedding
    return vetor_normalizado.tolist()