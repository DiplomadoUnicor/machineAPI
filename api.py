from fastapi import FastAPI
from typing import List
from classApi import ModelInput, ModelOutput, APIModelBackEnd

# Creamos el objeto app
app = FastAPI(title="API de ML del Diplomado", version="1.1.0")


@app.post("/predict", response_model=List[ModelOutput])
async def predict_(inputs: List[ModelInput]):
    response = list()
    # Iteramos por todas las entradas que damos
    for Input in inputs:
        model = APIModelBackEnd(
            Input.DEPARTAMENTO, Input.MUNICIPIO, Input.ARMAS_MEDIOS, Input.AÑO,Input.MES, Input.DIA, Input.GENERO, Input.GRUPO_ETARIO
            #,  Input.DIA, Input.GENERO, Input.GRUPO_ETARIO, Input.ARMAS_MEDIOS, Input.MUNICIPIO, Input.DEPARTAMENTO
        )
        response.append(model.predict()[0])
    # Retorna  la lista con todas las predicciones hechas.
    return response