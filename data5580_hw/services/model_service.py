from typing import Union

from flask import Flask

from data5580_hw.models.prediction import Prediction, Model


class ModelService:
    
    @staticmethod
    def create_inference(model: Model, prediction: Prediction) -> Union[str, int, float]:
        return 10_000
    

model_service = ModelService()