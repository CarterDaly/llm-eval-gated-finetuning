from typing import Union, List

from flask import Flask

from data5580_hw.models.prediction import Prediction, Model, Explanation, Explanations


class ExplainerService:
    
    @staticmethod
    def create_explanation(model: Model, prediction: Prediction) -> Explanations:
        explanation = model._explainer.predict(prediction.get_pandas_frame_of_inputs())

        explanations = Explanations()

        for name, values, in explanations:
            explanations.explanations.append(
                Explanation(
                    name=name,
                    values=values
                )
            )

        return explanations
    

explainer_service = ExplainerService()