"""
A BERT model for text classification built using HuggingFace Transformers
"""
from typing import Dict

from ...models import Model


class TextClassificationModel(Model):

    def post_process(self, inputs, **kwargs) -> Dict:
        return_all_scores = kwargs.get("return_all_scores", False)
        logits = inputs["logits"]
        if return_all_scores:
            predictions = logits
            predictions_probs = logits.softmax(1)
            outputs = []
            for sample_index in range(predictions.shape[0]):
                sample_outputs = []
                for prediction, prob in zip(predictions[sample_index], predictions_probs[sample_index]):
                    label = self.config.id2label[prediction.item()]
                    sample_outputs.append({"label": label, "score": prob.item()})
                outputs.append(sample_outputs)
        else:
            predictions = logits.argmax(1)
            predictions_probs = logits.softmax(1).max(1)
            outputs = {"labels": [], "probs": []}
            for prediction, prob in zip(predictions, predictions_probs):
                label = self.config.id2label[prediction.item()]
                outputs["labels"].append(label)
                outputs["probs"].append(prob.item())
        return outputs
