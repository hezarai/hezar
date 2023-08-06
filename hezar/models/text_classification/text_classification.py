"""
A Base class for text classification models built using HuggingFace Transformers
"""
from typing import Dict, Union, List

from ...models import Model


class TextClassificationModel(Model):

    def preprocess(self, inputs: Union[str, List[str]], **kwargs):
        if isinstance(inputs, str):
            inputs = [inputs]
        if "text_normalizer" in self.preprocessor:
            normalizer = self.preprocessor["text_normalizer"]
            inputs = normalizer(inputs)
        tokenizer = self.preprocessor[self.tokenizer_name]
        inputs = tokenizer(inputs, return_tensors="pt", device=self.device)
        return inputs

    def post_process(self, inputs, **kwargs) -> Dict:
        return_all_scores = kwargs.get("return_all_scores", False)
        logits = inputs["logits"]
        if return_all_scores:
            predictions = logits
            predictions_probs = logits.softmax(1)
            outputs = []
            for sample_index in range(predictions.shape[0]):
                sample_outputs = []
                for label_index, score in enumerate(predictions_probs[sample_index]):
                    label = self.config.id2label[label_index]
                    sample_outputs.append({"label": label, "score": score.item()})
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
