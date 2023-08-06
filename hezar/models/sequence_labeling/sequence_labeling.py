"""
A Base class for sequence labeling models
"""
from typing import Dict, Union, List

from ...models import Model


class SequenceLabelingModel(Model):
    """
    Base class for all sequence labeling models (Also a Model subclass)
    """
    def preprocess(self, inputs: Union[str, List[str]], **kwargs):
        if isinstance(inputs, str):
            inputs = [inputs]
        if "text_normalizer" in self.preprocessor:
            normalizer = self.preprocessor["text_normalizer"]
            inputs = normalizer(inputs)
        tokenizer = self.preprocessor[self.tokenizer_name]
        inputs = tokenizer(
            inputs,
            return_word_ids=True,
            return_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            device=self.device,
        )
        return inputs

    def post_process(self, inputs, **kwargs):
        # TODO sequence labeling outputs should consider rejoining split words into single words with proper tag
        logits = inputs["logits"]
        tokens = inputs["tokens"]
        word_ids = inputs["word_ids"]  # noqa
        predictions = logits.argmax(2).cpu()
        predictions = [[self.config.id2label[p.item()] for p in prediction] for prediction in predictions]
        outputs = []
        for tokens_list, prediction in zip(tokens, predictions):
            results = []
            for token, tag in zip(tokens_list, prediction):
                if token not in self.config.prediction_skip_tokens:
                    results.append({"token": token, "tag": tag})
            outputs.append(results)
        return outputs
