from typing import Union, List

import torch

from ..model import Model


class Text2TextModel(Model):
    def preprocess(self, inputs: Union[str, List[str]], **kwargs):
        if isinstance(inputs, str):
            inputs = [inputs]
        if "text_normalizer" in self.preprocessor:
            normalizer = self.preprocessor["text_normalizer"]
            inputs = normalizer(inputs)
        tokenizer = self.preprocessor[self.tokenizer_name]
        inputs = tokenizer(inputs, return_tensors="pt", device=self.device)
        return inputs

    def post_process(self, inputs, **kwargs):
        records = []
        tokenizer = self.preprocessor["sentencepiece_unigram_tokenizer"]
        for output_ids in inputs["output_ids"][0]:
            if isinstance(output_ids, torch.Tensor):
                output_ids = output_ids.numpy().tolist()
            record = {
                "output_text": tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                )
            }
            records.append(record)
        return records
