from hezar import Tokenizer, Logger

tokenizer_to_example_repo = {
    "wordpiece": "hezarai/bert-base-fa",
    "bpe": "hezarai/roberta-base-fa",
    "sentencepiece_unigram": "hezarai/t5-base-fa",
    "whisper_bpe": "hezarai/whisper-small-fa"
}

logger = Logger(__name__)


def test_tokenizer_loading():
    tokenizers = []
    for e, repo in tokenizer_to_example_repo.items():
        tokenizer = Tokenizer.load(repo)
        tokenizers.append(tokenizer)
        logger.info(f"Successfully loaded {repo} for {e}")
    return tokenizers


def test_tokenizer_saving():
    tokenizers = test_tokenizer_loading()
    for e in tokenizers:
        e.save(e.config.name)
        logger.info(f"Successfully saved {e.config.name}")
