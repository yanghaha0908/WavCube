# Copyright 2025 Xiaomi Corporation.
import os

from .model import MiMoAudioTokenizer
from .utils import (load_audio, unpacking, padding, mel_spectrogram)  # noqa

# TODO: add sha256s check for the model files
_MODELS = {
    "v1": "",
}

_SHA256S = {
    "v1": "",
}


def load_model(model_path: str, ) -> MiMoAudioTokenizer:
    """
    Load a MiMoAudioTokenizer model

    Parameters
    ----------
    model_path: str
        path to the MiMoAudioTokenizer model files.

    Returns
    -------
    model : MiMoAudioTokenizer
        The MiMoAudioTokenizer model instance
    """

    assert model_path is not None
    assert os.path.isdir(model_path)

    model = MiMoAudioTokenizer.from_pretrained(model_path, process=False)

    return model
