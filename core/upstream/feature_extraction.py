import logging

from abc import ABC, abstractmethod

import nemo.collections.asr as nemo_asr
import torch
import torch.nn.functional as F
import whisper

from speechbrain.inference.classifiers import EncoderClassifier
from torch import nn
from transformers import Wav2Vec2Model

logging.getLogger("nemo_logger").setLevel(logging.ERROR)


class BaseFeatureExtractor(nn.Module, ABC):
    def __init__(self, model_id):
        super().__init__()

        self.model_id = model_id

    @abstractmethod
    def load(self, cache_dir):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError


def _load_xlsr(
    model_id,
    finetuned=True,
    cache_dir=None,
    file="/home/common/speech_phylo/models/xlsr_300m_voxlingua107_ft.pt",
):
    """
    Load the Wav2Vec2Model for speech feature extraction.

    Parameters
    ----------
    model_id : str
        Model ID (see model_ids)
    finetuned : bool, optional
        Specifies whether to load a finetuned model, by default True.
    cache_dir : str, optional
        Path to the folder where cached files are stored, by default None
    file : str, optional
        Path to fine-tuned checkpoint
        (from https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/xlsr/README.md)

    Returns
    -------
    model_ : Wav2Vec2Model
        The loaded Wav2Vec2Model for speech feature extraction.
    """
    model_ = Wav2Vec2Model.from_pretrained(model_id, cache_dir=cache_dir)

    model_.freeze_feature_encoder()

    if finetuned:
        state_dict = torch.load(file)

        tmp = {
            k.replace("w2v_encoder.w2v_model.", "")
            .replace("mask_emb", "masked_spec_embed")
            .replace(".0.weight", ".conv.weight")
            .replace(".0.bias", ".conv.bias")
            .replace("fc1", "feed_forward.intermediate_dense")
            .replace("fc2", "feed_forward.output_dense")
            .replace("self_attn", "attention")
            .replace("attention_layer_norm", "layer_norm")
            .replace(".2.1", ".layer_norm")
            .replace("post_extract_proj", "feature_projection.projection")
            .replace("pos_conv", "pos_conv_embed")
            .replace("embed.conv.weight", "embed.conv.parametrizations.weight")
            .replace("weight_g", "weight.original0")
            .replace("weight_v", "weight.original1"): v
            for (k, v) in state_dict["model"].items()
        }

        tmp["feature_projection.layer_norm.bias"] = tmp.pop("layer_norm.bias")
        tmp["feature_projection.layer_norm.weight"] = tmp.pop("layer_norm.weight")

        missing_keys, unexpected_keys = model_.load_state_dict(tmp, strict=False)

        print(f"missing keys: {missing_keys}\n" f"unexpec keys: {unexpected_keys}")

    return model_


class HuggingfaceFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        model_id,
        finetuned=True,
        cache_dir=None,
    ):
        super().__init__(model_id)

        if cache_dir is not None:
            cache_dir = f"{cache_dir}/huggingface"

        self.finetuned = finetuned

        self.load(cache_dir)

    def load(self, cache_dir=None):
        if "xls-r" in self.model_id:
            feature_extractor = _load_xlsr(
                self.model_id, finetuned=self.finetuned, cache_dir=cache_dir
            )

            feature_extractor.freeze_feature_encoder()
            emb_dim = 1024
        elif "mms" in self.model_id:
            feature_extractor = Wav2Vec2Model.from_pretrained(
                self.model_id, cache_dir=cache_dir
            )

            feature_extractor.freeze_feature_encoder()
            emb_dim = 1280
        else:
            raise ValueError(f"Unknown model ID: {self.model_id}")

        self.feature_extractor = feature_extractor
        self.emb_dim = emb_dim

    def forward(self, x):
        return self.feature_extractor(x).last_hidden_state


class NeMoFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, model_id, cache_dir=None):
        super().__init__(model_id)

        if cache_dir is not None:
            cache_dir = f"{cache_dir}/NeMo"
        else:
            cache_dir = "~/.cache/NeMo"

        self.load(cache_dir)

    def load(self, cache_dir):
        if self.model_id == "NeMo_ambernet":
            model_name = self.model_id.split("_")[-1]

            feature_extractor = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(
                restore_path=f"{cache_dir}/{model_name}.nemo"
            )

            feature_extractor.freeze()

            emb_dim = feature_extractor.decoder.final.in_features
        else:
            raise ValueError(f"Unknown model ID: {self.model_id}")

        self.feature_extractor = feature_extractor
        self.emb_dim = emb_dim

    def forward(self, x):
        # Input shape: B x T
        _, emb = self.feature_extractor(
            input_signal=x,
            input_signal_length=torch.tensor([x.shape[1]], device=x.device),
        )

        # Output shape: B x D

        return emb


class SpeechbrainFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        model_id,
        cache_dir=None,
        device="cpu",
    ):
        super().__init__(model_id)

        if cache_dir is not None:
            cache_dir = f"{cache_dir}/huggingface"

        self.load(cache_dir, device)

        self.load_audio = self.feature_extractor.load_audio

    def load(self, cache_dir=None, device="cpu"):
        if self.model_id == "speechbrain/lang-id-voxlingua107-ecapa":
            self.feature_extractor = EncoderClassifier.from_hparams(
                source=self.model_id, savedir=cache_dir, run_opts={"device": device}
            )
            self.emb_dim = 256
        else:
            raise ValueError(f"Unknown model ID: {self.model_id}")

    def forward(self, x):
        return self.feature_extractor.encode_batch(x)


class WhisperFeatureExtractor(BaseFeatureExtractor):
    def __init__(
        self,
        model_id,
        cache_dir=None,
    ):
        super().__init__(model_id)

        if cache_dir is not None:
            cache_dir = f"{cache_dir}/whisper"

        self.load(cache_dir)

        self.tokenizer = whisper.tokenizer.get_tokenizer(
            self.feature_extractor.is_multilingual,
            num_languages=self.feature_extractor.num_languages,
        )

    def load(self, cache_dir=None):
        model_size = self.model_id.split("_")[-1]

        if "turbo" in model_size:
            self.n_mels = 128
        else:
            self.n_mels = 80

        self.feature_extractor = whisper.load_model(model_size, download_root=cache_dir)

        self.emb_dim = self.feature_extractor.decoder.ln.weight.shape[0]

    def lms(self, x):
        """Log-mel-spectrogram

        Parameters
        ----------
        x : torch.Tensor
            Input of size (B, T)
            B: batch size
            T: sample length

        Returns
        -------
        x : torch.Tensor
            Output of size (B, n_mels, n_frames)
        """
        # Pad so that input is valid for whisper models
        x = whisper.pad_or_trim(x)

        # B x T --> B x M x N
        x = whisper.log_mel_spectrogram(x, n_mels=self.n_mels)

        return x

    def encode(self, x):
        """
        Forward method of whisper.AudioEncoder with addition of hidden_states

        Adapted from https://github.com/openai/whisper/blob/cdb81479623391f0651f4f9175ad986e85777f31/whisper/model.py#L188
        """
        encoder = self.feature_extractor.encoder

        x = F.gelu(encoder.conv1(x))
        x = F.gelu(encoder.conv2(x))
        x = x.permute(0, 2, 1)

        assert (
            x.shape[1:] == encoder.positional_embedding.shape
        ), "incorrect audio shape"
        x = (x + encoder.positional_embedding).to(x.dtype)

        hidden_state_list = []

        for block in encoder.blocks:
            x = block(x)
            hidden_state_list.append(x)

        x = encoder.ln_post(x)

        # B x n_layers x n_chunks x emb_dim
        hidden_states = torch.stack(hidden_state_list, dim=1)

        return x, hidden_states

    def decode(self, x, kv_cache=None):
        """
        Forward method of whisper.TextDecoder with addition of hidden_states

        Adapted from https://github.com/openai/whisper/blob/cdb81479623391f0651f4f9175ad986e85777f31/whisper/model.py#L227
        """
        n_audio = x.shape[0]

        xt = torch.tensor([[self.tokenizer.sot]] * n_audio).to(x.device)

        decoder = self.feature_extractor.decoder

        # From whisper.TextDecoder.forward
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        xt = (
            decoder.token_embedding(xt)
            + decoder.positional_embedding[offset : offset + xt.shape[-1]]
        )
        xt = xt.to(x.dtype)

        hidden_state_list = [xt]

        for block in decoder.blocks:
            xt = block(xt, x, mask=decoder.mask, kv_cache=kv_cache)
            hidden_state_list.append(xt)

        xt = decoder.ln(xt)

        # B x n_layers x 1 x emb_dim
        hidden_states = torch.stack(hidden_state_list, dim=1)

        return xt, hidden_states

    def forward(self, x, kv_cache=None):
        x = self.lms(x)

        x, _ = self.encode(x)

        xt, _ = self.decode(x, kv_cache=kv_cache)

        return xt


def load_feature_extractor(
    model_id, cache_dir=None, device="cpu", finetuned=True
) -> BaseFeatureExtractor:
    """Load a feature extractor (to be used downstream)

    Parameters
    ----------
    model_id : str
        Model ID (see model_ids)
    cache_dir : str, optional
        Path to the folder where cached files are stored, by default None
    device : str, optional
        Device on which a torch.Tensor is or will be allocated, by default "cpu"
    finetuned : bool, optional
        Whether to use a fine-tuned XLS-R or not, by default True

    Returns
    -------
    BaseFeatureExtractor (nn.Module)
        A feature extraction model with a forward method
    """
    if any(_ in model_id for _ in ["xls-r", "mms"]):
        feature_extractor = HuggingfaceFeatureExtractor(
            model_id, finetuned=finetuned, cache_dir=cache_dir
        )
    elif "speechbrain" in model_id:
        feature_extractor = SpeechbrainFeatureExtractor(
            model_id, cache_dir=cache_dir, device=device
        )
    elif "NeMo" in model_id:
        feature_extractor = NeMoFeatureExtractor(model_id, cache_dir=cache_dir)
    elif "whisper" in model_id:
        feature_extractor = WhisperFeatureExtractor(model_id, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown model name: {model_id}")

    feature_extractor.eval()

    feature_extractor.to(device)

    return feature_extractor
