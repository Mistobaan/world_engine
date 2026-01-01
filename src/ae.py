import torch
from torch import Tensor


"""
WARNING:
- Always assumes scale=1, shift=0
"""


class InferenceAE:
    def __init__(self, ae_model, device=None, dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.ae_model = ae_model.eval().to(device=device, dtype=dtype)

    @classmethod
    def from_pretrained(cls, model_uri: str, **kwargs):
        import pathlib

        import huggingface_hub
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        from owl_vaes import get_model_cls

        base = pathlib.Path(huggingface_hub.snapshot_download(model_uri))

        model = torch.nn.Module()
        for name in ("encoder", "decoder"):
            cfg = OmegaConf.load(base / f"{name}_conf.yml")
            sd = load_file(base / f"{name}.safetensors", device="cpu")
            mod = getattr(get_model_cls(cfg.model.model_id)(cfg.model), name)
            mod.load_state_dict(sd, strict=True)
            setattr(model, name, mod)

        return cls(model, **kwargs)

    def encode(self, img: Tensor):
        """RGB -> RGB+D -> latent"""
        assert img.dim() == 3, "Expected [H, W, C] image tensor"
        img = img.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        rgb = img.permute(0, 3, 1, 2).contiguous().div(255).mul(2).sub(1)
        return self.ae_model.encoder(rgb)

    @torch.compile
    @torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    def decode(self, latent: Tensor):
        decoded = self.ae_model.decoder(latent)
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = (decoded * 255).round().to(torch.uint8)
        return decoded.squeeze(0).permute(1, 2, 0)[..., :3]
