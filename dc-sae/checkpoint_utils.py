"""Strict inference loading, excluding the training-only perceptual loss network."""
import torch


def load_sae_checkpoint(model, ckpt_path, device=None, verbose=True):
    payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = payload.get('model', payload.get('state_dict', payload))
    state = {key.removeprefix('module.'): value for key, value in state.items()}
    state = {key: value for key, value in state.items() if not key.startswith('_lpips_loss_fn.')}
    model.load_state_dict(state, strict=True)
    step = payload.get('step', 0)
    if verbose:
        print(f'Loaded SAE checkpoint {ckpt_path} (step={step}, strict model weights)')
    return step
