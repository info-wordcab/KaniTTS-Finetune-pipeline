#!/usr/bin/env python3
"""Generate cloned voice samples using a reference utterance and finetuned KaniTTS checkpoints.

This script loads a finetuned KaniTTS model, encodes a short reference audio clip with the
NeMo nano codec, and uses that prompt plus the sentences defined in `config/eval_set.yaml`
to synthesise audio in the reference voice.

Example:
    python voice_clone.py \
        --reference-audio /path/to/ref.wav \
        --transcript "This is what the speaker says in the reference clip." \
        --checkpoint-dir /scratch2/kani_tts/au_500k/checkpoints \
        --model-id au-lora64
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torchaudio
from nemo.collections.tts.models import AudioCodecModel
from scipy.io import wavfile
from transformers import AutoModelForCausalLM, AutoTokenizer

from config_loader import config_loader


def _resolve_checkpoint(args_model_id: str | None, checkpoints_dir: Path) -> tuple[str, bool]:
    """Return (identifier, is_local) for model selection."""

    def is_remote(model_id: str) -> bool:
        return "/" in model_id or model_id.startswith("hf://")

    experiments_cfg = config_loader.get_experiments_config()
    if args_model_id:
        model_id = args_model_id
        if is_remote(model_id):
            return model_id, False
    else:
        if not experiments_cfg.experiments:
            raise ValueError("No experiments defined in config/experiments.yaml; provide --model-id explicitly.")
        model_id = experiments_cfg.experiments[0].base.model_id

    candidate = (checkpoints_dir / model_id).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    return str(candidate), True


def _prepare_reference_audio(
    audio_path: Path,
    target_sample_rate: int,
    max_duration_sec: float,
) -> torch.Tensor:
    """Load, resample and trim the reference audio."""
    waveform, sample_rate = torchaudio.load(str(audio_path))

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)

    max_samples = int(target_sample_rate * max_duration_sec)
    if waveform.size(-1) > max_samples:
        waveform = waveform[..., :max_samples]

    return waveform.to(dtype=torch.float32)


def _encode_reference_audio(
    waveform: torch.Tensor,
    codec_model: AudioCodecModel,
    tokens_cfg,
) -> List[int]:
    """Encode reference waveform into flattened audio tokens for the prompt."""
    device = next(codec_model.parameters()).device
    waveform = waveform.to(device)
    audio_len = torch.tensor([waveform.shape[-1]], dtype=torch.long, device=device)

    with torch.inference_mode():
        encoded_tokens, _ = codec_model.encode(audio=waveform, audio_len=audio_len)

    # encoded_tokens: [batch, num_codebooks, seq_len]
    codes = encoded_tokens[0].to(torch.long)  # [num_codebooks, seq_len]
    num_codebooks = codes.shape[0]
    offsets = torch.arange(num_codebooks, device=codes.device) * tokens_cfg.codec.codebook_size
    codes = codes + offsets.unsqueeze(-1)
    codes = codes + tokens_cfg.codec.audio_tokens_start
    flattened = codes.transpose(0, 1).reshape(-1).tolist()
    return flattened


def _decode_audio_tokens(
    audio_tokens: Iterable[int],
    codec_model: AudioCodecModel,
    tokens_cfg,
) -> np.ndarray:
    """Convert generated audio token ids back to waveform."""
    tokens = np.fromiter(audio_tokens, dtype=np.int64)
    num_codebooks = tokens_cfg.codec.num_codebooks
    if tokens.size == 0:
        raise ValueError("No audio tokens to decode.")

    usable = (tokens.size // num_codebooks) * num_codebooks
    if usable != tokens.size:
        tokens = tokens[:usable]

    tokens = torch.from_numpy(tokens)
    tokens = tokens.view(-1, num_codebooks)

    offsets = torch.arange(num_codebooks) * tokens_cfg.codec.codebook_size
    tokens = tokens - (tokens_cfg.codec.audio_tokens_start + offsets)

    if (tokens < 0).any():
        raise ValueError("Generated tokens contain values below audio_tokens_start; cannot decode.")

    tokens = tokens.transpose(0, 1).unsqueeze(0).to(next(codec_model.parameters()).device)
    tokens_len = torch.tensor([tokens.size(-1)], dtype=torch.long, device=tokens.device)

    with torch.inference_mode():
        waveform, _ = codec_model.decode(tokens=tokens, tokens_len=tokens_len)

    return waveform.squeeze(0).cpu().numpy()


def build_prompt(
    tokenizer: AutoTokenizer,
    transcript: str,
    target_text: str,
    voice_tokens: List[int],
    tokens_cfg,
    double_sos: bool = False,
) -> List[int]:
    """Create the prompt token sequence for voice cloning."""
    transcript = transcript.strip()
    target_text = target_text.strip()
    combined_text = f"{transcript} {target_text}".strip()

    combined_ids = tokenizer.encode(combined_text, add_special_tokens=False)

    prompt_ids = [
        tokens_cfg.special_tokens.start_of_human,
        *combined_ids,
        tokens_cfg.special_tokens.end_of_text,
        tokens_cfg.special_tokens.end_of_human,
        tokens_cfg.special_tokens.start_of_ai,
        tokens_cfg.special_tokens.start_of_speech,
    ]

    if double_sos:
        prompt_ids.append(tokens_cfg.special_tokens.start_of_speech)

    prompt_ids.extend(voice_tokens)
    return prompt_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone a voice using a finetuned KaniTTS checkpoint.")
    parser.add_argument("--reference-audio", type=Path, required=True, help="Path to the reference audio file (wav/flac/etc).")
    parser.add_argument("--transcript", type=str, required=True, help="Exact transcript of the reference audio clip.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory containing finetuned checkpoints (defaults to eval_config).")
    parser.add_argument("--model-id", type=str, default=None, help="Checkpoint subdirectory name; defaults to first experiment model_id.")
    parser.add_argument("--output-dir", type=Path, default=Path("voice_clone_outputs"), help="Directory to write generated wav files.")
    parser.add_argument("--max-reference-seconds", type=float, default=15.0, help="Trim reference audio to this duration (seconds).")
    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Override nucleus sampling top-p.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Override repetition penalty.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max_new_tokens for generation.")
    parser.add_argument("--min-new-tokens", type=int, default=None, help="Force a minimum number of generated tokens before stopping.")
    parser.add_argument("--strip-speaker-prefix", action="store_true", help="Drop everything before the first ':' in each eval_set sentence.")
    parser.add_argument("--double-sos", action="store_true", help="Insert an extra start_of_speech token after the reference audio cue.")
    parser.add_argument("--token-log-count", type=int, default=0, help="If >0, print the first N generated token ids for debugging.")
    args = parser.parse_args()

    reference_audio = args.reference_audio.expanduser().resolve()
    if not reference_audio.exists():
        raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

    eval_cfg = config_loader.get_eval_config()
    checkpoints_root = (
        args.checkpoint_dir.expanduser().resolve()
        if args.checkpoint_dir
        else Path(eval_cfg.paths.checkpoints_dir).expanduser().resolve()
    )

    model_identifier, is_local = _resolve_checkpoint(args.model_id, checkpoints_root)

    inference_cfg = config_loader.get_inference_config()
    model_cfg = config_loader.get_model_config()
    eval_set_cfg = config_loader.get_eval_set()

    dtype_str = inference_cfg.model.torch_dtype
    torch_dtype = getattr(torch, dtype_str)

    load_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": inference_cfg.model.device_map,
        "trust_remote_code": True,
    }
    tokenizer_kwargs = {"trust_remote_code": True}
    if is_local:
        load_kwargs["local_files_only"] = True
        tokenizer_kwargs["local_files_only"] = True
        model_id = model_identifier
    else:
        model_id = model_identifier

    tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    codec_model = AudioCodecModel.from_pretrained(model_cfg.codec.model_name).eval()
    codec_model.to(model.device)

    save_prefix = Path(model_identifier).name if is_local else model_identifier.replace('/', '__')

    waveform = _prepare_reference_audio(reference_audio, model_cfg.codec.sample_rate, args.max_reference_seconds)
    voice_tokens = _encode_reference_audio(waveform, codec_model, model_cfg)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_kwargs = dict(
        do_sample=inference_cfg.generation.do_sample,
        temperature=args.temperature if args.temperature is not None else inference_cfg.generation.temperature,
        top_p=args.top_p if args.top_p is not None else inference_cfg.generation.top_p,
        repetition_penalty=args.repetition_penalty if args.repetition_penalty is not None else inference_cfg.generation.repetition_penalty,
        max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else inference_cfg.generation.max_new_tokens,
        num_return_sequences=inference_cfg.generation.num_return_sequences,
        eos_token_id=model_cfg.special_tokens.end_of_speech,
        pad_token_id=model_cfg.special_tokens.pad_token,
        use_cache=True,
    )

    if args.min_new_tokens is not None:
        generation_kwargs["min_new_tokens"] = args.min_new_tokens

    prompts = eval_set_cfg.eval_set
    if not prompts:
        raise ValueError("Evaluation set is empty; add prompts to config/eval_set.yaml")

    for entry in prompts:
        sample_id, text = next(iter(entry.items()))
        if args.strip_speaker_prefix and ":" in text:
            text = text.split(":", 1)[1].lstrip()

        prompt_ids = build_prompt(
            tokenizer,
            args.transcript,
            text,
            voice_tokens,
            model_cfg,
            double_sos=args.double_sos,
        )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)

        with torch.no_grad():
            outputs = model.generate(input_ids, **generation_kwargs)

        generated_ids = outputs[0].tolist()
        prompt_len = len(prompt_ids)
        speech_tokens = generated_ids[prompt_len:]

        if args.token_log_count > 0:
            preview = speech_tokens[: args.token_log_count]
            print(f"[{sample_id}] first {len(preview)} generated tokens: {preview}")

        if model_cfg.special_tokens.end_of_speech in speech_tokens:
            eos_index = speech_tokens.index(model_cfg.special_tokens.end_of_speech)
            speech_tokens = speech_tokens[:eos_index]

        audio_tokens = [tok for tok in speech_tokens if tok >= model_cfg.codec.audio_tokens_start]
        if not audio_tokens:
            raise RuntimeError(f"No audio tokens generated for {sample_id}")

        audio_np = _decode_audio_tokens(audio_tokens, codec_model, model_cfg)
        output_path = output_dir / f"{save_prefix}_{sample_id}.wav"
        wavfile.write(output_path, model_cfg.codec.sample_rate, audio_np)
        print(f"✓ Wrote {output_path}")


if __name__ == "__main__":
    main()
