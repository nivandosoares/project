#!/usr/bin/env python3
"""Gera mini clip curto (15-20s) a partir de música + banco de imagens.

Pipeline:
1) Detecta trecho de maior destaque (energia + fluxo espectral).
2) Transcreve esse trecho (faster-whisper com timestamp por palavra).
3) Divide em chunks curtos de legenda sincronizada.
4) Seleciona imagem por chunk (matching lexical em metadados/nomes).
5) Renderiza vídeo 1080x1920 com áudio original recortado.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple



@dataclass
class WordTimestamp:
    text: str
    start: float
    end: float


@dataclass
class SegmentPlan:
    start: float
    end: float
    words: List[WordTimestamp]


STOPWORDS_PT = {
    "a", "o", "e", "de", "do", "da", "em", "um", "uma", "com", "que", "pra", "para",
    "na", "no", "nos", "nas", "por", "eu", "tu", "ele", "ela", "nós", "você", "vocês",
}


def _tokenize(text: str) -> List[str]:
    import re

    tokens = re.findall(r"[\wÀ-ÿ]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS_PT and len(token) > 2]


def _assert_ffmpeg_available() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg não encontrado no PATH. Instale ffmpeg antes de executar o pipeline.")


def detect_highlight_window(audio_path: str, target_seconds: int = 18) -> Tuple[float, float]:
    """Seleciona janela com maior saliência usando energia + fluxo espectral.

    Regras:
    - target_seconds é limitado para [15, 20]
    - se o áudio for menor que o alvo, retorna áudio inteiro
    """
    import librosa
    import numpy as np

    target_seconds = max(15, min(20, int(target_seconds)))

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    duration = len(y) / sr if len(y) else 0.0
    if duration <= target_seconds:
        return 0.0, max(duration, 0.1)

    hop = 512
    frame = 2048

    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    min_len = min(len(rms), len(onset_env))
    rms = rms[:min_len]
    onset_env = onset_env[:min_len]

    rms_n = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9)
    flux_n = (onset_env - onset_env.min()) / (onset_env.max() - onset_env.min() + 1e-9)
    salience = 0.7 * rms_n + 0.3 * flux_n

    frames_per_win = max(1, int((target_seconds * sr) / hop))
    summed = np.convolve(salience, np.ones(frames_per_win, dtype=np.float32), mode="valid")
    best_frame = int(np.argmax(summed)) if len(summed) else 0

    start = best_frame * hop / sr
    end = min(duration, start + target_seconds)
    return float(start), float(end)


def transcribe_with_timestamps(
    audio_path: str,
    start: float,
    end: float,
    language: str = "pt",
    model_name: str = "small",
) -> List[WordTimestamp]:
    """Transcreve trecho usando faster-whisper (word timestamps)."""
    try:
        from faster_whisper import WhisperModel
        from pydub import AudioSegment
    except Exception as exc:
        raise RuntimeError("Dependências de transcrição não encontradas. Rode pip install -r requirements.txt") from exc

    clip = AudioSegment.from_file(audio_path)[int(start * 1000): int(end * 1000)]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_wav = Path(tmp.name)
    clip.export(str(temp_wav), format="wav")

    try:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(
            str(temp_wav),
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )

        words: List[WordTimestamp] = []
        for segment in segments:
            for word in segment.words or []:
                clean_word = word.word.strip()
                if clean_word:
                    words.append(
                        WordTimestamp(
                            text=clean_word,
                            start=float(start + word.start),
                            end=float(start + word.end),
                        )
                    )
    finally:
        temp_wav.unlink(missing_ok=True)

    if not words:
        words = [WordTimestamp(text="[instrumental]", start=start, end=end)]
    return words


def split_words_into_caption_chunks(words: Sequence[WordTimestamp], max_chunk_seconds: float = 2.8) -> List[Tuple[str, float, float]]:
    chunks: List[Tuple[str, float, float]] = []
    if not words:
        return chunks

    buffer_words = [words[0].text]
    chunk_start = words[0].start
    chunk_end = words[0].end

    for word in words[1:]:
        if (word.end - chunk_start) <= max_chunk_seconds:
            buffer_words.append(word.text)
            chunk_end = word.end
            continue

        chunks.append((" ".join(buffer_words), chunk_start, chunk_end))
        buffer_words = [word.text]
        chunk_start = word.start
        chunk_end = word.end

    chunks.append((" ".join(buffer_words), chunk_start, chunk_end))
    return chunks


def load_image_bank(image_dir: str, metadata_json: str | None = None) -> List[dict]:
    path = Path(image_dir)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Diretório de imagens inválido: {image_dir}")

    allowed_ext = {".jpg", ".jpeg", ".png", ".webp"}
    images = [file for file in path.iterdir() if file.suffix.lower() in allowed_ext]

    metadata = {}
    if metadata_json:
        metadata_path = Path(metadata_json)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    bank = []
    for image in images:
        info = metadata.get(image.name, {})
        searchable = " ".join([image.stem, info.get("caption", ""), *info.get("tags", [])]).strip()
        bank.append({"path": str(image), "searchable": searchable})

    if not bank:
        raise ValueError("Banco de imagens vazio. Adicione pelo menos 1 arquivo de imagem.")
    return bank


def score_image_for_text(text: str, image_searchable: str) -> float:
    lyric_tokens = set(_tokenize(text))
    image_tokens = set(_tokenize(image_searchable))
    if not lyric_tokens or not image_tokens:
        return 0.0

    overlap = len(lyric_tokens & image_tokens)
    jaccard = overlap / max(1, len(lyric_tokens | image_tokens))
    return overlap + jaccard


def choose_images_for_chunks(chunks: Sequence[Tuple[str, float, float]], image_bank: Sequence[dict]) -> List[str]:
    chosen: List[str] = []
    previous = None

    for text, _, _ in chunks:
        ranked = sorted(image_bank, key=lambda item: score_image_for_text(text, item["searchable"]), reverse=True)
        candidate = ranked[0]["path"]
        if candidate == previous and len(ranked) > 1:
            candidate = ranked[1]["path"]
        chosen.append(candidate)
        previous = candidate

    return chosen


def render_video(
    audio_path: str,
    plan: SegmentPlan,
    chunks: Sequence[Tuple[str, float, float]],
    chosen_images: Sequence[str],
    output_path: str,
) -> None:
    from moviepy.editor import AudioFileClip, CompositeVideoClip, ImageClip, TextClip, concatenate_videoclips

    width, height = 1080, 1920
    visual_clips = []

    for index, (text, c_start, c_end) in enumerate(chunks):
        duration = max(0.2, c_end - c_start)

        image = (
            ImageClip(chosen_images[index])
            .set_duration(duration)
            .resize(height=height)
            .on_color(size=(width, height), color=(0, 0, 0), pos=("center", "center"))
        )

        subtitle = (
            TextClip(
                text,
                fontsize=56,
                color="white",
                stroke_color="black",
                stroke_width=2,
                method="caption",
                size=(int(width * 0.9), None),
                align="center",
                font="DejaVu-Sans",
            )
            .set_position(("center", int(height * 0.78)))
            .set_duration(duration)
        )

        visual_clips.append(CompositeVideoClip([image, subtitle]).set_duration(duration))

    video = concatenate_videoclips(visual_clips, method="compose")
    audio = AudioFileClip(audio_path).subclip(plan.start, plan.end)
    final = video.set_audio(audio)
    final.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac", preset="medium")


def build_clip(
    song_path: str,
    image_dir: str,
    output_path: str,
    metadata_json: str | None = None,
    language: str = "pt",
    target_seconds: int = 18,
    whisper_model: str = "small",
) -> None:
    _assert_ffmpeg_available()

    start, end = detect_highlight_window(song_path, target_seconds=target_seconds)
    words = transcribe_with_timestamps(song_path, start, end, language=language, model_name=whisper_model)

    plan = SegmentPlan(start=start, end=end, words=words)
    chunks = split_words_into_caption_chunks(plan.words)
    image_bank = load_image_bank(image_dir, metadata_json)
    chosen_images = choose_images_for_chunks(chunks, image_bank)

    render_video(song_path, plan, chunks, chosen_images, output_path)

    print(f"✅ Mini clip gerado: {output_path}")
    print(f"Janela escolhida: {start:.2f}s -> {end:.2f}s (duração: {end - start:.2f}s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protótipo de mini clip automático com legenda sincronizada")
    parser.add_argument("--song", required=True, help="Arquivo de música/vídeo de entrada")
    parser.add_argument("--images", required=True, help="Diretório de imagens")
    parser.add_argument("--output", default="mini_clip.mp4", help="Arquivo MP4 de saída")
    parser.add_argument("--metadata", default=None, help="JSON opcional com tags/caption por imagem")
    parser.add_argument("--language", default="pt", help="Idioma da transcrição")
    parser.add_argument("--target-seconds", type=int, default=18, help="Duração alvo (forçada para 15..20)")
    parser.add_argument("--whisper-model", default="small", help="Modelo Whisper: tiny/base/small/medium")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build_clip(
        song_path=arguments.song,
        image_dir=arguments.images,
        output_path=arguments.output,
        metadata_json=arguments.metadata,
        language=arguments.language,
        target_seconds=arguments.target_seconds,
        whisper_model=arguments.whisper_model,
    )
