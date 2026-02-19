#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from clip_prototype import MEDIA_EXTENSIONS, _infer_tags_from_filename, process_clip

st.set_page_config(page_title="Mini Clip Builder", layout="wide")
st.title("🎬 Mini Clip Builder (localhost)")
st.caption("Teste local em: http://localhost:8501")

st.write(
    "Upload da música + banco de mídia (imagem/gif/vídeo), ajuste as tags e gere o mini clip com legenda sincronizada."
)

song = st.file_uploader("1) Música", type=["mp3", "wav", "m4a", "mp4", "ogg"])
media_files = st.file_uploader(
    "2) Banco de mídia (imagens, gifs, vídeos)",
    type=["jpg", "jpeg", "png", "webp", "gif", "mp4", "mov", "webm", "mkv"],
    accept_multiple_files=True,
)

col1, col2, col3 = st.columns(3)
language = col1.selectbox("Idioma", ["pt", "en", "es"], index=0)
whisper_model = col2.selectbox("Modelo Whisper", ["tiny", "base", "small", "medium"], index=2)
target_seconds = col3.slider("Duração alvo", min_value=15, max_value=20, value=18)

if media_files:
    st.subheader("3) Tags da mídia")
    st.caption("Tags iniciais são inferidas pelo nome do arquivo. Você pode editar antes de processar.")

    for idx, media in enumerate(media_files):
        extension = Path(media.name).suffix.lower()
        if extension not in MEDIA_EXTENSIONS:
            continue
        default_tags = ", ".join(_infer_tags_from_filename(Path(media.name)))
        st.text_input(f"Tags para {media.name}", value=default_tags, key=f"tags_{idx}")

if st.button("🚀 Processar e gerar mini clip", type="primary"):
    if not song:
        st.error("Envie uma música primeiro.")
        st.stop()
    if not media_files:
        st.error("Envie ao menos 1 item de mídia.")
        st.stop()

    with st.spinner("Processando... isso pode levar alguns minutos"):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            song_path = base_path / song.name
            song_path.write_bytes(song.read())

            media_dir = base_path / "media"
            media_dir.mkdir(parents=True, exist_ok=True)

            metadata = {}
            for idx, media in enumerate(media_files):
                media_path = media_dir / media.name
                media_path.write_bytes(media.read())
                raw_tags = st.session_state.get(f"tags_{idx}", "")
                tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
                metadata[media.name] = {"caption": "", "tags": tags}

            metadata_path = base_path / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

            output_path = base_path / "mini_clip.mp4"
            result = process_clip(
                song_path=str(song_path),
                media_dir=str(media_dir),
                output_path=str(output_path),
                metadata_json=str(metadata_path),
                language=language,
                target_seconds=target_seconds,
                whisper_model=whisper_model,
            )

            output_bytes = output_path.read_bytes()

    st.success("Mini clip gerado com sucesso!")
    st.video(output_bytes)
    st.download_button("⬇️ Baixar mini clip", data=output_bytes, file_name="mini_clip.mp4", mime="video/mp4")

    st.subheader("Resumo")
    st.write(f"Trecho detectado: {result['start']:.2f}s → {result['end']:.2f}s ({result['duration']:.2f}s)")
    for index, (caption, _, _) in enumerate(result["chunks"]):
        media_name = Path(result["selected_media"][index]["path"]).name
        st.write(f"**Legenda {index + 1}:** {caption}")
        st.caption(f"Mídia escolhida: {media_name}")
