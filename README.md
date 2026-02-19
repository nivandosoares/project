# Mini Clip Builder (CLI + Frontend Web)

Você pode testar localmente no navegador em **http://localhost:8501**.

## O que faz

1. Recebe música.
2. Detecta trecho de destaque (15–20s).
3. Transcreve com timestamps.
4. Faz matching de contexto da letra com banco de mídia (imagem/gif/vídeo).
5. Gera mini clip vertical com legenda sincronizada.

## Requisitos

- Python 3.10+
- ffmpeg no sistema

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Teste no browser (localhost)

```bash
./run_local.sh
```

ou

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Depois abra: `http://localhost:8501`

## Fluxo no frontend

- Upload de música
- Upload de banco de mídia (`jpg`, `png`, `webp`, `gif`, `mp4`, `mov`, `webm`, `mkv`)
- Edição de tags sugeridas automaticamente por nome de arquivo
- Processamento e preview do resultado
- Download do mini clip

## Uso CLI

```bash
python clip_prototype.py \
  --song ./musica.mp3 \
  --media ./assets/media \
  --metadata ./metadata.json \
  --output ./mini_clip.mp4 \
  --language pt \
  --target-seconds 18 \
  --whisper-model small
```
