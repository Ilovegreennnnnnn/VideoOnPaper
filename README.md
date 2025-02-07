# Convertisseur Vidéo vers Texte

Ce projet permet de convertir des vidéos en texte en utilisant la reconnaissance vocale (Whisper) et de générer des résumés automatiques.

## Prérequis

- Python 3.8 ou supérieur
- FFmpeg installé sur votre système
- GPU NVIDIA (recommandé mais non obligatoire)

## Installation

1. Cloner le repository :
```bash
git clone [votre-repo]
cd [votre-repo]
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Placez votre fichier vidéo dans le dossier du projet

2. Modifiez la variable `video_path` dans le fichier `video_to_text.py`

3. Exécutez le script :
```bash
python video_to_text.py
```

Les fichiers de sortie seront générés dans le dossier `outputs/` :
- `[nom_video]_transcript.txt` : Transcription complète
- `[nom_video]_summary.txt` : Résumé généré

## Structure du projet

```
.
├── models/         # Dossier pour les modèles téléchargés
├── temp_audio/     # Dossier temporaire pour l'extraction audio
├── outputs/        # Dossier des résultats
├── requirements.txt
├── README.md
└── video_to_text.py
```

## Notes importantes

- Le premier lancement peut prendre du temps car les modèles seront téléchargés
- La taille des modèles peut varier de 1 Go à 10 Go selon la configuration
- L'utilisation d'un GPU accélère considérablement le traitement

## Personnalisation

Vous pouvez modifier la taille du modèle Whisper dans `video_to_text.py` :
- "tiny" : Le plus rapide, moins précis
- "base" : Bon compromis
- "medium" : Recommandé (par défaut)
- "large" : Le plus précis, plus lent
