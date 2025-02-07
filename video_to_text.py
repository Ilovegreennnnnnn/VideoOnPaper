import os
import whisper
from moviepy.editor import VideoFileClip
from transformers import pipeline
from pathlib import Path

class VideoToTextConverter:
    def __init__(self, model_size="medium"):
        self.model_size = model_size
        self.whisper_model = None
        self.summarizer = None
        self.setup_models()

    def setup_models(self):
        """Initialise les modèles Whisper et le summarizer"""
        print("Chargement des modèles...")
        self.whisper_model = whisper.load_model(self.model_size)
        self.summarizer = pipeline("summarization", 
                                 model="philschmid/bart-large-cnn-samsum",
                                 max_length=150)
        print("Modèles chargés avec succès!")

    def extract_audio(self, video_path):
        """Extrait l'audio d'une vidéo"""
        print("Extraction de l'audio...")
        temp_audio_path = os.path.join("temp_audio", "temp_audio.wav")
        
        # Assure que le dossier temp existe
        os.makedirs("temp_audio", exist_ok=True)
        
        # Extraction audio
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(temp_audio_path)
        video.close()
        
        return temp_audio_path

    def transcribe_audio(self, audio_path):
        """Transcrit l'audio en texte"""
        print("Transcription de l'audio...")
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]

    def summarize_text(self, text):
        """Génère un résumé du texte"""
        print("Génération du résumé...")
        summary = self.summarizer(text)
        return summary[0]['summary_text']

    def process_video(self, video_path, output_dir="outputs"):
        """Traite une vidéo complète"""
        try:
            # Création du dossier de sortie
            os.makedirs(output_dir, exist_ok=True)
            
            # Extraction de l'audio
            audio_path = self.extract_audio(video_path)
            
            # Transcription
            transcript = self.transcribe_audio(audio_path)
            
            # Génération du résumé
            summary = self.summarize_text(transcript)
            
            # Sauvegarde des résultats
            video_name = Path(video_path).stem
            output_base = os.path.join(output_dir, video_name)
            
            with open(f"{output_base}_transcript.txt", "w", encoding="utf-8") as f:
                f.write(transcript)
            
            with open(f"{output_base}_summary.txt", "w", encoding="utf-8") as f:
                f.write(summary)
            
            # Nettoyage
            if os.path.exists(audio_path):
                os.remove(audio_path)
                
            return {
                "transcript": transcript,
                "summary": summary,
                "transcript_file": f"{output_base}_transcript.txt",
                "summary_file": f"{output_base}_summary.txt"
            }
            
        except Exception as e:
            print(f"Une erreur est survenue: {str(e)}")
            raise

def main():
    # Exemple d'utilisation
    converter = VideoToTextConverter(model_size="medium")
    
    # Remplacer par le chemin de votre vidéo
    video_path = "chemin/vers/votre/video.mp4"
    
    if os.path.exists(video_path):
        results = converter.process_video(video_path)
        print("\nTraitement terminé!")
        print(f"Transcription sauvegardée dans: {results['transcript_file']}")
        print(f"Résumé sauvegardé dans: {results['summary_file']}")
    else:
        print("Fichier vidéo non trouvé!")

if __name__ == "__main__":
    main()
