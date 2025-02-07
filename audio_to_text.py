import os
import time
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path
from tqdm import tqdm

class AudioToTextConverter:
    def __init__(self, model_size="medium"):
        self.model_size = model_size
        self.whisper_model = None
        self.summarizer = None
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.setup_models()

    def download_model(self, model_name):
        """Télécharge le modèle localement s'il n'existe pas déjà"""
        local_model_path = os.path.join(self.models_dir, model_name.split('/')[-1])
        
        if not os.path.exists(local_model_path):
            print(f"Téléchargement du modèle {model_name}...")
            print(f"Le modèle sera sauvegardé dans : {local_model_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            print("Sauvegarde du modèle localement...")
            model.save_pretrained(local_model_path)
            tokenizer.save_pretrained(local_model_path)
            print("Modèle sauvegardé avec succès!")
            
        return local_model_path

    def setup_models(self):
        """Initialise les modèles Whisper et Mistral"""
        print("Chargement des modèles...")
        self.whisper_model = whisper.load_model(self.model_size)
        
        print("Configuration du modèle Mistral pour le résumé...")
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        local_model_path = self.download_model(model_name)
        
        print("Chargement du modèle Mistral depuis le stockage local...")
        self.summarizer = pipeline(
            "text-generation",
            model=local_model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        print("Modèles chargés avec succès!")

    def generate_summary_prompt(self, text):
        """Génère un prompt pour Mistral"""
        return f"""<s>[INST] Voici un texte à résumer. Fais un résumé concis et informatif en français :

{text}

[/INST]"""

    def split_text(self, text, max_length=1000):
        """Découpe le texte en morceaux plus petits"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_length += len(word) + 1  # +1 pour l'espace
            if current_length > max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def summarize_text(self, text):
        """Génère un résumé du texte avec Mistral"""
        print("\nGénération du résumé...")
        start_time = time.time()
        
        # Découper le texte en morceaux plus petits
        chunks = self.split_text(text)
        print(f"Le texte a été divisé en {len(chunks)} parties pour le résumé")
        
        # Résumer chaque morceau
        summaries = []
        for i, chunk in enumerate(chunks, 1):
            print(f"\nRésumé de la partie {i}/{len(chunks)}...")
            try:
                prompt = self.generate_summary_prompt(chunk)
                response = self.summarizer(
                    prompt,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    num_return_sequences=1,
                )[0]['generated_text']
                
                # Extraire le résumé de la réponse (après le prompt)
                summary = response.split("[/INST]")[-1].strip()
                summaries.append(summary)
            except Exception as e:
                print(f"Erreur lors du résumé de la partie {i}: {str(e)}")
                continue
        
        # Combiner les résumés
        final_summary = " ".join(summaries)
        
        # Si le résumé combiné est encore trop long, le résumer une dernière fois
        if len(final_summary.split()) > 200:
            try:
                prompt = self.generate_summary_prompt(final_summary)
                response = self.summarizer(
                    prompt,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    num_return_sequences=1,
                )[0]['generated_text']
                final_summary = response.split("[/INST]")[-1].strip()
            except Exception as e:
                print(f"Erreur lors de la génération du résumé final: {str(e)}")
        
        duration = time.time() - start_time
        print(f"\nRésumé généré en {duration:.1f} secondes")
        
        return final_summary

    def transcribe_audio(self, audio_path):
        """Transcrit l'audio en texte"""
        print("Transcription de l'audio...")
        print("Cette opération peut prendre plusieurs minutes. Veuillez patienter...")
        
        start_time = time.time()
        
        result = self.whisper_model.transcribe(audio_path)
        
        duration = time.time() - start_time
        print(f"\nTranscription terminée en {duration:.1f} secondes")
        
        return result["text"]

    def process_audio(self, audio_path, output_dir="outputs"):
        """Traite un fichier audio"""
        try:
            # Création du dossier de sortie
            os.makedirs(output_dir, exist_ok=True)
            
            # Transcription
            transcript = self.transcribe_audio(audio_path)
            
            # Génération du résumé
            summary = self.summarize_text(transcript)
            
            # Sauvegarde des résultats
            audio_name = Path(audio_path).stem
            output_base = os.path.join(output_dir, audio_name)
            
            with open(f"{output_base}_transcript.txt", "w", encoding="utf-8") as f:
                f.write(transcript)
            
            with open(f"{output_base}_summary.txt", "w", encoding="utf-8") as f:
                f.write(summary)
                
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Convertir un fichier audio en texte et générer un résumé')
    parser.add_argument('audio_path', help='Chemin vers le fichier audio à traiter')
    parser.add_argument('--model', default='medium', choices=['tiny', 'base', 'small', 'medium', 'large'],
                      help='Taille du modèle Whisper à utiliser (default: medium)')
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_path):
        print(f"Erreur : Le fichier audio '{args.audio_path}' n'existe pas!")
        return

    try:
        converter = AudioToTextConverter(model_size=args.model)
        print(f"\nTraitement du fichier audio : {args.audio_path}")
        print(f"Modèle Whisper utilisé : {args.model}")
        
        results = converter.process_audio(args.audio_path)
        
        print("\nTraitement terminé avec succès!")
        print(f"Transcription sauvegardée dans: {results['transcript_file']}")
        print(f"Résumé sauvegardé dans: {results['summary_file']}")
        
        # Afficher un extrait du résumé
        print("\nExtrait du résumé :")
        print("-" * 50)
        summary = results['summary'][:500] + "..." if len(results['summary']) > 500 else results['summary']
        print(summary)
        print("-" * 50)
        
    except Exception as e:
        print(f"\nUne erreur est survenue lors du traitement : {str(e)}")

if __name__ == "__main__":
    main()
