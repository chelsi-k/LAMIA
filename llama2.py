import assemblyai as aai
from elevenlabs.client import ElevenLabs 
from elevenlabs import stream
import ollama
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AI_Assistant:
    def __init__(self):
        aai.settings.api_key = "cbc7b2b1ef7f46439d17ab8fcaca05d1"
        self.client = ElevenLabs(api_key="cbc7b2b1ef7f46439d17ab8fcaca05d1")
        self.transcriber = None
        self.full_transcript = [
            {"role": "system", "content": "You are a LLM called llama3 created by meta, answer the question "}
        ]
    
    def start_transcription(self):
        logging.info("Starting real-time transcription")
        try:
            self.transcriber = aai.RealtimeTranscriber(
                sample_rate=16_000,
                on_data=self.on_data,
                on_error=self.on_error,
                on_open=self.on_open,
                on_close=self.on_close,
            )

            self.transcriber.connect()

            microphone_stream = aai.extras.MicrophoneStream(sample_rate=16_000)
            self.transcriber.stream(microphone_stream)
        except Exception as e:
            logging.error(f"Error starting transcription: {e}")

    def close_transcription(self):
        if self.transcriber:
            self.transcriber.close()
            self.transcriber = None
            logging.info("Transcription closed")

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        logging.info(f"Session opened with ID: {session_opened.session_id}")

    def on_data(self, transcript: aai.RealtimeTranscript):
        if not transcript.text:
            return

        if isinstance(transcript, aai.RealtimeFinalTranscript):
            logging.info(f"Final transcript received: {transcript.text}")
            self.generate_ai_response(transcript)
        else:
            logging.info(f"Interim transcript received: {transcript.text}")

    def on_error(self, error: aai.RealtimeError):
        logging.error(f"An error occurred: {error}")

    def on_close(self):
        logging.info("Session closed")
    
    def generate_ai_response(self, transcript):
        self.close_transcription()
        
        self.full_transcript.append({"role": "user", "content": transcript.text})
        
        logging.info(f"User: {transcript.text}")
       
        try:
            ollama_stream = ollama.chat(
                model="llama3",
                messages=self.full_transcript,
                stream=True,
            )
            
            logging.info("Llama 3 response stream started")
            
            text_buffer = ""
            full_text = ""
            for chunk in ollama_stream:
                text_buffer += chunk['messages']['content']
                if text_buffer.endswith('.'):
                    audio_stream = self.client.generate(text=text_buffer, model="eleven_turbo_v2", stream=True)
                    logging.info(f"Llama 3: {text_buffer}")
                    stream(audio_stream)
                    full_text += text_buffer
                    text_buffer = ""
                    
            if text_buffer:
                audio_stream = self.client.generate(text=text_buffer, model="eleven_turbo_v2", stream=True)
                logging.info(f"Llama 3: {text_buffer}")
                stream(audio_stream)
                full_text += text_buffer

            self.full_transcript.append({"role": "assistant", "content": full_text})
        except Exception as e:
            logging.error(f"Error generating AI response: {e}")

        self.start_transcription()

ai_assistant = AI_Assistant()
ai_assistant.start_transcription()
