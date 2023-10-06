from efficient_word.streams import SimpleMicStream
from efficient_word.engine import HotwordDetector
from efficient_word.audio_processing import Resnet50_Arc_loss
from datetime import datetime, timedelta
import sounddevice as sd
import soundfile as sf
import numpy as np
import speech_recognition as sr


class VoiceRecognition:
    def __init__(self, sample_rate=44100, channels=1, device=None):
        self.output_file = "audio.wav"
        self.duration = 5
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.base_model = Resnet50_Arc_loss()
        self.ruby = HotwordDetector(
            hotword="ruby",
            model=self.base_model,
            reference_file="ruby_ref.json",
            threshold=0.7,
            relaxation_time=2
        )
        self.mic_stream = SimpleMicStream(
            window_length_secs=1.5,
            sliding_window_secs=0.1,
        )

    def record_audio(self):
        silence_threshold = 0.005

        sd.default.samplerate = self.sample_rate
        sd.default.channels = self.channels
        sd.default.device = self.device

        recording = True
        peak_detected = False
        start_time = datetime.now()
        collected_frames = []

        def callback(indata, frames, time, status):
            nonlocal recording, peak_detected

            if np.max(indata) > 0.1:
                peak_detected = True

            elapsed_time = datetime.now() - start_time
            if elapsed_time >= timedelta(seconds=self.duration) or peak_detected is True:
                if np.max(indata) > silence_threshold:
                    recording = True
                else:
                    recording = False

            if status or elapsed_time >= timedelta(seconds=45):
                recording = False

            collected_frames.append(indata.copy())

        with sd.InputStream(callback=callback):
            sd.sleep(self.duration * 100)

            while recording:
                sd.sleep(100)

        audio_data = np.concatenate(collected_frames)

        sf.write(self.output_file, audio_data, samplerate=self.sample_rate)

    def recognize_speech(self):
        r = sr.Recognizer()
        with sr.AudioFile(self.output_file) as source:
            audio_data = r.record(source)

        try:
            text = r.recognize_google(audio_data)
            print("Recognized speech:", text)
        except sr.UnknownValueError:
            print("Could not understand speech")
        except sr.RequestError as e:
            print("Recognition request failed:", str(e))

    def start_recognition(self):
        self.mic_stream.start_stream()
        print("Say Ruby")
        while True:
            frame = self.mic_stream.getFrame()
            result = self.ruby.scoreFrame(frame)
            if result is None:
                continue
            if result["match"]:
                self.mic_stream.close_stream()
                print("Wakeword uttered", result["confidence"])
                self.record_audio()
                self.recognize_speech()
                self.start_recognition()


vr = VoiceRecognition()
vr.start_recognition()
