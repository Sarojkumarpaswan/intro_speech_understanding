import gtts
import speech_recognition as sr
import librosa
import soundfile as sf
import os


def synthesize(text, lang, filename):
    '''
    Use gtts.gTTS(text=text, lang=lang) to synthesize speech,
    then write it to filename (MP3).
    '''
    tts = gtts.gTTS(text=text, lang=lang)
    tts.save(filename)


def make_a_corpus(texts, languages, filenames):
    '''
    Create many speech files, convert MP3 -> WAV,
    then recognize using SpeechRecognition.
    '''
    recognized_texts = []
    recognizer = sr.Recognizer()

    for text, lang, rootname in zip(texts, languages, filenames):

        mp3_file = rootname + ".mp3"
        wav_file = rootname + ".wav"

        # 1️⃣ Synthesize MP3
        synthesize(text, lang, mp3_file)

        # 2️⃣ Convert MP3 to WAV
        audio, sr_rate = librosa.load(mp3_file, sr=None)
        sf.write(wav_file, audio, sr_rate)

        # 3️⃣ Recognize speech
        with sr.AudioFile(wav_file) as source:
            audio_data = recognizer.record(source)

        try:
            recognized = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            recognized = ""
        except sr.RequestError:
            recognized = ""

        recognized_texts.append(recognized)

    return recognized_texts
