import abc
import os.path
import subprocess
import torchaudio
from enum import Enum


class Extension(Enum):
    wav = "wav"
    flac = "flac"
    mp3 = "mp3"


class AudioInfo:
    def __init__(self, audio_file_path, transcript, speaker_id, ext: Extension):
        self.audio_file_path = audio_file_path
        self.transcript = transcript
        self.speaker_id = speaker_id
        self.ext = ext


"""
final class before being passed as an arg to the format writer which writes the audio and transcript in the required format
"""


class ModifiedAudioAndTranscript:
    def __init__(self, sig, sr, transcript, speaker_id, file_name_without_ext, ext: Extension):
        self.sig = sig
        self.sr = sr
        self.transcript = transcript
        self.speaker_id = speaker_id
        self.file_name_without_ext = file_name_without_ext
        self.ext = ext


class BaseDataFactory(object, metaclass=abc.ABCMeta):

    def __init__(self, write_dir, base_dir, formats=None):
        if formats is None:
            formats = ["ljspeech", "vctk"]
        self.base_dir = base_dir
        self.audio_infos = []
        self.writers = {}
        self.formats = formats
        for format in formats:
            format_writer: BaseFormatWriter
            if format == "ljspeech":
                format_writer = LJSpeechFormatWriter(write_dir)
            elif format == "vctk":
                format_writer = VCTKFormatWriter(write_dir)
            self.writers[format + "_" + write_dir] = format_writer

    """
        checks the basic passing requirements of the audio file and the transcript,
        modifies the transcript and the audio and writes in specified dir in required formats
    """

    @abc.abstractmethod
    def accept_modify_write_audio(self):
        pass

    @abc.abstractmethod
    def write_new_audio_dataset(self):
        pass

    def write_audio_and_transcript_in_formats(self, modified_audio_transcript: ModifiedAudioAndTranscript):
        for format_writer_key in self.writers:
            self.writers[format_writer_key].write_files_in_format(modified_audio_transcript)


class BaseFormatWriter(abc.ABC):
    def __init__(self, write_dir):
        self.write_dir = write_dir

    @abc.abstractmethod
    def write_files_in_format(self, modified_audio_and_transcript: ModifiedAudioAndTranscript):
        pass


"""
vctk dataset format
data_dir:
    txt:
        speaker_1:
            file1.txt
            file2.txt
        speaker_2:
            file3.txt
            file4.txt
    wav48:
        speaker_1:
            file1_mic1.flac
            file2_mic1.flac
        speaker_2:
            file3_mic1.flac
            file4_mic1.flac
"""


class VCTKFormatWriter(BaseFormatWriter):

    def __init__(self, write_dir):
        super().__init__(write_dir)
        self.vctk_audio_file_prefix = "_mic1"

    def write_files_in_format(self, modified_audio_and_transcript: ModifiedAudioAndTranscript):

        speaker_id = modified_audio_and_transcript.speaker_id
        actual_write_dir = os.path.join(self.write_dir, "vctk")
        if (not os.path.isdir(os.path.join(actual_write_dir, "wav48_silence_trimmed"))) or (
                not os.path.isdir(os.path.join(actual_write_dir, "txt"))):
            process = subprocess.Popen(["mkdir", os.path.join(actual_write_dir, "txt")])
            process.wait(timeout=10)
            process = subprocess.Popen(["mkdir", os.path.join(actual_write_dir, "wav48_silence_trimmed")])
            process.wait(timeout=10)
        txt_dir = os.path.join(actual_write_dir, "txt")
        wav_dir = os.path.join(actual_write_dir, "wav48_silence_trimmed")
        if (not os.path.exists(os.path.join(txt_dir, speaker_id))) or (
                not os.path.exists(os.path.join(wav_dir, speaker_id))):
            process = subprocess.Popen(["mkdir", os.path.join(wav_dir, speaker_id)])
            process.wait(timeout=10)
            process = subprocess.Popen(["mkdir", os.path.join(txt_dir, speaker_id)])
            process.wait(timeout=10)
        transcript_file_to_write = os.path.join(os.path.join(txt_dir, speaker_id),
                                                modified_audio_and_transcript.file_name_without_ext + ".txt")
        audio_file_to_write = os.path.join(os.path.join(wav_dir, speaker_id),
                                           modified_audio_and_transcript.file_name_without_ext + self.vctk_audio_file_prefix + "." + modified_audio_and_transcript.ext.value)
        with open(transcript_file_to_write, 'w', encoding='utf-8') as transcript_file:
            transcript_file.write(modified_audio_and_transcript.transcript)
        torchaudio.save(audio_file_to_write, modified_audio_and_transcript.sig, modified_audio_and_transcript.sr,
                        format=modified_audio_and_transcript.ext.value)


"""
data_dir:
    wavs:
        file1.wav
        file2.wav
    metadata.csv(file):
        file1|trasncript
        file2|transcript
"""


class LJSpeechFormatWriter(BaseFormatWriter):

    def __init__(self, write_dir):
        super().__init__(write_dir)

    def write_files_in_format(self, modified_audio_and_transcript: ModifiedAudioAndTranscript):
        speaker_id = modified_audio_and_transcript.speaker_id
        actual_write_dir = os.path.join(self.write_dir, "ljspeech")
        if (not os.path.isdir(os.path.join(actual_write_dir, "wavs"))):
            process = subprocess.Popen(["mkdir", os.path.join(self.write_dir, "wavs")])
            process.wait(timeout=10)

        csv_file_to_write = os.path.join(actual_write_dir, "metadata.csv")
        wav_dir = os.path.join(actual_write_dir, "wavs")
        audio_file_to_write = os.path.join(wav_dir,
                                           modified_audio_and_transcript.file_name_without_ext + "." + modified_audio_and_transcript.ext.value)

        with open(csv_file_to_write, 'a') as csv_file:
            added_row = modified_audio_and_transcript.file_name_without_ext + "|" + modified_audio_and_transcript.transcript + "\n"
            csv_file.write(added_row)
        torchaudio.save(audio_file_to_write, modified_audio_and_transcript.sig, modified_audio_and_transcript.sr,
                        format=modified_audio_and_transcript.ext.value)
