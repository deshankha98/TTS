from finetuning_dataset_creation.BaseDataFactory import BaseDataFactory, AudioInfo, ModifiedAudioAndTranscript, Extension
import os
import csv
import torchaudio
import torch
import re

sr_required = 22050


class EnglishDataFactory(BaseDataFactory):

    def __init__(self, write_dir, base_dir, formats):
        super(EnglishDataFactory, self).__init__(write_dir=write_dir, base_dir=base_dir, formats=formats)

    """
    checks the basic passing requirements of the audio file and the transcript,
    modifies the transcript and the audio and writes in specified dir in required formats
    """

    def accept_modify_write_audio(self):

        audio_infos = self.audio_infos
        for audio_info in audio_infos:
            base_file_name = os.path.basename(audio_info.audio_file_path)
            base_file_name_without_ext = base_file_name.split(".")[0]
            acceptable_transcript = True
            transcript = audio_info.transcript
            modified_transcript = ""
            for char in transcript:
                modified_char = self.acceptable_and_modify_transcript(char)
                if modified_char == None:
                    acceptable_transcript = False
                    break
                else:
                    modified_transcript += modified_char.lower()
            if not acceptable_transcript:
                continue
            ### sample audio in sampling rate
            ext = audio_info.ext.value
            sig, sr_actual = torchaudio.load(audio_info.audio_file_path, normalize=True, format=ext)
            sig = torch.mean(sig, dim=0, keepdim=True)
            ### trim silence - difficult to calibrate as it was trimming the whole audio
            # sig = torchaudio.functional.vad(sig, sr_actual)
            # sig = torchaudio.functional.vad(sig.flip(dims=[1]), sr_actual)
            # sig = sig.flip(dims=[1])
            ### perform resampling
            if sr_actual != sr_required:
                sig = torchaudio.functional.resample(sig, sr_actual, sr_required)

            ### check duration
            audio_duration = len(sig[0]) / sr_required
            if audio_duration > 10:
                continue
            self.write_audio_and_transcript_in_formats(
                ModifiedAudioAndTranscript(sig, sr_required, modified_transcript, audio_info.speaker_id,
                                           base_file_name_without_ext, Extension.flac))


    def write_new_audio_dataset(self):
        files = os.listdir(self.base_dir)
        for file in files:
            if file == ".DS_Store":
                continue
            if not file.__contains__("tacotronDDC"):
                continue
            transcript = file.split("__")[0] + "."
            ext = Extension(file.split(".")[1])
            speaker = file.split("__")[1].split(".")[0].split("_")[0]
            audio_info = AudioInfo(os.path.join(self.base_dir, file), transcript, speaker, ext)
            self.audio_infos.append(audio_info)



        self.accept_modify_write_audio()

    def acceptable_and_modify_transcript(self, char):
        ## if hindi character pass
        if ((ord(u'\u0041') <= ord(char) <= ord(u'\u005A')) or (
                ord(u'\u0061') <= ord(char) <= ord(u'\u007A')) or char == " "):
            return char
        elif (char == "\""):
            return ""
        ## if in [, ! ? '] then pass and return as it is. These symbols used for punctuation
        elif (char == "," or char == "!" or char == "?" or char == "'" or char == "."):
            return char
        return None
