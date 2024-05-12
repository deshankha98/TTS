import abc
import os
import random
import re

import torch
import torchaudio
from tts_outputs.Word import Word

from tts_execution.TTSUtils import OS_PATH_DELIMITER, add_random_uniform_noise, add_silence_with_noise, \
    speedup_with_noise


class BaseTTSExecutor(object, metaclass=abc.ABCMeta):
    model_vars = {}
    def __init__(self, clone_speakers_dir: str, language: str, output_dir, keywords=[], random_words=[]):
        self.clone_speakers_dir = clone_speakers_dir
        self.keywords = keywords
        self.random_words = random_words
        self.language = language
        self.output_dir = output_dir

    @staticmethod
    @abc.abstractmethod
    def create_dataset(tts_executors, word: Word, speaker: str, is_keyword):
        pass

    @staticmethod
    def create_dir(output_dir, word: Word, is_keyword):
        if is_keyword:
            # if not (
            #         os.path.isdir(os.path.join(os.path.join(output_dir, "keywords"), word.dir_suffix))):
            #     os.mkdir(os.path.join(os.path.join(output_dir, "keywords"), word.dir_suffix))
            if not (
                    os.path.isdir(os.path.join(os.path.join(output_dir, "augmented_keywords"), word.dir_suffix))):
                os.mkdir(os.path.join(os.path.join(output_dir, "augmented_keywords"), word.dir_suffix))
        else:
            # if not (
            #         os.path.isdir(
            #             os.path.join(os.path.join(output_dir, "random_words"), word.dir_suffix))):
            #     os.mkdir(os.path.join(os.path.join(output_dir, "random_words"), word.dir_suffix))
            if not (
                    os.path.isdir(
                        os.path.join(os.path.join(output_dir, "augmented_random_words"), word.dir_suffix))):
                os.mkdir(os.path.join(os.path.join(output_dir, "augmented_random_words"), word.dir_suffix))

        # try:
        #     lock_.acquire()
        #     if is_keyword:
        #         if not (
        #         os.path.isdir(os.path.join(os.path.join(tts_executor.output_dir, "keywords"), word.dir_suffix))):
        #             os.mkdir(os.path.join(os.path.join(tts_executor.output_dir, "keywords"), word.dir_suffix))
        #             dir_path = os.path.join(os.path.join(tts_executor.output_dir, "keywords"), word.dir_suffix)
        #     else:
        #         if not (
        #         os.path.isdir(os.path.join(os.path.join(tts_executor.output_dir, "random_words"), word.dir_suffix))):
        #             os.mkdir(os.path.join(os.path.join(tts_executor.output_dir, "random_words"), word.dir_suffix))
        #             dir_path = os.path.join(os.path.join(tts_executor.output_dir, "random_words"), word.dir_suffix)
        # except Exception as e:
        #     exc = e
        # finally:
        #     lock_.release()
        #     if exc is not None:
        #         print("Exception occured while performing dir creation for word", word, "exc:", exc)
        #         raise RuntimeError("dir creation error")

    @staticmethod
    def find_random_voice_files_for_keyword(word: Word, voice_dir, common_list, is_keyword=True, augment_ratio=-1, augment_no=-1):
        regex = re.compile('.*\.wav')
        if is_keyword:
            dir_path = os.path.join(voice_dir, "keywords")
        else:
            dir_path = os.path.join(voice_dir, "random_words")

        dir_path = os.path.join(dir_path, word.dir_suffix)
        dir_path_ = dir_path + OS_PATH_DELIMITER
        all_files = os.listdir(dir_path_)
        filelist = []
        for file in all_files:
            if regex.match(file):
                filelist.append(file)

        if augment_no > 0 or augment_ratio > 0:
            if augment_no > 0:
                sampled_audio_files = random.sample(filelist, augment_no)
            elif augment_ratio > 0:
                augment_no = int(len(filelist) * augment_ratio)
                sampled_audio_files = random.sample(filelist, augment_no)
        else:
            sampled_audio_files = filelist

        for file in sampled_audio_files:
            common_list.append((os.path.join(dir_path, file), word.dir_suffix, is_keyword))


    @staticmethod
    def augment_voice_data(input_file_name, dir_suffix, base_output_dir, is_keyword=True, augmentation_types=["noise_uniform"]):
        if is_keyword:
            output_dir = os.path.join(os.path.join(base_output_dir, "augmented_keywords"), dir_suffix)
        else:
            output_dir = os.path.join(os.path.join(base_output_dir, "augmented_random_words"), dir_suffix)

        sig, sr_actual = torchaudio.load(input_file_name, normalize=True, format="wav")
        sig = torch.mean(sig, dim=0, keepdim=True)

        if "noise_uniform" in augmentation_types:
            sig = add_random_uniform_noise(sig, sr_actual)
            base_file_name = os.path.basename(input_file_name)
            text = base_file_name.split("__")[0]
            speaker_name = base_file_name.split("__")[1].split("_")[0]
            cardinality = base_file_name.split("__")[1].split("_")[1]
            new_speaker_name = speaker_name + "|" + "noiseAugmentationSnr5db"
            output_file_name = os.path.join(output_dir, text + "__" + new_speaker_name + "_" + cardinality)
        if "noise_random" in augmentation_types:
            sig = add_random_uniform_noise(sig, sr_actual)
            base_file_name = os.path.basename(input_file_name)
            text = base_file_name.split("__")[0]
            speaker_name = base_file_name.split("__")[1].split("_")[0]
            cardinality = base_file_name.split("__")[1].split("_")[1]
            new_speaker_name = speaker_name + "|" + "normalNoiseAugmentationSnr5db"
            output_file_name = os.path.join(output_dir, text + "__" + new_speaker_name + "_" + cardinality)
        if "add_silence_with_noise" in augmentation_types:
            sig = add_silence_with_noise(sig, sr_actual)
            base_file_name = os.path.basename(input_file_name)
            text = base_file_name.split("__")[0]
            speaker_name = base_file_name.split("__")[1].split("_")[0]
            cardinality = base_file_name.split("__")[1].split("_")[1]
            new_speaker_name = speaker_name + "|" + "silenceAugmentationSnr5db"
            output_file_name = os.path.join(output_dir, text + "__" + new_speaker_name + "_" + cardinality)
        if "speedup_with_noise" in augmentation_types:
            sig = speedup_with_noise(input_file_name, snr=5)
            base_file_name = os.path.basename(input_file_name)
            text = base_file_name.split("__")[0]
            speaker_name = base_file_name.split("__")[1].split("_")[0]
            cardinality = base_file_name.split("__")[1].split("_")[1]
            new_speaker_name = speaker_name + "|" + "speedupAugmentationSnr5db"
            output_file_name = os.path.join(output_dir, text + "__" + new_speaker_name + "_" + cardinality)
        torchaudio.save(output_file_name, sig, sr_actual)




