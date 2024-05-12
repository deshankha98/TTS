import os
import random
from multiprocessing import Pool, Manager

from TTS.api import TTS
from tts_outputs.english.Keywords import EnglishKeyWords

from tts_execution.BaseTTSExecutor import BaseTTSExecutor
from tts_execution.TTSUtils import *
from tts_outputs.Word import Word

"""
freevc24 is a model inspired from vits architecture owing to vits' waveform reconstruction capability along with its ability
to extract and disentangle the speaker characteristics from the audio content features. 
freevc24 will convert the content of the source_wav to target_wav i.e. the resultant audio will have the prosody, emotion, phonemes of 
the target_wav
"""

LANGUAGE_VS_MODEL_PATHS = {
    "en_US": {
        "model_path": "/Users/shankhajyoti.de/Library/ApplicationSupport/tts/tts_models--en--vctk--vits/model_file.pth",
        "config_path": "/Users/shankhajyoti.de/Library/ApplicationSupport/tts/tts_models--en--vctk--vits/config.json"
    }
}

class VitsTTSExecutor(BaseTTSExecutor):

    def __init__(self, clone_speakers_dir: str, audio_dir: str, language: str, output_dir, keywords=[],
                 random_words=[]):
        super(VitsTTSExecutor, self).__init__(clone_speakers_dir, language, output_dir, keywords, random_words)
        if (LANGUAGE_VS_MODEL_PATHS.get(language) == None):
            raise RuntimeError("Model not available for language")
        self.audio_dir = audio_dir
        with torch.no_grad():
            self.tts = TTS(
                config_path=LANGUAGE_VS_MODEL_PATHS.get(self.language).get("config_path"),
                progress_bar=False, gpu=False,
                model_path=LANGUAGE_VS_MODEL_PATHS.get(self.language).get("model_path"))
        self.speakers = self.tts.speakers
        BaseTTSExecutor.model_vars["vits"] = (self.tts, self.output_dir, self.speakers)

    @staticmethod
    def create_dataset(tts_executor, word: Word, speaker, is_keyword):
        if tts_executor.clone_speakers_dir is not None:  ## don't use vits for cloning purposes as freevc24 gives better results
            return
        suffix = "keywords" if is_keyword else "random_words"
        dir_path = os.path.join(os.path.join(tts_executor.output_dir, suffix), word.dir_suffix)
        punctuated_word = word.text + "."
        complete_speaker_name = speaker + "Vits"  ### text + __ + speaker + _ + cardinality

        ### transform text, speaker to remove spaces and replace them with | 'pipe' symbol
        transformed_text = space2pipe(word.text)
        transformed_speaker_name = space2pipe(complete_speaker_name)
        file_path = os.path.join(dir_path, transformed_text + "__" + transformed_speaker_name + "_" + "1") + ".wav"
        tts_executor.tts.tts_to_file(punctuated_word, file_path=file_path, speaker=speaker)


def custom_error_callback(error):
    print(f'Got error: {error}')

def get_selected_speakers(speakers, n):
    return random.sample(speakers, n) if (0 < n < len(speakers)) else speakers


if __name__ == "__main__":
    ############# for keyword and random words generation #####################
    # with Manager() as manager:
    #     keywords = [keyword for keyword in EnglishKeyWords]
    #     random_words = [random_word for random_word in EnglishRandomWords]
    #     executor = VitsTTSExecutor(clone_speakers_dir=None, audio_dir=None, language="en_US",
    #                                    output_dir="/Users/shankhajyoti.de/PythonProjects/MultiSpeakerTTS"
    #                                               "/tts_outputs/english", keywords=keywords, random_words=random_words)
    #     with Pool(processes=10) as pool:
    #         items = [(executor.output_dir, word, True) for word in executor.keywords]
    #         # items += [(executor.output_dir, word, False) for word in executor.random_words]
    #         res = pool.starmap_async(VitsTTSExecutor.create_dir, items, error_callback=custom_error_callback)
    #         res.wait()
    #         speaker_args = []
    #         n = 5
    #         selected_speakers = get_selected_speakers(executor.speakers, 5)
    #         for _, j, k in items:
    #             for speaker in selected_speakers:
    #                 speaker_args.append((executor, j, speaker, k))
    #         res = pool.starmap_async(VitsTTSExecutor.create_dataset, speaker_args)
    #         res.wait()

    ############# for voice data augmentation ###################################
    with Manager() as manager:
        keywords = [keyword for keyword in EnglishKeyWords]
        executor = VitsTTSExecutor(clone_speakers_dir=None, audio_dir=None, language="en_US",
                                   output_dir="/Users/shankhajyoti.de/PythonProjects/MultiSpeakerTTS"
                                              "/tts_outputs/english", keywords=keywords)
        with Pool(processes=1) as pool:
            sampled_audio_files = manager.list()
            args_ = [(word, executor.output_dir, sampled_audio_files, True,) for word in keywords]
            res = pool.starmap_async(VitsTTSExecutor.find_random_voice_files_for_keyword, args_, error_callback=custom_error_callback)
            res.wait()

            args_ = [(file, dir_suffix, executor.output_dir, is_keyword, ["add_silence_with_noise"]) for file, dir_suffix, is_keyword in sampled_audio_files]
            res = pool.starmap_async(VitsTTSExecutor.augment_voice_data, args_)
            res.wait()



