from tts_execution.BaseTTSExecutor import BaseTTSExecutor

"""
freevc24 is a model inspired from vits architecture owing to vits' waveform reconstruction capability along with its ability
to extract and disentangle the speaker characteristics from the audio content features. 
freevc24 will convert the content of the source_wav to target_wav i.e. the resultant audio will have the prosody, emotion, phonemes of 
the target_wav
"""
class FreeVc24Executor(BaseTTSExecutor):

    def __init__(self, clone_speakers_dir: str, clone_audio_dir: str, language: str, keywords=[], random_words=[]):
        super(FreeVc24Executor, self).__init__(clone_speakers_dir, language, keywords, random_words)
        self.clone_audio_dir = clone_audio_dir

    def create_dataset(self):
        pass
