# with ph -> with phonemes
# with se -> speaker embeddings

import os
import random

import torch
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.utils.downloaders import download_vctk

torch.set_num_threads(24)

RUN_NAME = "YourTTS-EN-VCTK"

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "your_tts_model_with_ph_with_se")

RESTORE_PATH = "/Users/shankhajyoti.de/Library/ApplicationSupport/tts/tts_models--multilingual--multi-dataset--your_tts/model_file.pth"  # "/root/.local/share/tts/tts_models--multilingual--multi-dataset--your_tts/model_file.pth"
# RESTORE_PATH = "/TTSExecution/English/your_tts_model_with_ph_without_se/YourTTS-EN-VCTK-June-10-2023_07+31AM-0000000/best_model.pth"
# This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False

# Set here the batch size to be used in training and evaluation
BATCH_SIZE = 12
EVAL_BATCH_SIZE = 1

SAMPLE_RATE = 16_000

# Max audio length in seconds to be used in training (every audio bigger than it will be ignored)
MAX_AUDIO_LEN_IN_SECONDS = 10
DATASET_PATH = "/english_dataset/vctk"

# init configs
vctk_config = BaseDatasetConfig(
    formatter="vctk",
    dataset_name="digits_phonepe_ivr_usecase",
    meta_file_train="",
    meta_file_val="",
    path=DATASET_PATH,
    language="en"
)
DATASETS_CONFIG_LIST = [vctk_config]

### Extract speaker embeddings
SPEAKER_ENCODER_CHECKPOINT_PATH = (
    "/Users/shankhajyoti.de/Library/ApplicationSupport/tts/tts_models--multilingual--multi-dataset--your_tts/model_se.pth"
)
SPEAKER_ENCODER_CONFIG_PATH = "/Users/shankhajyoti.de/Library/ApplicationSupport/tts/tts_models--multilingual--multi-dataset--your_tts/config_se.json"



# Audio config used in training.
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
)

# Init VITSArgs setting the arguments that are needed for the YourTTS model
model_args = VitsArgs(
    num_layers_text_encoder=10,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    resblock_type_decoder="2",  # In the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
    # Useful parameters to enable the Speaker Consistency Loss (SCL) described in the paper
    use_speaker_encoder_as_loss=True,
    use_speaker_embedding=True
    # Useful parameters to enable multilingual training
    # use_language_embedding=True,
    # embedded_language_dim=4,
)

# General training config, here you can change the batch size and others useful parameters
config = VitsConfig(
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name="YourTTS",
    run_description="""
            - Finetuning your_tts for digit Keywords.py 
        """,
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    # batch_group_size=10,
    eval_batch_size=EVAL_BATCH_SIZE,
    num_loader_workers=0,
    num_eval_loader_workers=0,
    epochs=30,
    save_step=1000,
    save_best_after=1000,
    print_step=50,
    plot_step=100,
    log_model_step=1000,
    save_n_checkpoints=2,
    save_checkpoints=True,
    target_loss="loss_1",
    print_eval=False,
    use_phonemes=True,
    phonemizer="espeak",
    phoneme_language="en",
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="english_cleaners",
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz¯·ßàáâãäæçèéêëìíîïñòóôõöùúûüÿāąćēęěīıłńōőœśūűźżǎǐǒǔабвгдежзийклмнопрстуфхцчшщъыьэюяёєіїґ–!'(),-.:;? ɐɑɒɓɔɕɖɗɘəɚɛɜɝɞɟɠɡɢɣɤɥɦɧɨɩɪɫɬɭɮɯɰɱɲɳɴɵɶɷɸɹɺɻɼɽɾɿʀʁʂʃʄʅʆʇʈʉʊʋʌʍʎʏʐʑʒʓʔʕʖʗʘʙʚʛʜʝʞʟʠʡʢʣʤʥʦʧʨʩʪʫʬʭʮˈˌːˑʼʴʰʱʲʷˠˤ˞ŋðθ",
        punctuations="!'(),-.:;? ",
        phonemes="",
        is_unique=True,
        is_sorted=True,
    ),
    phoneme_cache_path="/FinetuningExamples/vits/English/phoneme_cache",
    precompute_num_workers=10,
    start_by_longest=True,
    datasets=DATASETS_CONFIG_LIST,
    cudnn_benchmark=False,
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=False,
    eval_split_size=0,
    # Enable the weighted sampler
    use_weighted_sampler=True,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    weighted_sampler_attrs={"speaker_name": 1.0},
    weighted_sampler_multipliers={},
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the paper
    speaker_encoder_loss_alpha=9.0,
    run_eval=False,
    test_sentences=[
                ["one."],
                ["two."],
                ["three."],
                ["four."],
                ["five."],
                ["six."],
                ["seven."],
                ["eight."],
                ["nine."],
                ["zero."],
                ["star."],
                ["repeat."]
            ]
)

# Load all the datasets samples and split traning and evaluation sets
train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=False,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

ap = AudioProcessor.init_from_config(config)
tokenizer, new_config = TTSTokenizer.init_from_config(config)

speaker_manager = SpeakerManager()
if eval_samples == None:
    final_samples = train_samples
else:
    final_samples = train_samples + eval_samples

speaker_manager.set_ids_from_data(final_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers

if config.model_args.speaker_encoder_model_path:
    speaker_manager.init_encoder(
        config.model_args.speaker_encoder_model_path, config.model_args.speaker_encoder_config_path
    )

# Init the model
model = Vits(config, ap, tokenizer, speaker_manager)

# Init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
