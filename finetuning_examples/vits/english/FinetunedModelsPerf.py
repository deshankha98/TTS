from TTS.api import TTS

tts = TTS(
    config_path="/Users/shankhajyoti.de/Library/ApplicationSupport/tts/tts_models--en--vctk--vits/config.json",
    progress_bar=False, gpu=False,
    model_path="/Users/shankhajyoti.de/Library/ApplicationSupport/tts/tts_models--en--vctk--vits/model_file.pth")  ## can be path to any finetuned model
# print(tts.speakers)
tts.tts_to_file("star.", file_path="output.wav", speaker="p226")
