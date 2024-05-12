
class Word:
    def __init__(self, text: str, dir_suffix: str):
        self.text = text
        self.dir_suffix = dir_suffix if dir_suffix is not None else text

