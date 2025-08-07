from config import load_trained_model


class TransformerHandler:
    def __init__(self):
        self.model = load_trained_model()

    def encode_input(self, input: str):
        return self.model.encode(input).tolist()
