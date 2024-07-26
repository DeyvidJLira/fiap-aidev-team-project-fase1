import joblib

class SaveModelManager:

    @staticmethod
    def save_model(model, filename):
        joblib.dump(model, f'../model/{filename}')
        print(f'Saved model: {filename}')

    @staticmethod
    def load_model(filename):
        model = joblib.load(filename)
        print(f'Loaded model: {filename}')
        return model