from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class Util:

    @staticmethod
    def calculate_rmse(mse):
        return np.sqrt(mse)
    
    
    @staticmethod
    def calculate_mape(labels, predictions):
        errors = np.abs(labels - predictions)
        relative_errors = errors / np.abs(labels)
        mape = np.mean(relative_errors) * 100
        return mape
    
    @staticmethod
    def evaluate(model, x_test, y_test):
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = Util.calculate_rmse(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape =  Util.calculate_mape(y_test, y_pred)
        print(f'Mean squared error: {mse}')
        print(f'Sqrt Mean squared error: {rmse}')
        print(f'R2 score: {r2}')
        print(f'Mean absolute error: {mae}')
        print(f'MAPE: {mape:.2f}%')
        