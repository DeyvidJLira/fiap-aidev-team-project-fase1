from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import matplotlib.pyplot as plt
from save_model_manager import SaveModelManager
from util import Util


class TrainModelBase:

    def __init__(self, p_name):
        self.name_model = p_name
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.random_state = 42


    def train(self, X, y, test_size):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        print(len(self.x_train), "training +", len(self.x_test), "tests")
        print(len(self.y_train), "training +", len(self.y_test), "tests")
        return self.x_train, self.x_test, self.y_train, self.y_test
    

    def get_model(self):
        return self.model
    
    
    def evaluate(self):
        Util.evaluate(self.model, self.x_test, self.y_test)
    

    def evaluate_pred(self):
        y_pred = self.model.predict(self.x_test)
        plt.scatter(self.y_test, y_pred)
        plt.xlabel("Valores reais")
        plt.ylabel("Previsões")
        plt.title("Comparando")
        plt.show()

    
    def save(self):
        filename = self.name_model + '.pkl'
        SaveModelManager.save_model(self.model, filename)

    
class TrainModelLinearRegression(TrainModelBase):  

    def generate_model(self):
        self.model = LinearRegression()
        self.model.fit(self.x_train, self.y_train)
        return self.model


class TrainModelRidgeRegression(TrainModelBase):   
        
    def generate_model(self):
        self.model = Ridge(alpha=1.0, random_state=self.random_state)
        self.model.fit(self.x_train, self.y_train)
        return self.model
    

class TrainModelLassoRegression(TrainModelBase):   
        
    def generate_model(self):
        self.model = Lasso(alpha=0.1, random_state=self.random_state)
        self.model.fit(self.x_train, self.y_train)
        return self.model


class TrainModelElasticNetRegression(TrainModelBase):   
        
    def generate_model(self):
        self.model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state)
        self.model.fit(self.x_train, self.y_train)
        return self.model
    
