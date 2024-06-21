from .MonteCarlo.mc import MonteCarlo
from .ShapReg.ShapleyRegression import ShapleyRegression
from .ShapReg.utils_shapreg import MarginalExtension, PredictionGame
from .fastshap.fastshap.fastshap import FastSHAP
from .fastshap.fastshap.utils import MaskLayer1d, MaskLayer2d, KLDivLoss
from .fastshap.fastshap.surrogate import Surrogate
from .DASP.dasp.dasp import DASP
from shap.explainers import Exact
from .DeepExplainers import DeepExplainer
# from .shapley_regression import removal, s

from keras.models import Sequential, Model, load_model
import shap
import torch 
import torch.nn as nn
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DASPModel():
    def __init__(self, df, model, M, num_features, data , data_val, load, dataset):
        self.explainer = DASP(Model(model.inputs, model.layers[-2].output))
        self.num_features = num_features
        self.time_train=0
        self.name='DASP'
        self.list_time=[]
        self.list_l1=[]
        self.list_l2=[]
        self.list_kendall=[]
        self.list_sv=[]

    def compute(self, IDX, x, y, kernelshap_iters):
        sv=self.explainer.run(x, self.num_features)
        svy=sv[0][y]
        return [svy]
    
class MonteCarloModel():
    def __init__(self, df, model, M, num_features, data , data_val, load, dataset):
        self.df = df
        self.model = model
        self.M = M
        self.num_features = num_features
        self.time_train=0
        self.name='MonteCarlo'
        self.list_time=[]
        self.list_l1=[]
        self.list_l2=[]
        self.list_kendall=[]
        self.list_sv=[]

    def compute(self, IDX, x, y, kernelshap_iters):
        svmc=MonteCarlo(IDX, self.df, self.model, self.M, self.num_features)
        svmc=svmc[:,y]
        return [svmc]
    
class DeepExplainerModel():
    def __init__(self, df, model, M, num_features, data , data_val, load, dataset):
        self.explainer = DeepExplainer(model, data[:1000])
        self.time_train=0
        self.name='DeepExplainer'
        self.list_time=[]
        self.list_l1=[]
        self.list_l2=[]
        self.list_kendall=[]
        self.list_sv=[]

    def compute(self, IDX, x, y, kernelshap_iters):
        svd = self.explainer.shap_values(x)
        return [svd[y][0]]
    
class GradientExplainerModel():
    def __init__(self, df, model, M, num_features, data , data_val, load, dataset):
        self.explainer = shap.GradientExplainer(model, data[:1000])
        self.time_train=0
        self.name='GradientExplainer'
        self.list_time=[]
        self.list_l1=[]
        self.list_l2=[]
        self.list_kendall=[]
        self.list_sv=[]

    def compute(self, IDX, x, y, kernelshap_iters):
        svd = self.explainer.shap_values(x)
        return [svd[y][0]]
    
class ExactExplainerModel():
    def __init__(self, df, model, M, num_features, data , data_val, load, dataset):
        self.explainer = shap.explainers.Exact(model.predict, data[:1000])
        self.time_train=0
        self.name='Exact'
        self.list_time=[]
        self.list_l1=[]
        self.list_l2=[]
        self.list_kendall=[]
        self.list_sv=[]

    def compute(self, IDX, x, y, kernelshap_iters):
        svd = self.explainer(x)
        return [svd.values[0][:,y]]
    

class FastSHAPModel():
    def __init__(self, df, model, M, num_features, data , data_val, load, dataset):

        if os.path.isfile(f'models/{dataset}_surrogate.pt') and load:
            self.time_train=0
            print('Loading saved surrogate model')
            surr = torch.load(f'models/{dataset}_surrogate.pt').to(device)
            surrogate = Surrogate(surr, num_features)
            # else:
                # run init with load=False
                # self.__init__(df, model, M, num_features, data , data_val, False, dataset)
            # if os.path.isfile(f'{dataset}_explainer.pt'):
            print('Loading saved explainer model')
            explainer = torch.load(f'models/{dataset}_explainer.pt').to(device)
            fastshap = FastSHAP(explainer, surrogate, normalization='additive', link=nn.Softmax(dim=-1))
            self.explainer = fastshap
            # else:
            #     # run init with load=False
            #     self.__init__(df, model, M, num_features, data , data_val, False, dataset)
        else:
            LAYER_SIZE = 256
            surr = nn.Sequential(
                MaskLayer1d(value=0, append=True),
                nn.Linear(2 * num_features, LAYER_SIZE),
                nn.ReLU(inplace=True),
                nn.Linear(LAYER_SIZE, LAYER_SIZE),
                nn.ReLU(inplace=True),
                nn.Linear(LAYER_SIZE, LAYER_SIZE),
                nn.ReLU(inplace=True),
                nn.Linear(LAYER_SIZE, 2)).to(device)

            # Set up surrogate object
            surrogate = Surrogate(surr, num_features)

            # Set up original model
            def original_model(x):
                pred = model.predict(x.cpu().numpy())
                # pred = np.stack([1 - pred, pred]).T
                return torch.tensor(pred, dtype=torch.float32, device=x.device)

            # Train
            print("\t\tTraining Surrogate")
            start_sur = time.time()
            surrogate.train_original_model(
                data,
                data_val,
                original_model,
                batch_size=8,
                lr=1e-5,
                max_epochs=400,
                loss_fn=KLDivLoss(),
                validation_samples=10,
                validation_batch_size=10000,
                lookback=20,
                verbose=False)
            time_surr = time.time()-start_sur

            surr.cpu()
            torch.save(surr, f'{dataset}_surrogate.pt')
            surr.to(device)
            
            explainer = nn.Sequential(
                nn.Linear(num_features, LAYER_SIZE),
                nn.LeakyReLU(inplace=True),
                nn.Linear(LAYER_SIZE, LAYER_SIZE),
                nn.LeakyReLU(inplace=True),
                nn.Linear(LAYER_SIZE, LAYER_SIZE),
                nn.LeakyReLU(inplace=True),
                nn.Linear(LAYER_SIZE, 2 * num_features)).to(device)


            # Set up FastSHAP object
            fastshap = FastSHAP(explainer, surrogate, normalization='additive',link=nn.Softmax(dim=-1))

            # Train
            print("\t\tTraining FastSHAP")
            start_fs = time.time()
            fastshap.train(
                data,
                data_val,
                lr=1e-4,
                batch_size=8,
                num_samples=16,
                max_epochs=400,
                validation_samples=128,
                lookback=20,
                verbose=False)
            time_fastshap = time.time()-start_fs

            explainer.cpu()
            torch.save(explainer, f'{dataset}_explainer.pt')
            explainer.to(device)

            self.time_train=time_surr+time_fastshap
            self.explainer = fastshap
        self.name='FastSHAP'
        self.list_time=[]
        self.list_l1=[]
        self.list_l2=[]
        self.list_kendall=[]
        self.list_sv=[]

    def compute(self, IDX, x, y, kernelshap_iters):
        fastshap_values = self.explainer.shap_values(x)[0]
        return [fastshap_values[:,y]]
    
class ShapleyRegressionModel(): #THRESHOLD CURRENTLY HARD-CODED
    def __init__(self, df, model, M, num_features, data , data_val, load, dataset):
        self.model_lam = lambda x: model.predict(x)
        self.marginal_extension = MarginalExtension(data, self.model_lam)
        self.time_train=0
        self.name='ShapleyRegression'
        self.list_time=[]
        self.list_l1=[]
        self.list_ul1=[]
        self.list_l2=[]
        self.list_ul2=[]
        self.list_kendall=[]
        self.list_ukendall=[]
        self.list_usv=[]
        self.list_sv=[]

    def compute(self, IDX, x, y, kernelshap_iters):
        game = PredictionGame(self.marginal_extension, x)
        # print(game)
        shap_values_UKS, all_results_KS = ShapleyRegression(game, batch_size=32, paired_sampling=False, detect_convergence=True, thresh=0.1, bar=False, return_all=True)
        svuks=shap_values_UKS.values[:,y]
        svks=all_results_KS['values'][list(all_results_KS['iters']).index(kernelshap_iters)][:,y]
        return [svuks, svks]