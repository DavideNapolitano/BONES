import random
import numpy as np
from tqdm import tqdm

def MonteCarlo(IND, DATA, MODEL, M, NUM_FEATURES):

    sv_mc=[]

    x=DATA.iloc[IND]

    for j in range(NUM_FEATURES):
        n_features = len(x)
        marginal_contributions = []
        marginal_contributions_U =[]

        feature_idxs = list(range(n_features))
        feature_idxs.remove(j)
        for itr in (range(M)):
            z = DATA.sample(1).values[0]
            x_idx = random.sample(feature_idxs, min(max(int(0.2*n_features), random.choice(feature_idxs)), int(0.8*n_features))) #estraggo 0.8*feature_idx
            z_idx = [idx for idx in feature_idxs if idx not in x_idx] # features non estratte. Ricorda che una feauture, quella su cui si calcola lo SV, è sempre esclusa

            # construct two new instances
            x_plus_j = np.array([x[i] if i in x_idx + [j] else z[i] for i in range(n_features)])
            x_minus_j = np.array([z[i] if i in z_idx + [j] else x[i] for i in range(n_features)])

            ##############################################################################à
            # calculate marginal contribution
            x_plus_j=x_plus_j.reshape(1, -1)#np.expand_dims(x_plus_j, axis=0)
            x_minus_j=x_minus_j.reshape(1, -1)#np.expand_dims(x_minus_j, axis=0)

            # v_m_plus = MODEL.predict_proba(x_plus_j)[0]
            # v_m_minus = MODEL.predict_proba(x_minus_j)[0]
            v_m_plus = MODEL.predict(x_plus_j)[0]
            v_m_minus = MODEL.predict(x_minus_j)[0]
            # print(v_m_plus, v_m_minus)

            marginal_contribution = v_m_plus - v_m_minus 
            marginal_contributions.append(marginal_contribution)
            
            # break

        marginal_contributions=np.array(marginal_contributions)

        phi_j_x = np.sum(marginal_contributions, axis=0) / len(marginal_contributions)  # our shaply value

        sv_mc.append(phi_j_x)
        
    return np.array(sv_mc)