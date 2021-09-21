import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import icecream as ic


@nb.njit
def fermi_update(payoff_arr, update_arr, AdjMat, beta):
    N = len(payoff_arr)

    plCentral = np.random.choice(N, 1)[0]
    
    Neighs = np.where(AdjMat[plCentral]==1)[0]
    plNeigh = np.random.choice(Neighs, 1)[0]
    
    payCentral = payoff_arr[plCentral][0]
    payNeigh = payoff_arr[plNeigh][0]
    
    prob = 1./(1+np.exp((payCentral-payNeigh)*beta))
    
    if np.random.random()<prob:
        update_arr[plCentral] = update_arr[plNeigh]
    
    
    return update_arr

@nb.njit
def game(act_arr, AM):

    T, R, P, S = 4,3,1,2
    
    Pmat = np.array([[R, S], [T, P]])
    pay_arr = np.zeros_like(act_arr)

    N = len(pay_arr)
    for p1Ind in range(N):
        Neighs = np.where(AM[p1Ind]==1)[0]
        neighs = Neighs[Neighs>p1Ind]
        for p2Ind in neighs:
        #for p2Ind in range(p1Ind+1, N, 1):
            act1 = act_arr[p1Ind][0]
            act2 = act_arr[p2Ind][0]
            pay_arr[p1Ind][0] = pay_arr[p1Ind][0]+Pmat[int(1-act1),
                                                       int(1-act2)]
            pay_arr[p2Ind][0] = pay_arr[p2Ind][0]+Pmat[int(1-act2),
                                                       int(1-act1)]
    return(pay_arr)
    
@nb.njit
def fixation_probability(N, AM, beta, trials=100):
    
    FixProb_arr = np.zeros((N+1,1))
    init_c = np.arange(0,N+1,1)

    initc_counter=-1
    for initc in init_c:
        fix_prob = np.zeros((trials,1))
        initc_counter = initc_counter + 1
        print(initc)
        
        for tr in range(trials):
            act_arr = np.zeros((N,1))
            
            for i in range(initc):
                act_arr[i] = 1.            
            runtime = 0
            while np.sum(act_arr)!=N and np.sum(act_arr)!=0 and runtime <500:
                runtime = runtime + 1
                pay_arr = game(act_arr, AM)
                act_arr = fermi_update(pay_arr, act_arr, AM, beta)
            fix_prob[tr][0] = np.sum(act_arr)/N
        FixProb_arr[initc_counter][0] = np.sum(fix_prob)/len(fix_prob)

    return FixProb_arr, init_c

        
if __name__ == "__main__":

    import time
    
    '''
    pay_arr = np.array([[10.],[20.]])
    act_arr = np.array([[0],[1]])
    AM = np.array([[[0],[1]],[[1],[0]]])
    act_arr = fermi_update(pay_arr, act_arr, AM, 1.)
    print(act_arr)
    pay_arr = game(act_arr, AM)
    print(pay_arr)
    '''
    tt = time.time()

    N=20
    beta_arr = [0.1, 0.05]
    AM = np.ones((N,N))-np.eye(N)

    for beta in beta_arr:
        print(beta)
        [fp, x] = fixation_probability(N, AM, beta, 1000)
        plt.plot(x, fp, 'o-', label=r'$\beta=$'+str(beta))
        print(time.time()-tt)
    plt.ylabel("Fixation probability")
    plt.xlabel("Initial C-frac")
    plt.legend()
    plt.title("SnowDrift-4,3,1,2")
    plt.savefig("Fixation_SnowDrift.png")
    plt.show()
