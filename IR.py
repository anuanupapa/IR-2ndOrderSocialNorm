'''
Some problem 
'''




import numpy as np
import numba as nb
import matplotlib.pyplot as plt

#reputation_arr - reputation is public
#strategy_arr - decision in the form of duple (pG, pB)


@nb.njit
def initialize(N):

    #Random reputation assigned to each of the players
    rep_arr = np.random.randint(0,2,(N, 1))
    #Random initial config with
    #25% of each of the ALLD, ALLC, pDisc & Disc players
    strat_arr = np.random.randint(0,2,(N, 2))

    return rep_arr, strat_arr


@nb.njit
def sim(N, soc_norm, explore, trials=10, gens=20000):

    player_arr = np.arange(0,N,1)
    cfrac_arr = np.zeros((trials, gens))
    round_arr = np.zeros((trials, gens))
    
    for trial_no in range(trials):        
        print(trial_no)
        #initialize
        [reputation_arr, strategy_arr
         ] = initialize(N)
        
        round_no = -1
        for _ in range(gens):

            #Choose a player for the round
            agentA = np.random.choice(player_arr)
            '''
            If the player decides to explore then
            the round is not considered
            '''
            if np.random.random()<explore:
                strategy_arr[agentA,:
                             ] = np.array([np.random.randint(0,2),
                                           np.random.randint(0,2)])

            else:
                agentB = find_partner(agentA, player_arr)
                round_no = round_no + 1
                cumpayA = 0
                cumpayB = 0
                C_frac = 0
                for games in range(2*N):
                    A = find_partner(agentA, player_arr)
                    [currpayA, Cf
                     ] = payoffreputation_update(agentA, A,
                                                 strategy_arr,
                                                 reputation_arr,
                                                 soc_norm)
                    cumpayA = cumpayA + currpayA
                    C_frac = C_frac + Cf
                    
                    B = find_partner(agentB, player_arr)
                    [currpayB, Cf
                     ]= payoffreputation_update(agentB, B,
                                                strategy_arr,
                                                reputation_arr,
                                                soc_norm)
                    cumpayB = cumpayB + currpayB
                    C_frac = C_frac + Cf
                    
                payoffA = cumpayA/(2*N)
                payoffB = cumpayB/(2*N)
                if np.random.random()<1/(1+np.exp(payoffA-payoffB)):
                    strategy_arr[agentA,:
                                 ]=np.copy(strategy_arr[agentB,:])
                else:
                    pass

            cfrac_arr[trial_no, round_no] = C_frac/(8*N)
            round_arr[trial_no, round_no] = round_no
                
    return cfrac_arr, round_arr


@nb.njit
def payoffreputation_update(agent1, agent2, strat_arr, rep_arr,
                            Norm,
                            assessmentErr=0.01, executionErr=0.08,
                            assignementErr=0.01,
                            b=5, c=1, reputation_timescale=1):

    act1 = strat_arr[agent1,:]
    act2 = strat_arr[agent2,:]
    rep1 = rep_arr[agent1,0]
    rep2 = rep_arr[agent2,0]
    
    cfrac = 0
    #Game is played with p1 donating and p2 recieving    
    if np.random.random()<assessmentErr:
        if np.random.random()<executionErr:
            don1 = 1-act1[1-rep2]
        else:
            don1 = act1[1-rep2]
    else:
        if np.random.random()<executionErr:
            don1 = 1-act1[rep2]
        else:
            don1 = act1[rep2]
    #Game is played with p2 donating and p1 recieving
    if np.random.random()<assessmentErr:
        if np.random.random()<executionErr:
            don2 = 1-act2[1-rep1]
        else:
            don2 = act2[1-rep1]
    else:
        if np.random.random()<executionErr:
            don2 = 1-act2[rep1]
        else:
            don2 = act2[rep1]        
    cfrac = (don1 + don2)/c
    
    #Player 1 reputation is assigned
    if np.random.random() < reputation_timescale:
        if np.random.random() < assignementErr:            
            rep_arr[agent1,0] = 1-Norm[act1[rep2],rep2]
        else:
            rep_arr[agent1,0] = Norm[act1[rep2],rep2]
    else:
        pass
    #Player 2 reputation is assigned
    if np.random.random() < reputation_timescale:
        if np.random.random() < assignementErr:            
            rep_arr[agent2,0] = 1-Norm[act2[rep1],rep1]
        else:
            rep_arr[agent2,0] = Norm[act2[rep1],rep1]
    else:
        pass
        
    return b*don2 - c*don1, cfrac


@nb.njit                
def find_partner(agentID, agents):
    selectedID = agentID
    while selectedID == agentID:
        selectedID = np.random.choice(agents)
    return selectedID

'''
@nb.njit                
def find_partner(agentID, agents):
    agent_arr = np.delete(agents, agentID)
    selectedID = np.random.choice(agent_arr)[0]
    return selectedID
'''

if __name__=="__main__":

    import time
    tt = time.time()
    
    
    N = 50
    trials=1
    gens=100000
    soc_norm = np.array([[1,0],[0,1]])
    
    [cfrac, r] = sim(N, soc_norm, 0., trials, gens)
    print(np.mean(np.mean(cfrac[:,-1000:], axis=-1), axis=-1))
    print("time taken : ", time.time()-tt)
    
    plt.plot(r[0,:], cfrac[0,:], 'o-')
    plt.show()

    '''

    # Versus N

    trials=10
    gens=25000
    N_arr = [10,20,30,40,50,60,70,80,90,100]
    c_arr = []
    
    soc_norm = np.array([[1,0],[0,1]])

    ind=-1
    for N in N_arr:
        print(N)
        ind = ind + 1
        [cfrac, r] = sim(N, soc_norm, 0.1/N, trials, gens)
        c = np.mean(np.mean(cfrac[:,20000:],axis=-1),axis=-1)
        c_arr.append(c)
        print(c_arr)
    plt.plot(N_arr, c_arr, 'o-')
    plt.ylim(0,1)
    print("time taken : ", time.time()-tt)

    plt.show()
    
    '''
