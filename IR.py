"""
simple standing and shunning are reversed
Reputation table for 4 types of social norms
SJ - SternJudging - [[G,B],[B,G]]
SH - Shunning - [[G,B],[B,B]]
SS - SimpleStanding - [[G,G],[B,G]]
ID - ImageScoring - [[G,G],[B,B]]
Good - 0
Bad - 1
"""
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import csv

#reputation_arr - reputation is public
#strategy_arr - decision in the form of duple (pG, pB)


@nb.njit
def initialize(N):

    #Random reputation assigned to each of the players
    # 0 - bad, 1 - good
    rep_arr = np.random.randint(0,2,(N, 1))
    #Random initial config with
    #25% of each of the ALLD, ALLC, pDisc & Disc players
    strat_arr = np.random.randint(0,2,(N, 2))
    return rep_arr, strat_arr



@nb.njit
def payoffreputation_update(agent1, agent2, strat_arr, rep_arr,
                            Norm,
                            assessmentErr, executionErr,
                            assignmentErr,
                            b, c, reputation_timescale):
    #rep_updated_arr = np.copy(rep_arr)
    act1 = strat_arr[agent1,:]
    act2 = strat_arr[agent2,:]
    rep1 = rep_arr[agent1,0]
    rep2 = rep_arr[agent2,0]

    '''
    act=[pG, pB]
    norm=[[d(GC),d(BC)], 
          [d(GD),d(BD)]]
    G - 0
    B - 1
    C - 1
    D - 0
    '''
    
    cfrac = 0
    '''Actions are chosen'''
    #Game is played with p1 donating and p2 receiving    
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
    #Game is played with p2 donating and p1 receiving
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
    cfrac = don1 + don2
    
    '''Assign Reputations'''
    #Player 1 reputation is assigned
    if np.random.random() < reputation_timescale:
        if np.random.random() < assignmentErr:            
            rep_arr[agent1,0] = 1-Norm[1-don1,rep2]
        else:
            rep_arr[agent1,0] = Norm[1-don1,rep2]
        # 1-don1 because of indexing (C - 1, D - 0)
    else:
        pass
    #Player 2 reputation is assigned
    if np.random.random() < reputation_timescale:
        if np.random.random() < assignmentErr:            
            rep_arr[agent2,0] = 1-Norm[1-don2,rep1]
        else:
            rep_arr[agent2,0] = Norm[1-don2,rep1]
    else:
        pass
        
    return b*don2 - c*don1, cfrac


@nb.njit
def sim(N, soc_norm, explore, trials=10, gens=20000,
        assessmentErr=0.01, executionErr=0.08, assignmentErr=0.01,
        b=5, c=1, reputation_timescale=1):
        

    player_arr = np.arange(0,N,1)
    cfrac_arr = np.zeros((trials, gens))
    round_arr = np.zeros((trials, gens))
    
    for trial_no in range(trials):        
        print("trial-",trial_no)
        #initialize
        [reputation_arr, strategy_arr
         ] = initialize(N)
        
        round_no = -1

        for _ in range(gens):

            #Choose a player for the round
            [agentA, agentB] = np.random.choice(player_arr, 2,
                                                replace=False)
            '''
            If the player decides to explore then
            the round is not considered
            '''
            '''
            #Exploration
            if np.random.random()<explore:
                strategy_arr[agentA,:
                             ] = np.array([np.random.randint(0,2),
                                           np.random.randint(0,2)])
            '''
            #'''
            #SMA
            
            if np.shape(np.unique(
                    strategy_arr[:,0]))[0]==1 or np.shape(np.unique(
                            strategy_arr[:,0]))[0]==1:
                strategy_arr[agentA,:
                             ] = np.array([np.random.randint(0,2),
                                           np.random.randint(0,2)])
                
            #'''
            else:
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
                                                 soc_norm,
                                                 assessmentErr,
                                                 executionErr,
                                                 assignmentErr,
                                                 b, c,
                                                 reputation_timescale)
                    cumpayA = cumpayA + currpayA
                    C_frac = C_frac + Cf
                    
                    B = find_partner(agentB, player_arr)
                    [currpayB, Cf
                     ] = payoffreputation_update(agentB, B,
                                                 strategy_arr,
                                                 reputation_arr,
                                                 soc_norm,
                                                 assessmentErr,
                                                 executionErr,
                                                 assignmentErr,
                                                 b, c,
                                                 reputation_timescale)
                    cumpayB = cumpayB + currpayB
                    C_frac = C_frac + Cf

                # Normalize payoff
                payoffA = cumpayA/(2*N)
                payoffB = cumpayB/(2*N)
                
                # Copy strategy accorind to fermi rule
                if np.random.random()<1/(1+np.exp(payoffA-payoffB)):
                    strategy_arr[agentA,:
                                 ]=np.copy(strategy_arr[agentB,:])
                else:
                    pass

            cfrac_arr[trial_no, round_no] = C_frac/(8*N)
            round_arr[trial_no, round_no] = round_no
                
    return cfrac_arr, round_arr



@nb.njit                
def find_partner(agentID, agents):
    selectedID = agentID
    while selectedID == agentID:
        selectedID = np.random.choice(agents)
    return selectedID


if __name__=="__main__":

    import time
    import csv
    tt = time.time()
    '''
    N = 150
    trials = 5
    gens=200000
    soc_norm = np.array([[0,0],[1,0]])
    
    [cfrac, r] = sim(N, soc_norm, 0.1/N, trials, gens)
    indexout = np.where(r[0,1:]==0)[0][0]
    print(indexout)
    print(np.mean(np.mean(cfrac[:,10000:indexout], axis=-1), axis=-1))
    print("time taken : ", time.time()-tt)
    for tr in range(trials):
        plt.plot(r[tr,:indexout], cfrac[tr,:indexout], ',', linewidth=0.1, alpha=tr/trials)
    plt.show()
    '''
    '''
    # Versus N
    plt.clf()
    w=csv.writer(open("Nvary.csv", "w"))
    trials=10
    gens=3*int(10**5)
    Z_arr = [10,20,30,40,50,60,80,100]
    soc_norm_arr = np.array([[[1,0],[0,1]],[[1,1],[0,1]],
                             [[1,0],[0,0]],[[1,1],[0,0]]])
    Labels = ["SJ", "SS", "SH", "IS"]
    dict_save={}
    soc_label=-1
    for soc_norm in soc_norm_arr:
        soc_label = soc_label + 1
        print(soc_norm)
        c_arr = []
        ind=-1
        for Z in Z_arr:
            print(Z)
            ind = ind + 1
            [cfrac, r] = sim(Z, soc_norm, 0, trials, gens)
            indexout = np.where(r[0,1:]==0)[0][0]
            c = np.mean(np.mean(cfrac[:,int(gens/10):indexout],
                                axis=-1),axis=-1)
            c_arr.append(c)
            print(c_arr)
        plt.plot(N_arr, c_arr,
                 'o-', label=Labels[soc_label])
        dict_save[Labels[soc_label]]=c_arr
    for key, val in dict_save.items():
        w.writerows([key, val])
    plt.xscale("log")
    #plt.ylim(0,1)
    plt.legend()
    plt.ylabel("cooperation index")
    plt.xlabel("population size (N)")
    print("time taken : ", time.time()-tt)
    plt.savefig("cfrac_N.png")
    plt.show()
    '''

    
    # Versus assignmentErr
    
    trials=10
    gens=1*int(10**6)
    N=60
    assignmentErr_arr = [0.001,0.005,0.01,0.025,0.04,0.055,
                         0.075,0.1,0.2]
    #ssignmentErr_arr = assignmentErr_arr[:-2]
    print(assignmentErr_arr)
    soc_norm_arr = np.array([[[0,1],[1,0]],[[0,0],[1,0]],
                             [[0,1],[1,1]],[[0,0],[1,1]]])
    Labels = ["SJ", "SS", "SH", "IS"]
    dict_save={}
    soc_label=-1
    #c_Arr = np.zeros((gens,))
    cVSaE_arr = np.zeros((4, len(assignmentErr_arr)))
    for soc_norm in soc_norm_arr:
        soc_label = soc_label + 1
        print(soc_norm)
        c_arr = []
        indaE=-1
        for aE in assignmentErr_arr:
            print("assErr=",aE)
            indaE = indaE + 1
            [cfrac, r] = sim(N, soc_norm, 1/N, trials, gens,
                             assignmentErr=aE)
            indexout = np.where(r[0,1:]==0)[0][0]
            print(indexout)
            c = np.mean(np.mean(cfrac[:,int(gens/2):indexout],
                                axis=-1),axis=-1)
            c_arr.append(c)
            cVSaE_arr[soc_label,indaE] = c            
            print(c_arr)
        plt.plot(assignmentErr_arr, c_arr,
                 'o-', label=Labels[soc_label],
                 colors=["r","g","b",'k'])
    np.savez("assignmentErr.npz",
             assErr=assignmentErr_arr, soc_norm = soc_norm_arr,
             cfrac = cVSaE_arr)
    plt.xscale("log")
    plt.ylim(0,1)
    plt.legend()
    plt.ylabel("cooperation index")
    plt.xlabel("assignment error")
    print("time taken : ", time.time()-tt)
    plt.savefig("cfrac_assignmentErr.png")
    plt.show()
    

    '''
    # Versus assessmentErr
    
    w=csv.writer(open("assessmentErr.csv", "w"))
    trials=10
    gens=3*int(10**5)
    N=60
    assessmentErr_arr = np.logspace(-3,0,10)
    assessmentErr_arr = assessmentErr_arr[:-2]
    print(assessmentErr_arr)
    soc_norm_arr = np.array([[[1,0],[0,1]],[[1,1],[0,1]],
                             [[1,0],[0,0]],[[1,1],[0,0]]])
    Labels = ["SJ", "SS", "SH", "IS"]
    dict_save={}
    soc_label=-1
    for soc_norm in soc_norm_arr:
        soc_label = soc_label + 1
        print(soc_norm)
        c_arr = []
        ind=-1
        for aE in assessmentErr_arr:
            print(N)
            ind = ind + 1
            [cfrac, r] = sim(N, soc_norm, 1/N, trials, gens,
                             assessmentErr=aE)
            indexout = np.where(r[0,1:]==0)[0][0]
            c = np.mean(np.mean(cfrac[:,int(gens/10):indexout],
                                axis=-1),axis=-1)
            c_arr.append(c)
            print(c_arr)
        plt.plot(assessmentErr_arr, c_arr,
                 'o-', label=Labels[soc_label])
        dict_save[Labels[soc_label]]=c_arr
    for key, val in dict_save.items():
        w.writerows([key, val])
    plt.xscale("log")
    plt.ylim(0,1)
    plt.legend()
    plt.ylabel("cooperation index")
    plt.xlabel("assessment error")
    print("time taken : ", time.time()-tt)
    plt.savefig("cfrac_assessmentErr.png")
    plt.show()
    

    # Versus executionErr

    w=csv.writer(open("executionErr.csv", "w"))
    trials=10
    gens=3*int(10**5)
    N=60
    executionErr_arr = np.logspace(-3,0,10)
    executionErr_arr = executionErr_arr[:-2]
    print(executionErr_arr)
    soc_norm_arr = np.array([[[1,0],[0,1]],[[1,1],[0,1]],
                             [[1,0],[0,0]],[[1,1],[0,0]]])
    Labels = ["SJ", "SS", "SH", "IS"]
    soc_label=-1
    dict_save = {}
    for soc_norm in soc_norm_arr:
        soc_label = soc_label + 1
        print(soc_norm)
        c_arr = []
        ind=-1
        for aE in executionErr_arr:
            print(N)
            ind = ind + 1
            [cfrac, r] = sim(N, soc_norm, 1/N, trials, gens,
                             executionErr=aE)
            indexout = np.where(r[0,1:]==0)[0][0]
            c = np.mean(np.mean(cfrac[:,int(gens/10):indexout],
                                axis=-1),axis=-1)
            c_arr.append(c)
            print(c_arr)
        plt.plot(executionErr_arr, c_arr,
                 'o-', label=Labels[soc_label])
        dict_save[Labels[soc_label]]=c_arr
    for key, val in dict_save.items():
        w.writerows([key, val])
    plt.xscale("log")
    plt.ylim(0,1)
    plt.legend()
    plt.ylabel("cooperation index")
    plt.xlabel("execution error")    
    print("time taken : ", time.time()-tt)
    plt.savefig("cfrac_executionErr.png")
    plt.show()
    
    '''
