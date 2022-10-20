"""
Created on Wed Jun 16 15:28:33 2021

@author: wowohead
"""

import CCNMF
import numpy as np
import networkx as nx
import NewJaccard
import matplotlib.pyplot as plt
import networkx.algorithms.community as nx_comm
from sklearn import metrics
import Jaccard
import NewJaccard_3
import pur_score
import math

import sys
sys.path.append('/home/ssx/code/data_processing')
import texas
import cornell
import washington
import wisconsin
# import Smith
# import cora
# import KKI
import gene
import reality
# import OHSU
import citeseer
# import AIDS
# import PTC_MM
# import MSRC
import BZR
# import SW_10000
# import CL
# import soc
# import Fingerprint
# import letter
# import DD
# import DD242
# import DD6
# import Email
# import terroristrel
# import COX

# import bio_yeast




if __name__ == "__main__":


    # A = np.array(nx.adjacency_matrix(G).todense())
    # A,ground_community_labels = texas.get_A()
    # A,ground_community_labels = cornell.get_A()
    # A,ground_community_labels = washington.get_A()
    # A,ground_community_labels = wisconsin.get_A()
    # A,ground_community_labels = cora.get_A()
    # A,ground_community_labels = KKI.get_A()
    # A,ground_community_labels = gene.get_A()
    # A,ground_community_labels = reality.get_A()
    # A,ground_community_labels = OHSU.get_A()
    # A = Smith.get_A()
    # A,ground_community_labels = citeseer.get_A()
    # A,ground_community_labels = AIDS.get_A()
    # A,ground_community_labels = PTC_MM.get_A()
    # A,ground_community_labels = MSRC.get_A()
    A,ground_community_labels = BZR.get_A()
    # A,ground_community_labels = SW_10000.get_A()
    # A,ground_community_labels = CL.get_A()
    # A,ground_community_labels = soc.get_A()
    # A,ground_community_labels = Fingerprint.get_A()
    # A,ground_community_labels = letter.get_A()
    # A,ground_community_labels = DD.get_A()
    # A,ground_community_labels = DD242.get_A()
    # A,ground_community_labels = DD6.get_A()
    # A,ground_community_labels = Email.get_A()
    # A,ground_community_labels = terroristrel.get_A()
    # A,ground_community_labels = COX.get_A()

    # A = bio_yeast.get_A()
    G = nx.Graph(A)

    filename = 'citeseer'
    number = 6
    ACC_average = 0.2878
    F_average = 0.3278
    F_standard = 0
    ACC_standard = 0


    size = 0
    max_component = []
    delete_component = []
    for component in nx.connected_components(G):
        if len(component) == 1:
            del_component = list(component)
            delete_component.append(del_component[0])
            # print(component)

    
    delete_component.sort(reverse=True)
    for i in range(len(delete_component)):
        G.remove_node(delete_component[i])
        ground_community_labels.pop(delete_component[i])
    
    # print(delete_component)

    
    A = np.array(nx.adjacency_matrix(G).todense())
    G = nx.Graph(A)
        
    
    # A = np.array(nx.adjacency_matrix(G).todense())
    # G = nx.Graph(A)




    # Jaccard = Jaccard.get_Jaccard(A)
    # S_1 = NewJaccard.getSimilarity_1_order(G)
    # S_2 = NewJaccard.getSimilarity(G)
    S_3 = NewJaccard_3.get_3_order_similarity(A)

    n = len(G.nodes())
    S_la = np.zeros((n,n))
    for i in range(n):
        temp = 0
        for j in range(n):
            temp = temp + S_3[i][j]
        S_la[i][i] = temp

    
    # lam = [10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3]
    # alpha = [10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3]
    Max_M = 0
    Max_N = 0
    Max_F = 0
    Max_A = -1
    Max_AUC = 0
    Max_P = 0
    Max_ACC = 0
    
    Min_M = 1
    Min_F = 1
    Min_A = 1
    Min_N = 1
    Min_AUC = 1
    Min_P = 1
    Min_ACC = 1

    total_M = 0
    total_F = 0
    total_A = 0
    total_N = 0
    total_AUC = 0
    total_P = 0
    total_ACC = 0
    for n in range(0,20):
        parameter = [10**3, 10**2, 10**1, 10**0, 10**-1, 10**-2, 10**-3]
        temp_M = 0
        temp_F = 0
        temp_A = 0
        temp_N = 0
        temp_P = 0
        temp_ACC = 0
        for lam in parameter:
            for al in parameter:
                # Max_M = 0
                # Max_A = 0
                # Max_N = 0
                # Max_F = 0

                H = CCNMF.CCNMF(A,S_la,S_3,max_iter=500,k=number,l=lam,alpha=al,cita=1000,beta=0.5)
                clusters = {}
                H_list = H.tolist()
                for i in range(len(H_list)):
                    clusters[i] = H_list[i].index(max(H_list[i]))
                

                com_num = []
                communities = []
                detected_community_labels = []
                for i in range(len(clusters)):
                    if clusters[i] not in com_num:
                        com_num.append(clusters[i])
                
                for i in range(len(com_num)):
                    communities.append(set())
                    
                for i in range(len(clusters)):
                    communities[com_num.index(clusters[i])].add(i)
                for i in range(len(clusters)):
                    detected_community_labels.append(clusters[i])
                

                matrix = metrics.cluster.pair_confusion_matrix(ground_community_labels,detected_community_labels)
                F_score = (2*matrix[1][1]) / (2*matrix[1][1] + matrix[0][1] + matrix[1][0])
                if (temp_F < F_score):
                    temp_F = F_score
                
                # ARI = metrics.adjusted_rand_score(ground_community_labels,detected_community_labels)
                # if (temp_A < ARI):
                #     temp_A = ARI


                # NMI = metrics.normalized_mutual_info_score(ground_community_labels,detected_community_labels)
                # if (temp_N < NMI):
                #     temp_N = NMI


                # pur = metrics.accuracy_score(ground_community_labels,detected_community_labels)
                # if (temp_P < pur):
                #     temp_P = pur

                
                # # auc = metrics.roc_auc_score(ground_community_labels,detected_community_labels)
                # # total_AUC = total_AUC + auc
                # # if (Max_AUC < auc):
                # #     Max_AUC = auc
                # # if (Min_AUC > auc):
                # #     Min_AUC = auc
                

                # modularity = nx_comm.modularity(G,communities)
                # if (temp_M < modularity):
                #     temp_M = modularity

                
                acc = metrics.accuracy_score(ground_community_labels,detected_community_labels)
                if (temp_ACC < acc):
                    temp_ACC = acc

        if Max_F < temp_F:
            Max_F = temp_F
        if Min_F > temp_F:
            Min_F = temp_F
        total_F = total_F + temp_F
        F_standard = F_standard + (F_average-temp_F)*(F_average-temp_F)

        # total_A = total_A + temp_A
        # if (Max_A < temp_A):
        #     Max_A = temp_A
        # if (Min_A > temp_A):
        #     Min_A = temp_A

        # total_N = total_N + temp_N
        # if (Max_N < temp_N):
        #     Max_N = temp_N
        # if (Min_N > temp_N):
        #     Min_N = temp_N

        # total_P = total_P + temp_P
        # if (Max_P < temp_P):
        #     Max_P = temp_P
        # if (Min_P > temp_P):
        #     Min_P = temp_P

        # total_M = total_M + temp_M
        # if (Max_M < temp_M):
        #     Max_M = temp_M
        # if (Min_M > temp_M):
        #     Min_M = temp_M
        
        total_ACC = total_ACC + temp_ACC
        if (Max_ACC < temp_ACC):
            Max_ACC = temp_ACC
        if (Min_ACC > temp_ACC):
            Min_ACC = temp_ACC
        ACC_standard = ACC_standard + (ACC_average-temp_ACC)*(ACC_average-temp_ACC)


    f = open('/home/ssx/code/CCNMF/standard/F_score/'+filename,'w')
    f.write(str(math.sqrt(F_standard/20)))
    f.close()

    f = open('/home/ssx/code/CCNMF/standard/ACC/'+filename,'w')
    f.write(str(math.sqrt(ACC_standard/20)))
    f.close()


    # f = open('/home/ssx/code/CCNMF/DATASET/F_score/'+filename,'w')
    # f.write(str(total_F/20)+'       '+str(Max_F)+'          '+str(Min_F))
    # f.close()

    # f = open('/home/ssx/code/CCNMF/DATASET/NMI/'+filename,'w')
    # f.write(str(total_N/20)+'       '+str(Max_N)+'          '+str(Min_N))
    # f.close()

    # f = open('/home/ssx/code/CCNMF/DATASET/ARI/'+filename,'w')
    # f.write(str(total_A/20)+'       '+str(Max_A)+'          '+str(Min_A))
    # f.close()

    # f = open('/home/ssx/code/CCNMF/DATASET/purity/'+filename,'w')
    # f.write(str(total_P/20)+'       '+str(Max_P)+'          '+str(Min_P))
    # f.close()

    # # f = open('/home/ssx/code/karateclub_algorithm/DANMF/DATASET/AUC/texas','w')
    # # f.write(str(total_AUC/20)+'       '+str(Max_AUC)+'          '+str(Min_AUC))
    # # f.close()

    # f = open('/home/ssx/code/CCNMF/DATASET/modularity/'+filename,'w')
    # f.write(str(total_M/20)+'       '+str(Max_M)+'          '+str(Min_M))
    # f.close()

    # f = open('/home/ssx/code/CCNMF/DATASET/ACC/'+filename,'w')
    # f.write(str(total_ACC/20)+'       '+str(Max_ACC)+'          '+str(Min_ACC))
    # f.close()
        


    