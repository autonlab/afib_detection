import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import auc, roc_auc_score, roc_curve

def roc(y_test, y_score, title, posLabel="ATRIAL_FIBRILLATION", dstTitle='roc.png'):

    lw = 2 #line width
    plt.figure()
    # Case for multiple scores or one
    if isinstance(y_score[0], float):
        fpr, tpr, whoknowswhatthisis = roc_curve(y_test, y_score, pos_label=posLabel)
        rocauc = auc(fpr, tpr)
        #then single plot
        plt.plot(
            fpr,
            tpr,
            color='darkorange',
            label=f'ROC curve (auc = {rocauc:.2})'
        )
    else:
        #then multiple plots
        colors = cycle(['aqua', 'yellowgreen', 'cornflowerblue', 'purple', 'orange'])
        for i, (scoreDescriptor, scores) in enumerate(y_score):
            fpr, tpr, whoknowswhatthisis = roc_curve(y_test[i], scores, pos_label=posLabel)
            rocauc = auc(fpr, tpr)
            #then single plot
            plt.plot(
                fpr,
                tpr,
                color=next(colors),
                label=f'{scoreDescriptor} (auc = {rocauc:.2})'
            )

    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(
        '/home/rkaufman/workspace/afib_detection/results/assets/' + dstTitle
    )


# Plot the three ROC graphs
# op=output prefix
import os
import numpy as np
import pandas as pd
# from matplotlib.patches import Polygon
from shapely.geometry import Polygon
from descartes import PolygonPatch
def three_roc_plot_mult(title='ROC Curves', op='', predictionsOutput=None, model="rf.model", predictionsFile="~/workspace/afib_detection/testset_binary_norm.csv"):
    # ! ./random_forest load {model} option predict testds {predictionsFile}
    if (predictionsOutput is None):
        os.system(f"./random_forest load {model} option predict testds {predictionsFile}")

    # ! sed -i "1s/.*/A0,A1,true_output/" {op}predictions.csv
        os.system(f'sed -i "1s/.*/A0,A1,true_output/" {op}predictions.csv')
    else:
        os.system(f'sed -i "1s/.*/A0,A1,true_output/" {op}predictions.csv')

        
        # ! ./random_forest option roc ds {op}predictions.csv
        # ! python ../afib_detection/2class_process_roc_stds.py A1_roc
        # ! python ../afib_detection/2class_process_roc_stds.py A2_roc
    os.system(f"./random_forest option roc ds {op}predictions.csv")
    os.system(f"python ../afib_detection/data/2class_process_roc_stds.py A1_roc")
    os.system(f"python ../afib_detection/data/2class_process_roc_stds.py A2_roc")

    ## three roc curves in one plot

    ## ROC curves of a random classifier, for reference
    RANDOM_FP = np.arange(0, 1., 0.01)
    RANDOM_TP = np.arange(0, 1., 0.01)
    RANDOM_FP[0] = 1e-4
    RANDOM_TP[0] = 1e-4
    color='blue'

    ## Mean and confidence bands of FPR V.S. TPR curve
#     a1 = pd.read_csv('./'+op+'A1_roc_std.csv')
    a1 = pd.read_csv('./A1_roc_std.csv')
    conf_UB = a1[['FP1UB', 'TP1UB']].copy()
    conf_UB.columns = ['X', 'Y']
    conf_LB = a1[['FP1LB', 'TP1LB']].iloc[range(a1.shape[0]-1, -1, -1)]
    conf_LB.columns = ['X', 'Y']
    conf = pd.concat((conf_UB, conf_LB))

    fig = plt.figure(figsize=(15, 4))

    ## FPR V.S. TPR curve
    ax = fig.add_subplot(1, 3, 1)
    plt.plot(a1['FP1'], a1['TP1'], color=color, label='RandomForest')
    poly = Polygon(list(zip(conf['X'], conf['Y'])))
    poly = PolygonPatch(poly, linewidth=0, fc=color, alpha=0.4)
    ax.add_patch(poly)
    plt.plot(RANDOM_FP, RANDOM_TP, color='black', linestyle='--')
    plt.xlabel('FPR', fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylabel('TPR', fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(linestyle='--')

    ## FPR (log-scale) V.S. TPR curve
    ax = fig.add_subplot(1, 3, 2)  
    plt.plot(a1['FP1'], a1['TP1'], color=color, label='RandomForest') 
    poly = Polygon(list(zip(conf['X'], conf['Y'])))
    poly = PolygonPatch(poly, linewidth=0, fc=color, alpha=0.4)
    ax.add_patch(poly)
    plt.plot(RANDOM_FP, RANDOM_TP, color='black', linestyle='--')
    plt.xlabel('FPR', fontsize=14)
    plt.xscale('log',base=10) 
    plt.xlim(0.6*1e-4, 1.6)
    plt.xticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0], ['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '1.0'])
    plt.ylabel('TPR', fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(linestyle='--')
    plt.title(title, fontsize=15, pad=10)

    ## here A2 treats the 2nd class 'usa' as positive and the 1st class 'asia' as negative 
    a2 = pd.read_csv('./A2_roc_std.csv')
    conf_UB = a2[['FP1UB', 'TP1UB']].copy()
    conf_UB.columns = ['X', 'Y']
    conf_LB = a2[['FP1LB', 'TP1LB']].iloc[range(a2.shape[0]-1, -1, -1)]
    conf_LB.columns = ['X', 'Y']
    conf = pd.concat((conf_UB, conf_LB))

    ax = fig.add_subplot(1, 3, 3) 
    plt.plot(a2['FP1'], a2['TP1'], color=color, label='RandomForest') #plt.plot(a2['FP1'], a2['TP1'], color='C0') 
    poly = Polygon(list(zip(conf['X'], conf['Y'])))
    poly = PolygonPatch(poly, linewidth=0, fc=color, alpha=0.4) #poly = PolygonPatch(poly, linewidth=0, fc='C0', alpha=0.4)
    ax.add_patch(poly)
    plt.plot(RANDOM_FP, RANDOM_TP, color='black', linestyle='--')
    plt.xlabel('FNR', fontsize=14)
    plt.xscale('log',base=10) 
    plt.xlim(0.6*1e-4, 1.6)
    plt.xticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0], ['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '1.0'])
    plt.ylabel('TNR', fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(linestyle='--')

    # three_roc_plot(fig=p, op="lr", predictionsOutput='yup', title="LogisticRegression ROCs")
    op = 'lr'; predictionsOutput='yup'; title='LogisticRegression ROCs'

    if (predictionsOutput is None):
        os.system(f"./random_forest load {model} option predict testds {predictionsFile}")

    # ! sed -i "1s/.*/A0,A1,true_output/" {op}predictions.csv
        os.system(f'sed -i "1s/.*/A0,A1,true_output/" {op}predictions.csv')
    else:
        os.system(f'sed -i "1s/.*/A0,A1,true_output/" {op}predictions.csv')

        
        # ! ./random_forest option roc ds {op}predictions.csv
        # ! python ../afib_detection/2class_process_roc_stds.py A1_roc
        # ! python ../afib_detection/2class_process_roc_stds.py A2_roc
    os.system(f"./random_forest option roc ds {op}predictions.csv")
    os.system(f"python ../afib_detection/data/2class_process_roc_stds.py A1_roc")
    os.system(f"python ../afib_detection/data/2class_process_roc_stds.py A2_roc")

    ## three roc curves in one plot

    ## ROC curves of a random classifier, for reference
    RANDOM_FP = np.arange(0, 1., 0.01)
    RANDOM_TP = np.arange(0, 1., 0.01)
    RANDOM_FP[0] = 1e-4
    RANDOM_TP[0] = 1e-4

    ## Mean and confidence bands of FPR V.S. TPR curve
#     a1 = pd.read_csv('./'+op+'A1_roc_std.csv')
    a1 = pd.read_csv('./A1_roc_std.csv')
    conf_UB = a1[['FP1UB', 'TP1UB']].copy()
    conf_UB.columns = ['X', 'Y']
    conf_LB = a1[['FP1LB', 'TP1LB']].iloc[range(a1.shape[0]-1, -1, -1)]
    conf_LB.columns = ['X', 'Y']
    conf = pd.concat((conf_UB, conf_LB))

    color = 'orange'
    ## FPR V.S. TPR curve
    ax = plt.subplot(1, 3, 1)
    plt.plot(a1['FP1'], a1['TP1'], color=color, label='LogisticRegressor')
    poly = Polygon(list(zip(conf['X'], conf['Y'])))
    poly = PolygonPatch(poly, linewidth=0, fc=color, alpha=0.4)
    ax.add_patch(poly)
    plt.plot(RANDOM_FP, RANDOM_TP, color='black', linestyle='--')
    plt.xlabel('FPR', fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylabel('TPR', fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(linestyle='--')

    ## FPR (log-scale) V.S. TPR curve
    ax = plt.subplot(1, 3, 2)  
    plt.plot(a1['FP1'], a1['TP1'], color=color, label='LogisticRegressor') 
    poly = Polygon(list(zip(conf['X'], conf['Y'])))
    poly = PolygonPatch(poly, linewidth=0, fc=color, alpha=0.4)
    ax.add_patch(poly)
    plt.plot(RANDOM_FP, RANDOM_TP, color='black', linestyle='--')
    plt.xlabel('FPR', fontsize=14)
    plt.xscale('log',base=10) 
    plt.xlim(0.6*1e-4, 1.6)
    plt.xticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0], ['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '1.0'])
    plt.ylabel('TPR', fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(linestyle='--')
    plt.title('ROC Comparison', fontsize=15, pad=10)

    ## here A2 treats the 2nd class 'usa' as positive and the 1st class 'asia' as negative 
    a2 = pd.read_csv('./A2_roc_std.csv')
    conf_UB = a2[['FP1UB', 'TP1UB']].copy()
    conf_UB.columns = ['X', 'Y']
    conf_LB = a2[['FP1LB', 'TP1LB']].iloc[range(a2.shape[0]-1, -1, -1)]
    conf_LB.columns = ['X', 'Y']
    conf = pd.concat((conf_UB, conf_LB))

    ax = plt.subplot(1, 3, 3) 
    plt.plot(a2['FP1'], a2['TP1'], color=color, label='LogisticRegressor') #plt.plot(a2['FP1'], a2['TP1'], color='C0') 
    poly = Polygon(list(zip(conf['X'], conf['Y'])))
    poly = PolygonPatch(poly, linewidth=0, fc=color, alpha=0.4) #poly = PolygonPatch(poly, linewidth=0, fc='C0', alpha=0.4)
    ax.add_patch(poly)
    plt.plot(RANDOM_FP, RANDOM_TP, color='black', linestyle='--')
    plt.xlabel('FNR', fontsize=14)
    plt.xscale('log',base=10) 
    plt.xlim(0.6*1e-4, 1.6)
    plt.xticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0], ['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '1.0'])
    plt.ylabel('TNR', fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(linestyle='--')
    plt.subplots_adjust(wspace=0.4)
    # plt.show()
    plt.gcf().set_size_inches(12, 5)

    from sklearn.metrics import confusion_matrix
    df = pd.read_csv('~/Downloads/phillips_alerts_final.csv')
    print(df[df['is_afib']==1])
    print(confusion_matrix(df['final_label'], df['is_afib']))
    cnf_matrix = confusion_matrix(df['final_label'], df['is_afib'])
    print(cnf_matrix)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)


    FP = FP.astype(float)[1]
    FN = FN.astype(float)[1]
    TP = TP.astype(float)[1]
    TN = TN.astype(float)[1]
    print(FP, TP, FN, TN)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    print(TPR, FPR)
    plt.subplot(1, 3, 1)
    plt.plot([FPR], [TPR], color='GREEN', marker="1", label="Phillips")
    plt.subplot(1, 3, 2)
    plt.plot([FPR], [TPR], color='GREEN', marker="1", label="Phillips")
    plt.subplot(1, 3, 3)
    plt.plot([FNR], [TNR], color='GREEN', marker="1", label="Phillips")
    plt.legend()
    plt.savefig(
        f'/home/romman/workspace/afib_detection/results/assets/roc.png'
    )
def three_roc_plot(title='ROC Curves', op='', predictionsOutput=None, model="rf.model", predictionsFile="~/workspace/afib_detection/testset_binary_norm.csv"):
    # p = three_roc_plot(title="Auton RandomForest ROCs")
    # ! ./random_forest load {model} option predict testds {predictionsFile}
    # three_roc_plot(fig=p, op="lr", predictionsOutput='yup', title="LogisticRegression ROCs")

    if (predictionsOutput is None):
        os.system(f"./random_forest load {model} option predict testds {predictionsFile}")

    # ! sed -i "1s/.*/A0,A1,true_output/" {op}predictions.csv
        os.system(f'sed -i "1s/.*/A0,A1,true_output/" {op}predictions.csv')
    else:
        os.system(f'sed -i "1s/.*/A0,A1,true_output/" {op}predictions.csv')

        # ! ./random_forest option roc ds {op}predictions.csv
        # ! python ../afib_detection/2class_process_roc_stds.py A1_roc
        # ! python ../afib_detection/2class_process_roc_stds.py A2_roc
    os.system(f"./random_forest option roc ds {op}predictions.csv")
    os.system(f"python ../afib_detection/data/2class_process_roc_stds.py A1_roc")
    os.system(f"python ../afib_detection/data/2class_process_roc_stds.py A2_roc")

    ## three roc curves in one plot

    ## ROC curves of a random classifier, for reference
    RANDOM_FP = np.arange(0, 1., 0.01)
    RANDOM_TP = np.arange(0, 1., 0.01)
    RANDOM_FP[0] = 1e-4
    RANDOM_TP[0] = 1e-4

    ## Mean and confidence bands of FPR V.S. TPR curve
#     a1 = pd.read_csv('./'+op+'A1_roc_std.csv')
    a1 = pd.read_csv('./A1_roc_std.csv')
    conf_UB = a1[['FP1UB', 'TP1UB']].copy()
    conf_UB.columns = ['X', 'Y']
    conf_LB = a1[['FP1LB', 'TP1LB']].iloc[range(a1.shape[0]-1, -1, -1)]
    conf_LB.columns = ['X', 'Y']
    conf = pd.concat((conf_UB, conf_LB))

    if (not fig):
        fig = plt.figure(figsize=(15, 4))

    ## FPR V.S. TPR curve
    ax = fig.add_subplot(1, 3, 1)
    plt.plot(a1['FP1'], a1['TP1'], color='C0')
    poly = Polygon(list(zip(conf['X'], conf['Y'])))
    poly = PolygonPatch(poly, linewidth=0, fc='C0', alpha=0.4)
    ax.add_patch(poly)
    plt.plot(RANDOM_FP, RANDOM_TP, color='black', linestyle='--')
    plt.xlabel('FPR', fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylabel('TPR', fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(linestyle='--')

    ## FPR (log-scale) V.S. TPR curve
    ax = fig.add_subplot(1, 3, 2)  
    plt.plot(a1['FP1'], a1['TP1'], color='C0') 
    poly = Polygon(list(zip(conf['X'], conf['Y'])))
    poly = PolygonPatch(poly, linewidth=0, fc='C0', alpha=0.4)
    ax.add_patch(poly)
    plt.plot(RANDOM_FP, RANDOM_TP, color='black', linestyle='--')
    plt.xlabel('FPR', fontsize=14)
    plt.xscale('log',base=10) 
    plt.xlim(0.6*1e-4, 1.6)
    plt.xticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0], ['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '1.0'])
    plt.ylabel('TPR', fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(linestyle='--')
    plt.title(title, fontsize=15, pad=10)

    ## here A2 treats the 2nd class 'usa' as positive and the 1st class 'asia' as negative 
    a2 = pd.read_csv('./A2_roc_std.csv')
    conf_UB = a2[['FP1UB', 'TP1UB']].copy()
    conf_UB.columns = ['X', 'Y']
    conf_LB = a2[['FP1LB', 'TP1LB']].iloc[range(a2.shape[0]-1, -1, -1)]
    conf_LB.columns = ['X', 'Y']
    conf = pd.concat((conf_UB, conf_LB))

    ax = fig.add_subplot(1, 3, 3) 
    plt.plot(a2['FP1'], a2['TP1'], color='C0') #plt.plot(a2['FP1'], a2['TP1'], color='C0') 
    poly = Polygon(list(zip(conf['X'], conf['Y'])))
    poly = PolygonPatch(poly, linewidth=0, fc='C0', alpha=0.4) #poly = PolygonPatch(poly, linewidth=0, fc='C0', alpha=0.4)
    ax.add_patch(poly)
    plt.plot(RANDOM_FP, RANDOM_TP, color='black', linestyle='--')
    plt.xlabel('FNR', fontsize=14)
    plt.xscale('log',base=10) 
    plt.xlim(0.6*1e-4, 1.6)
    plt.xticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0], ['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '1.0'])
    plt.ylabel('TNR', fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(linestyle='--')

    plt.subplots_adjust(wspace=0.4)
    # plt.show()
    if (predictionsOutput):
        plt.gcf().set_size_inches(12, 5)
        plt.savefig(
            f'/home/romman/workspace/afib_detection/results/assets/roc_{title}.png'
        )
    return fig

if __name__=='__main__':
    p = three_roc_plot_mult(title="Auton RandomForest ROCs")
#    three_roc_plot(fig=p, op="lr", predictionsOutput='yup', title="LogisticRegression ROCs")