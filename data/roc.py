import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import auc, roc_auc_score, roc_curve

def roc(y_test, y_score, title, posLabel="ATRIAL_FIBRILLATION"):

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
        '/home/rkaufman/workspace/afib_detection/results/assets/roc.png'
    )