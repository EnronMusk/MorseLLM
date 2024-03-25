import matplotlib.pyplot as plt

#Plots both test and train data by epoch.
def createDualLossPlot(train : list, test : list) -> None:

    #Plots training summary results on accuracy
    plt.plot(test, color='red', label='test', marker = 'o')
    plt.plot(train, color='red', label='train', alpha = 0.5, marker = 'o')
    plt.grid(alpha=0.3)
    plt.ylabel('C-E loss',fontweight='bold')
    plt.xlabel('epoch',fontweight='bold')
    plt.title("Training & Test Loss By Epoch")
    plt.legend()
    plt.show()

#Plots only train data by batch. Both loss and accuracy.
def createDualTrainPlot(train_loss : list, train_acc : list) -> None:

    #Plots training summary results on accuracy
    plt.plot(train_acc, color='red', label='train accuracy')
    plt.plot(train_loss, color='red', label='train loss', alpha = 0.5)
    plt.grid(alpha=0.3)
    plt.ylabel('accuracy',fontweight='bold')
    plt.xlabel('batch',fontweight='bold')
    plt.title("Training Loss & Accuracy By Batch")
    plt.legend()
    plt.show()

#Plots both test and train data by epoch.
def createDualAccuracyPlot(train : list, test : list) -> None:

    #Plots training summary results on accuracy
    plt.plot(test, color='red', label='test', marker = 'o')
    plt.plot(train, color='red', label='train', alpha = 0.5, marker = 'o')
    plt.grid(alpha=0.3)
    plt.ylabel('accuracy',fontweight='bold')
    plt.xlabel('epoch',fontweight='bold')
    plt.title("Training & Test Accuracy By Epoch")
    plt.legend()
    plt.show()
