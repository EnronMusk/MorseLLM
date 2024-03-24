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

#Plots only train data by batch.
def createLossPlot(losses : list) -> None:

    #Plots training summary results on accuracy
    plt.plot(losses, color='red', label="train")
    plt.grid(alpha=0.3)
    plt.ylabel('C-E loss',fontweight='bold')
    plt.xlabel('batch',fontweight='bold')
    plt.title("Training Loss By Batch")
    plt.legend()
    plt.show()