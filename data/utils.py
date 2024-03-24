import matplotlib.pyplot as plt

def createAccuracyPlot(losses, type : str):

    #Plots training summary results on accuracy
    plt.plot(losses, color='red', label=type)
    plt.grid(alpha=0.3)
    plt.ylabel('accuracy',fontweight='bold')
    plt.xlabel('batch',fontweight='bold')
    plt.title("Training Loss By Batch")
    plt.legend()
    plt.show()
