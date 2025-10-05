#Use GPT to provide visualization suggestions and a framework for plotting code
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#Predicted and True values comperasion
def PredictedTrue(y_test, y_pred, featurization):
    #Figure of comperaing Predicted and True values
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Predicted & True Lipophilicity of {featurization}")

    # Save figure as PNG
    #Ask GPT how to visualize the results for both Morgan fingerprints 
    #and molecular descriptors without overwriting
    plt.savefig(f"Regression/Figure/predicted_true_{featurization}.png", dpi=300)
    plt.show()
