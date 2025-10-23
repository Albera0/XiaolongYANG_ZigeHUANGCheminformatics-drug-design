#Use GPT to provide visualization suggestions and a framework for plotting code
import matplotlib.pyplot as plt
import os

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

#Visualize training & validation loss
def LossVis(epoch, train_losses, valid_losses, featurization):
    print(epoch)
    print(train_losses)
    print(valid_losses)
    #Figure of training & validation loss
    plt.plot(epoch, train_losses, label='Train')
    plt.plot(epoch, valid_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.title('Model Loss')

    # Save figure as PNG
    plt.savefig(f"Regression/Figure/train_curve_{featurization}.png", dpi=300)
    plt.show()