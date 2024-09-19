import matplotlib.pyplot as plt
import os


def plot_losses(losses, save_dir='plots/'):

    gloss = losses['G']
    dloss = losses['D']

    n_epochs = len(gloss)
    epochs = range(1, n_epochs + 1)

    plt.figure(figsize=(10, 6))
    
    # generator loss
    plt.plot(epochs, gloss, 'b-', label='Generator Loss')  
    
    # discriminator loss
    plt.plot(epochs, dloss, 'r-', label='Discriminator Loss') 
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Evolution')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, 'loss.png'))
