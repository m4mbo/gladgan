import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--DS', type=str, default="NCI1", help='Name of dataset for trainig.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--feat', type=str, default='deg', help='Type of node features to be used ("deg" for plain graphs, "default" for attributed graphs)')
    
    parser.add_argument('--wgan_epochs', type=int, default=10000, help='Number of epochs for WGAN training.')
    parser.add_argument('--epochs_decay', type=int, default=5000, help='Number of epochs over which to decay the learning rate.')
    parser.add_argument('--lr_update_step', type=int, default=1000, help='Step size for updating learning rates.')
    parser.add_argument('--patience', type=int, default=200, help='Number of epochs to wait for improvement before stopping.')
    parser.add_argument('--resume_training', type=bool, default=False, help='Whether to resume training from a checkpoint.')
    
    parser.add_argument('--g_lr', type=float, default=2e-4, help='Learning rate for the generator.')
    parser.add_argument('--d_lr', type=float, default=2e-4, help='Learning rate for the discriminator.')
    
    parser.add_argument('--e_lr', type=float, default=2e-4, help='Learning rate for the encoder.')
    parser.add_argument('--encoder_epochs', type=int, default=1000, help='Number of epochs for the encoder.')

    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--gumbell_type', type=str, default='hard-gumbell', help='Type of Gumbel distribution used.')
    parser.add_argument('--n_critic', type=int, default=5, help='Number of discriminator updates per generator update.')

    parser.add_argument('--wgan_pretrained', action='store_true', help='Use pretrained WGAN model.')
    parser.add_argument('--encoder_pretrained', action='store_true', help='Use pretrained encoder model.')

    parser.add_argument('--early_stop', action='store_true', help='Early stop WGAN training.')
    parser.add_argument('--plot_loss', action='store_true', help='Plot discriminator and generator losses over epochs.')
    parser.add_argument('--quiet', action='store_true', help='Do not print to console.')


    return parser.parse_args()
