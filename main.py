import glob
import warnings
import re
import os
import sys
from train.train_encoder import train_encoder
from train.train_wgan import train_wgan
from torch.utils.data import DataLoader
from utils.operations import *
from utils.plots import *
from utils.graph_processing import *
from data.graph_sampler import GraphSampler
from data.load_data import read_graphfile
from sklearn.model_selection import train_test_split
from models.gan import Generator, Discriminator
from models.encoder import Encoder
from utils.parser import parse_arguments
from train.anomaly_detection import detect_anomalies

def prepare_data(datadir, args):

    if args.DS.startswith("Tox21"):
        graphs_train = read_graphfile(datadir, args.DS + '_training', args.quiet, max_nodes=0)
        graphs_test = read_graphfile(datadir, args.DS + '_testing', args.quiet, max_nodes=0)
        graphs = graphs_train + graphs_test
        graphs_train_labels = [graph.graph['label'] for graph in graphs_train]
    else:
        graphs = read_graphfile(datadir, args.DS, args.quiet, max_nodes=0)  
        graphs_train, graphs_test = train_test_split(graphs, test_size=0.2, random_state=42)
        graphs_train_labels = [graph.graph['label'] for graph in graphs_train]

    # filtering out abnormal graphs for training
    graphs_train = [graph for graph, label in zip(graphs_train, graphs_train_labels) if label == 0]

    max_num_nodes = max([G.number_of_nodes() for G in graphs])
    dataset_sampler_train = GraphSampler(graphs_train, features=args.feat, normalize=False, max_num_nodes=max_num_nodes)
    dataset_sampler_test = GraphSampler(graphs_test, features=args.feat, normalize=False, max_num_nodes=max_num_nodes)

    data_loader_train = DataLoader(dataset_sampler_train, shuffle=True, batch_size=args.batch_size)
    data_loader_test = DataLoader(dataset_sampler_test, shuffle=True, batch_size=args.batch_size)

    return max_num_nodes, dataset_sampler_train, data_loader_train, data_loader_test

def find_latest_checkpoint(checkpoint_dir, prefix):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, f'{prefix}_checkpoint_*.pth'))
    
    if not checkpoint_files:
        return None
    
    def extract_epoch(filename):
        match = re.search(r'_checkpoint_(\d+)\.pth', filename)
        return int(match.group(1)) if match else -1
    
    # find file with highest epoch
    latest_file = max(checkpoint_files, key=extract_epoch)
    return latest_file

def load_latest_checkpoints(G, D, g_optimizer, d_optimizer, checkpoint_dir):
    # load generator checkpoint
    g_checkpoint_path = os.path.join(checkpoint_dir, 'G_checkpoint_final.pth')
    if not os.path.exists(g_checkpoint_path):
        g_checkpoint_path = find_latest_checkpoint(checkpoint_dir, 'G')
        if g_checkpoint_path:
            print(f"Loading latest G checkpoint: {g_checkpoint_path}")
        else:
            raise FileNotFoundError("No G checkpoint found!")
    else:
        print(f"Loading final G checkpoint: {g_checkpoint_path}")

    _, _ = load_checkpoint(G, g_optimizer, g_checkpoint_path)

    # load discriminator checkpoint
    d_checkpoint_path = os.path.join(checkpoint_dir, 'D_checkpoint_final.pth')
    if not os.path.exists(d_checkpoint_path):
        d_checkpoint_path = find_latest_checkpoint(checkpoint_dir, 'D')
        if d_checkpoint_path:
            print(f"Loading latest D checkpoint: {d_checkpoint_path}")
        else:
            raise FileNotFoundError("No D checkpoint found!")
    else:
        print(f"Loading final D checkpoint: {d_checkpoint_path}")

    _, _ = load_checkpoint(D, d_optimizer, d_checkpoint_path)

if __name__ == "__main__":

    warnings.filterwarnings("ignore", message=".*Attempting to run cuBLAS, but there was no current CUDA context.*")

    args = parse_arguments()

    repo_dir = os.path.abspath(os.path.dirname(sys.argv[0]))

    DATADIR = os.path.join(repo_dir, 'datasets/')
    DS = 'NCI1'
    CHECKPOINT_DIR = 'checkpoints/'
    PLOTS_DIR = 'plots/'
    LOGS_DIR = 'logs/'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_num_nodes, dataset_sampler_train, data_loader_train, data_loader_test = prepare_data(DATADIR, args)

    # pre-defined parameters
    G_CONV_DIM = [128,256,512]
    D_CONV_DIM = [dataset_sampler_train.feat_dim, [128, 64]]
    D_AGGR_DIM = 128
    D_LINEAR_DIM = [128, 64]
    Z_DIM = 8
    BETA1 = 0.5
    BETA2 = 0.99
    LAMBDA_GP = 10  # penalty coefficient
    ENCODER_SAVE_ITERS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]
    WGAN_SAVE_ITERS = [300, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 100000, 150000]
    E_LINEAR_DIM = [128, 64]
    KAPPA = 1.0

    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)
    
    if not os.path.exists(PLOTS_DIR) and args.plot_loss:
        os.mkdir(PLOTS_DIR)

    if not os.path.exists(LOGS_DIR):
        os.mkdir(LOGS_DIR)

    G = Generator(G_CONV_DIM, Z_DIM, max_num_nodes, dataset_sampler_train.feat_dim, args.dropout).to(DEVICE)
    D = Discriminator(D_CONV_DIM, D_AGGR_DIM, D_LINEAR_DIM, args.dropout).to(DEVICE)
    E = Encoder(dataset_sampler_train.feat_dim, E_LINEAR_DIM, Z_DIM, args.dropout).to(DEVICE)
    g_optimizer = torch.optim.Adam(G.parameters(), args.g_lr, [BETA1, BETA2])
    d_optimizer = torch.optim.Adam(D.parameters(), args.d_lr, [BETA1, BETA2])
    e_optimizer =torch.optim.Adam(E.parameters(), args.e_lr, [BETA1, BETA2])

    G.apply(weights_init)
    D.apply(weights_init)
    E.apply(weights_init)
    
    if args.resume_training:
        start_epoch, _ = load_checkpoint(G, g_optimizer, os.path.join(CHECKPOINT_DIR, 'G_checkpoint_latest.pth'))
        _, _ = load_checkpoint(D, d_optimizer, os.path.join(CHECKPOINT_DIR, 'D_checkpoint_latest.pth'))
    else:
        start_epoch = 0

    if not args.wgan_pretrained:
        losses = train_wgan(G, D, E, g_optimizer, d_optimizer, e_optimizer, start_epoch, args, data_loader_train, 
                   data_loader_test, DEVICE, Z_DIM, WGAN_SAVE_ITERS, CHECKPOINT_DIR, LAMBDA_GP, KAPPA)

    load_latest_checkpoints(G, D, g_optimizer, d_optimizer, CHECKPOINT_DIR)

    if not args.encoder_pretrained:
        train_encoder(G, D, E, e_optimizer, args.encoder_epochs, data_loader_train, DEVICE, KAPPA, args.gumbell_type, CHECKPOINT_DIR, args)

    _, _ = load_checkpoint(E, e_optimizer, os.path.join(CHECKPOINT_DIR, 'E_checkpoint_final.pth'))

    detect_anomalies(G, E, D, data_loader_test, KAPPA, DEVICE, args)

    scores = load_scores_from_file('scores.pkl')

    max_epoch = max(scores, key=scores.get) 
    max_score = scores[max_epoch]  

    with open("logs/output.txt", "w") as f:

        f.write("Parameters:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        
        f.write(f"\nMax score at epoch {max_epoch}: {max_score}\n")

    if args.plot_loss:
        plot_losses(losses, PLOTS_DIR)

