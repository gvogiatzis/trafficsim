import sys
import os.path
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import typer
from typing import Optional as Opt, List, Tuple
from typing_extensions import Annotated as Ann

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, add_completion=False)

@app.command()
def main(flowmat_fname: Ann[str, typer.Argument(help="The filename of the flow matrix for the traffic simulation in question. The flow matrix is a structure that links centralized actions with particular lanes that are shown the green light. It is a matrix of imensions num_actions x num_lanes and contains a 1 if the particular action green-lights a particular lane, and 0 otherwise.")],
         
         num_epochs:Ann[Opt[int], typer.Option(help='The number of epochs to train the model',default=...)]=50,

         dataset_size:Ann[Opt[int], typer.Option(help='The size of the artificial dataset to be used for training')] = 10000,

         network_layers:Ann[Opt[str], typer.Option(help="A string of integers separated by 'x' chars, denoting the size and number of hidden layers of the network architecture. E.g. '512x512x256' would create three hidden layers of dims 512,512 and 256. Ignored if 'in_model_fname' option is set.")] = "1024x1024",

         out_model_fname: Ann[Opt[str], typer.Option(help="If set, gives the filename to use when saving the trained model.")] = "sup_model.pt"
         ):
    

    from rl import SupervisedLearningPretrainer

    import matplotlib.pyplot as plt
    import torch
    import tqdm

    network_layers = [int(s) for s in network_layers.split("x") if s.isnumeric()]

    from rl.models import MLPnet, loadModel, saveModel, loadModel_from_dict
    import matplotlib.pyplot as plt

    dataset = []
    W = np.loadtxt(flowmat_fname)

    output_dim, input_dim = W.shape
    Wtorch = torch.tensor(W,dtype=torch.float32)
    print("Generating dataset:")
    for i in tqdm.tqdm(range(dataset_size)):
        x = 50*torch.randn(size=(input_dim,), dtype=torch.float32)
        a = np.argsort(Wtorch @ x)
        t = a[0]
        dataset.append((x, t))
    model = MLPnet(input_dim,*network_layers,output_dim)
    trainer = SupervisedLearningPretrainer(dataset, model)

    stats = trainer.train_epochs(num_epochs)

    saveModel(model, out_model_fname)
    plt.figure()
    plt.title("Loss")
    plt.plot(stats["training_loss_series"], 'r-')
    plt.plot(stats["test_loss_series"], 'b-')
    plt.figure()
    plt.title("Accuracy")
    plt.plot(stats["training_acc_series"], 'r-')
    plt.plot(stats["test_acc_series"], 'b-')
    plt.show() 

if __name__ == "__main__":
    app()