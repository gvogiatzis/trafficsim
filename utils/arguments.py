import argparse
import sys

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparsers = parser.add_subparsers(title="commands", help="traffic agent commands:", dest='cmd', required=True)

    # The options that are common to all modes of operation (training and testing)
    sharedoptionsparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,add_help=False)
    sharedoptionsparser.add_argument('--net','-n', type=str, required=True, help='the filename of the sumo network to use')
    sharedoptionsparser.add_argument('--num-episodes', type=int, default=50, help='the number of episodes to train the agent')
    sharedoptionsparser.add_argument('--spawn-rate', type=float, default=0.01, help='The average rate at which new vehicles are being spawned')
    sharedoptionsparser.add_argument('--episode-length', type=int, default=100, help='the number of timesteps for each episode')
    sharedoptionsparser.add_argument('--sumo-timestep', type=int, default=20, help='the number of sumo timesteps between RL timesteps (i.e. when actions are taken)')
    sharedoptionsparser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    """ some more arguments on the sumo simulation """
    sharedoptionsparser.add_argument('--use-gui', help="If set, performs the simulation using the sumo-gui command, i.e. with a graphical interface", action='store_true')
    sharedoptionsparser.add_argument('--plot-reward', help="If set, shows a plot of the reward time-series", action='store_true')
    sharedoptionsparser.add_argument('--seed', type=int, default=None, help='Random seed to be passed to sumo. This guarantees reproducible results.')

    parser_train = subparsers.add_parser("train", description='Train a traffic control agent using DQN', help="Train a RL agent to perform traffic control on SUMO using the DQN algorithm.", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[sharedoptionsparser])

    parser_train.add_argument('--batch-size', type=int, default=128, help='the sample batch size for optimizing the models')
    parser_train.add_argument('--replay-buffer-size', type=int, default=int(5000), help='the size of the buffer')
    parser_train.add_argument('--target-update-step', type=int, default=10, help='the step when the target network should be updated')
    parser_train.add_argument('--random-eps', type=float, default=0.1, help='eps for epsilon-greedy action exploration')
    parser_train.add_argument('--gamma', type=float, default=0.99, help='the discount factor for training models')
    parser_train.add_argument('--lr', type=float, default=0.001, help='the learning rate of the networks')
    parser_train.add_argument('--input', type=str, default=None, help='filename of a previously saved agent model which will be used as a starting point for further training')
    parser_train.add_argument('--output', type=str, default="agentmodel.pt", help='filename to use for the trained agent model')
    parser_train.add_argument('--network-layers', type=int, nargs='+', default=[512,512,512], help='filename to use for the trained agent model')
    parser_train.add_argument('--save-intermediate', help="If set, saves the trained model after every timestep", action='store_true')



    parser_test = subparsers.add_parser("test", description='Test a pre-trained traffic control agent', help="Test a previously trained RL agent on a sumo network.", formatter_class=argparse.ArgumentDefaultsHelpFormatter, parents=[sharedoptionsparser])
    parser_test.add_argument('--input', type=str, default=None, required=True, help="The filename of the saved agent model that will be tested")

    # parser.add_argument('--fullobs', type=bool, default=False, help='if use full observation for discrete training')
    # parser.add_argument('--nbobs', type=str, default='None', help='if use observation from neighbours for discrete training, options are either "close" or "far"')
    # parser.add_argument('--add-prevstep', type=bool, default=False, help='if use one step previous states into the observation for discrete training')
    # parser.add_argument('--lookup-step', type=int, default=1, help='the number of continuous steps that is sampled from each episode for learning')
    # parser.add_argument('--num-junctions', type=int, default=2, help='number of junctions, only valid for multi junction control')


    args = parser.parse_args()

    return args, parser

if __name__ == "__main__":
    args, parser = get_args()
    if len(sys.argv) == 1:
        parser.print_usage()
        exit(0)
    print(args)

