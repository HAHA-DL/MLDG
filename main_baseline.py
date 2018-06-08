import argparse

from model import ModelBaseline


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--test_every", type=int, default=50,
                                  help="number of test every steps")
    train_arg_parser.add_argument("--batch_size", type=int, default=128,
                                  help="batch size for training, default is 64")
    train_arg_parser.add_argument("--num_classes", type=int, default=10,
                                  help="number of classes")
    train_arg_parser.add_argument("--step_size", type=int, default=1,
                                  help="number of step size to decay the lr")
    train_arg_parser.add_argument("--inner_loops", type=int, default=200000,
                                  help="number of classes")
    train_arg_parser.add_argument("--unseen_index", type=int, default=0,
                                  help="index of unseen domain")
    train_arg_parser.add_argument("--lr", type=float, default=0.0001,
                                  help='learning rate of the model')
    train_arg_parser.add_argument("--weight_decay", type=float, default=0.00005,
                                  help='weight decay')
    train_arg_parser.add_argument("--momentum", type=float, default=0.9,
                                  help='momentum')
    train_arg_parser.add_argument("--logs", type=str, default='logs/',
                                  help='logs folder to write log')
    train_arg_parser.add_argument("--model_path", type=str, default='',
                                  help='folder for saving model')
    train_arg_parser.add_argument("--state_dict", type=str, default='',
                                  help='model of pre trained')
    train_arg_parser.add_argument("--data_root", type=str, default='',
                                  help='folder root of the data')
    train_arg_parser.add_argument("--debug", type=bool, default=False,
                                  help='whether for debug mode or not')
    args = main_arg_parser.parse_args()

    model_obj = ModelBaseline(flags=args)
    model_obj.train(flags=args)

    # after training, we should test the held out domain
    model_obj.heldout_test(flags=args)


if __name__ == "__main__":
    main()
