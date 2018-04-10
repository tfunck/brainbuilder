from sys import exit
from train_model import train_model
from apply_model import apply_model
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train-source',dest='train_source_dir', default='', help='Directory with raw images')
    parser.add_argument('--train-output',dest='train_output_dir', default='',  help='Directory name for outputs')
    parser.add_argument('--raw-source',dest='raw_source_dir', default='', help='Directory with raw images')
    parser.add_argument('--raw-output',dest='raw_output_dir', default='',  help='Directory name for outputs')
    parser.add_argument('--step',dest='step', default=0.1, type=float, help='File extension for input files (default=.tif)')
    parser.add_argument('--epochs',dest='epochs', default=1, type=int, help='Number of epochs')
    parser.add_argument('--clobber', dest='clobber', action='store_true', default=False, help='Clobber results')

    args = parser.parse_args()
    if args.train_source_dir != '' and args.train_output_dir != '' :
        train_model(args.train_source_dir, args.train_output_dir, step=args.step, epochs=args.epochs, clobber=args.clobber)
    else :
        print("Skipping train_model because either --train-source or --train-output is not set")
    
    if args.raw_source_dir != '' and args.raw_output_dir != '' and args.train_output_dir != '' :
        apply_model(args.train_output_dir,args.raw_source_dir,args.raw_output_dir,args.step, args.clobber)
    else :
        print("Skipping apply_model because either --train-output, --raw-source, or --raw-output are not set")

