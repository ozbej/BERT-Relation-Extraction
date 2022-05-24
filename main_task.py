#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:40:16 2019

@author: weetee
"""
from src.tasks.trainer import train_and_fit
from src.tasks.infer import infer_from_trained, FewRel, infer_from_termframe
import logging
from argparse import ArgumentParser

'''
This fine-tunes the BERT model on SemEval, FewRel tasks
'''

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default='semeval', help='semeval, fewrel, termframe')
    parser.add_argument("--train_data", type=str, default='./data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT', \
                        help="training data .txt file path")
    parser.add_argument("--test_data", type=str, default='./data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT', \
                        help="test data .txt file path")
    parser.add_argument("--infer_data", type=str, default='', \
                        help="infer data .txt file path")
    parser.add_argument("--use_pretrained_blanks", type=int, default=0, help="0: Don't use pre-trained blanks model, 1: use pre-trained blanks model")
    parser.add_argument("--num_classes", type=int, default=19, help='number of relation classes')
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
    parser.add_argument("--num_epochs", type=int, default=11, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00007, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help='''Model ID: 0 - BERT\n
                                                                            1 - ALBERT\n
                                                                            2 - BioBERT''')
    parser.add_argument("--model_size", type=str, default='bert-base-uncased', help="For BERT: 'bert-base-uncased', \
                                                                                                'bert-large-uncased',\
                                                                                    For ALBERT: 'albert-base-v2',\
                                                                                                'albert-large-v2'\
                                                                                    For BioBERT: 'bert-base-uncased' (biobert_v1.1_pubmed)")
    parser.add_argument("--train", type=int, default=1, help="0: Don't train, 1: train")
    parser.add_argument("--infer", type=int, default=1, help="0: Don't infer, 1: Infer")
    parser.add_argument("--detect_entities", type=int, default=0, help="0: Don't detect entities, 1: Detect entities")
    parser.add_argument("--annotated", type=int, default=1, help="0: Data is not annotated, 1: Data is annotated")
    
    args = parser.parse_args()
    
    if (args.train == 1) and (args.task != 'fewrel'):
        net = train_and_fit(args)
        
    if (args.infer == 1) and (args.task != 'fewrel'):
        inferer = infer_from_trained(args, detect_entities=True)
        infer_from_termframe(args.infer_data, inferer, detect_entities=args.detect_entities, annotated=args.annotated)
    
    if args.task == 'fewrel':
        fewrel = FewRel(args)
        meta_input, e1_e2_start, meta_labels, outputs = fewrel.evaluate()