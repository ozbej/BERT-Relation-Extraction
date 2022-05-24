#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:44:17 2019

@author: weetee
"""

import pickle
import os
import pandas as pd
import torch
import spacy
import re
from itertools import permutations
from tqdm import tqdm
from .preprocessing_funcs import load_dataloaders
from ..misc import save_as_pickle
from sklearn.metrics import f1_score
from operator import itemgetter

import logging

tqdm.pandas(desc="prog-bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

rel2idx = {
    "HAS_CAUSE(e1,e2)": 1,
    "HAS_CAUSE(e2,e1)": 2,
    "HAS_RESULT(e1,e2)": 3,
    "HAS_RESULT(e2,e1)": 4,
    "HAS_FORM(e1,e2)": 5,
    "HAS_FORM(e2,e1)": 6,
    "HAS_LOCATION(e1,e2)": 7,
    "HAS_LOCATION(e2,e1)": 8,
    "HAS_ATTRIBUTE(e1,e2)": 9,
    "HAS_ATTRIBUTE(e2,e1)": 10,
    "DEFINED_AS(e1,e2)": 11,
    "DEFINED_AS(e2,e1)": 12,
}

class infer_from_termframe(object):
    def __init__(self, input_file, inferer, detect_entities=False, annotated=True):
        self.input_file = input_file
        self.inferer = inferer
        self.detect_entities = detect_entities
        self.annotated = annotated

        if self.annotated:
            pass
        else:
            self.infer_unnanotated()

    def infer_unnanotated(self):
        """
        Evaluate termframe results (unannotated)
        """
        f_in = open(self.input_file, "r", encoding="utf-8")

        all_cases = 0

        while True:
            sentence = f_in.readline().replace("\n", "")
            if not sentence: break

            sentence = sentence.split("\t")[1]

            predicted = self.inferer.infer_sentence(sentence, detect_entities=True).strip(" \n")

            print("Sentence: %s" % sentence)
            print("Predicted: %s" % predicted)
            print()

            all_cases += 1

        print("Predicted %d cases" % (all_cases))

    def infer_annotated(self):
        """
        Evaluate termframe results (annotated)
        """
        f_in = open(self.input_file, "r", encoding="utf-8")

        all_cases = 0
        correct_cases = 0
        correct_cases_stripped = 0

        y_true = list()
        y_pred = list()

        while True:
            sentence = f_in.readline().replace("\n", "")
            sentence = sentence.replace("<e1>", "[E1]")
            sentence = sentence.replace("</e1>", "[/E1]")
            sentence = sentence.replace("<e2>", "[E2]")
            sentence = sentence.replace("</e2>", "[/E2]")
            if self.detect_entities:
                sentence = sentence.replace("[E1]", "")
                sentence = sentence.replace("[/E1]", "")
                sentence = sentence.replace("[E2]", "")
                sentence = sentence.replace("[/E2]", "")
            relations = f_in.readline()
            if not sentence or not relations: break
            f_in.readline()
            f_in.readline()

            sentence = sentence.split("\t")[1]

            predicted, _ = self.inferer.infer_sentence(sentence, detect_entities=self.detect_entities)

            predicted.strip(" \n")

            # Filter relations (remove newlines and spaces)
            relations = [relation.strip(" \n") for relation in relations.split(" ")]

            print("Sentence: %s" % sentence)
            print("Predicted: %s" % predicted)
            print("Real: %s" % relations)
            print()

            y_pred.append(rel2idx[predicted])

            # Count correct predictions (both relations and entity order)
            if predicted in relations:
                y_true.append(rel2idx[predicted])
                correct_cases += 1
            else:
                y_true.append(0)

            relations_stripped = list()
            predicted_stripped = re.sub(r'\(.*\)\n*', '', predicted)

            relations_stripped = [re.sub(r'\(.*\)\n*', '', relation) for relation in relations]

            # Count correct predictions (only relations)
            if predicted_stripped in relations_stripped:
                correct_cases_stripped += 1

            all_cases += 1

        f1 = f1_score(y_true, y_pred, average="macro")

        print("Accuracy (relations + entity order): %.3f (%d/%d)" % (correct_cases/all_cases, correct_cases, all_cases))
        print("Accuracy (relations): %.3f (%d/%d) " % (correct_cases_stripped/all_cases, correct_cases_stripped, all_cases))
        print("F1: %.3f" % f1)

class infer_from_trained(object):
    def __init__(self, args=None, detect_entities=False):
        if args is None:
            self.args = load_pickle("args.pkl")
        else:
            self.args = args
        self.cuda = torch.cuda.is_available()
        self.detect_entities = detect_entities
        
        if self.detect_entities:
            self.nlp = spacy.load("en_core_web_lg")
        else:
            self.nlp = None
        self.entities_of_interest = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", \
                                     "WORK_OF_ART", "LAW", "LANGUAGE", 'PER']
        
        logger.info("Loading tokenizer and model...")
        from .train_funcs import load_state
        
        if self.args.model_no == 0:
            from ..model.BERT.modeling_bert import BertModel as Model
            model = args.model_size #'bert-base-uncased'
            lower_case = True
            model_name = 'BERT'
            self.net = Model.from_pretrained(model, force_download=False, \
                                         model_size=args.model_size,\
                                         task='classification', n_classes_=self.args.num_classes)
        elif self.args.model_no == 1:
            from ..model.ALBERT.modeling_albert import AlbertModel as Model
            model = args.model_size #'albert-base-v2'
            lower_case = False
            model_name = 'ALBERT'
            self.net = Model.from_pretrained(model, force_download=False, \
                                         model_size=args.model_size,\
                                         task='classification', n_classes_=self.args.num_classes)
        elif args.model_no == 2: # BioBert
            from ..model.BERT.modeling_bert import BertModel, BertConfig
            model = 'bert-base-uncased'
            lower_case = False
            model_name = 'BioBERT'
            config = BertConfig.from_pretrained('./additional_models/biobert_v1.1_pubmed/bert_config.json')
            self.net = BertModel.from_pretrained(pretrained_model_name_or_path='./additional_models/biobert_v1.1_pubmed/biobert_v1.1_pubmed.bin', 
                                                 config=config,
                                                 force_download=False, \
                                                 model_size='bert-base-uncased',
                                                 task='classification',\
                                                 n_classes_=self.args.num_classes)
        
        self.tokenizer = load_pickle("%s_tokenizer.pkl" % model_name)
        self.net.resize_token_embeddings(len(self.tokenizer))
        if self.cuda:
            self.net.cuda()
        start_epoch, best_pred, amp_checkpoint = load_state(self.net, None, None, self.args, load_best=False)
        logger.info("Done!")
        
        self.e1_id = self.tokenizer.convert_tokens_to_ids('[E1]')
        self.e2_id = self.tokenizer.convert_tokens_to_ids('[E2]')
        self.pad_id = self.tokenizer.pad_token_id
        self.rm = load_pickle("relations.pkl")
        
    def get_all_ent_pairs(self, sent):
        if isinstance(sent, str):
            sents_doc = self.nlp(sent)
        else:
            sents_doc = sent
        ents = sents_doc.ents
        pairs = []
        if len(ents) > 1:
            for a, b in permutations([ent for ent in ents], 2):
                pairs.append((a,b))
        return pairs
    
    def get_all_sub_obj_pairs(self, sent):
        if isinstance(sent, str):
            sents_doc = self.nlp(sent)
        else:
            sents_doc = sent
        sent_ = next(sents_doc.sents)
        root = sent_.root
        #print('Root: ', root.text)
        
        subject = None; objs = []; pairs = []
        for child in root.children:
            #print(child.dep_)
            if child.dep_ in ["nsubj", "nsubjpass"]:
                if len(re.findall("[a-z]+",child.text.lower())) > 0: # filter out all numbers/symbols
                    subject = child; #print('Subject: ', child)
            elif child.dep_ in ["dobj", "attr", "prep", "ccomp"]:
                objs.append(child); #print('Object ', child)
        
        if (subject is not None) and (len(objs) > 0):
            for a, b in permutations([subject] + [obj for obj in objs], 2):
                a_ = [w for w in a.subtree]
                b_ = [w for w in b.subtree]
                pairs.append((a_[0] if (len(a_) == 1) else a_ , b_[0] if (len(b_) == 1) else b_))
                    
        return pairs
    
    def annotate_sent(self, sent_nlp, e1, e2):
        annotated = ''
        e1start, e1end, e2start, e2end = 0, 0, 0, 0
        for token in sent_nlp:
            if not isinstance(e1, list):
                if (token.text == e1.text) and (e1start == 0) and (e1end == 0):
                    annotated += ' [E1]' + token.text + '[/E1] '
                    e1start, e1end = 1, 1
                    continue
                
            else:
                if (token.text == e1[0].text) and (e1start == 0):
                    annotated += ' [E1]' + token.text + ' '
                    e1start += 1
                    continue
                elif (token.text == e1[-1].text) and (e1end == 0):
                    annotated += token.text + '[/E1] '
                    e1end += 1
                    continue
           
            if not isinstance(e2, list):
                if (token.text == e2.text) and (e2start == 0) and (e2end == 0):
                    annotated += ' [E2]' + token.text + '[/E2] '
                    e2start, e2end = 1, 1
                    continue
            else:
                if (token.text == e2[0].text) and (e2start == 0):
                    annotated += ' [E2]' + token.text + ' '
                    e2start += 1
                    continue
                elif (token.text == e2[-1].text) and (e2end == 0):
                    annotated += token.text + '[/E2] '
                    e2end += 1
                    continue
            annotated += ' ' + token.text + ' '
            
        annotated = annotated.strip()
        annotated = re.sub(' +', ' ', annotated)
        return annotated
    
    def get_annotated_sents(self, sent):
        sent_nlp = self.nlp(sent)
        pairs = self.get_all_ent_pairs(sent_nlp)
        pairs.extend(self.get_all_sub_obj_pairs(sent_nlp))
        if len(pairs) == 0:
            print('Found less than 2 entities!')
            return
        annotated_list = []
        for pair in pairs:
            annotated = self.annotate_sent(sent_nlp, pair[0], pair[1])
            annotated_list.append(annotated)
        return annotated_list
    
    def get_e1e2_start(self, x):
        e1_e2_start = ([i for i, e in enumerate(x) if e == self.e1_id][0],\
                        [i for i, e in enumerate(x) if e == self.e2_id][0])
        return e1_e2_start
    
    def infer_one_sentence(self, sentence):
        try:
            self.net.eval()
            tokenized = self.tokenizer.encode(sentence); #print(tokenized)
            e1_e2_start = self.get_e1e2_start(tokenized); #print(e1_e2_start)
            tokenized = torch.LongTensor(tokenized).unsqueeze(0)
            e1_e2_start = torch.LongTensor(e1_e2_start).unsqueeze(0)
            attention_mask = (tokenized != self.pad_id).float()
            token_type_ids = torch.zeros((tokenized.shape[0], tokenized.shape[1])).long()
            
            if self.cuda:
                tokenized = tokenized.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
            
            with torch.no_grad():
                classification_logits = self.net(tokenized, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
                                            e1_e2_start=e1_e2_start)
                predicted_item = torch.softmax(classification_logits, dim=1).max(1)[1].item()
                conf_score = predicted_item[0].item()
                predicted = predicted_item[1].item()
            return self.rm.idx2rel[predicted].strip(), conf_score
        except:
            return None, None
        #print("Sentence: ", sentence)
        #print("Predicted: ", self.rm.idx2rel[predicted].strip(), '\n')
    
    def infer_sentence(self, sentence, detect_entities=False):
        if detect_entities:
            sentences = self.get_annotated_sents(sentence)
            if sentences != None:
                preds = []
                for sent in sentences:
                    pred = self.infer_one_sentence(sent)
                    if pred[0] is not None and pred[1] is not None:
                        preds.append(pred)
                return max(preds,key=itemgetter(1))
        else:
            return self.infer_one_sentence(sentence)

class FewRel(object):
    def __init__(self, args=None):
        if args is None:
            self.args = load_pickle("args.pkl")
        else:
            self.args = args
        self.cuda = torch.cuda.is_available()
        
        if self.args.model_no == 0:
            from ..model.BERT.modeling_bert import BertModel as Model
            from ..model.BERT.tokenization_bert import BertTokenizer as Tokenizer
            model = args.model_size #'bert-large-uncased' 'bert-base-uncased'
            lower_case = True
            model_name = 'BERT'
            self.net = Model.from_pretrained(model, force_download=False, \
                                             model_size=args.model_size,\
                                             task='fewrel')
        elif self.args.model_no == 1:
            from ..model.ALBERT.modeling_albert import AlbertModel as Model
            from ..model.ALBERT.tokenization_albert import AlbertTokenizer as Tokenizer
            model = args.model_size #'albert-base-v2'
            lower_case = False
            model_name = 'ALBERT'
            self.net = Model.from_pretrained(model, force_download=False, \
                                             model_size=args.model_size,\
                                             task='fewrel')
        elif args.model_no == 2: # BioBert
            from ..model.BERT.modeling_bert import BertModel, BertConfig
            from ..model.BERT.tokenization_bert import BertTokenizer as Tokenizer
            model = 'bert-base-uncased'
            lower_case = False
            model_name = 'BioBERT'
            config = BertConfig.from_pretrained('./additional_models/biobert_v1.1_pubmed/bert_config.json')
            self.net = BertModel.from_pretrained(pretrained_model_name_or_path='./additional_models/biobert_v1.1_pubmed/biobert_v1.1_pubmed.bin', 
                                            config=config,
                                            force_download=False, \
                                            model_size='bert-base-uncased',
                                            task='fewrel')
        
        if os.path.isfile('./data/%s_tokenizer.pkl' % model_name):
            self.tokenizer = load_pickle("%s_tokenizer.pkl" % model_name)
            logger.info("Loaded tokenizer from saved file.")
        else:
            logger.info("Saved tokenizer not found, initializing new tokenizer...")
            if args.model_no == 2:
                self.tokenizer = Tokenizer(vocab_file='./additional_models/biobert_v1.1_pubmed/vocab.txt',
                                           do_lower_case=False)
            else:
                self.tokenizer = Tokenizer.from_pretrained(model, do_lower_case=False)
            self.tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]', '[BLANK]'])
            save_as_pickle("%s_tokenizer.pkl" % model_name, self.tokenizer)
            logger.info("Saved %s tokenizer at ./data/%s_tokenizer.pkl" %(model_name, model_name))
            
        
        self.net.resize_token_embeddings(len(self.tokenizer))
        self.pad_id = self.tokenizer.pad_token_id
        
        if self.cuda:
            self.net.cuda()
        
        if self.args.use_pretrained_blanks == 1:
            logger.info("Loading model pre-trained on blanks at ./data/test_checkpoint_%d.pth.tar..." % args.model_no)
            checkpoint_path = "./data/test_checkpoint_%d.pth.tar" % self.args.model_no
            checkpoint = torch.load(checkpoint_path)
            model_dict = self.net.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.net.load_state_dict(pretrained_dict, strict=False)
            del checkpoint, pretrained_dict, model_dict
        
        logger.info("Loading Fewrel dataloaders...")
        self.train_loader, _, self.train_length, _ = load_dataloaders(args)
        
    def evaluate(self):
        counts, hits = 0, 0
        logger.info("Evaluating...")
        with torch.no_grad():
            for meta_input, e1_e2_start, meta_labels in tqdm(self.train_loader, total=len(self.train_loader)):
                attention_mask = (meta_input != self.pad_id).float()
                token_type_ids = torch.zeros((meta_input.shape[0], meta_input.shape[1])).long()
        
                if self.cuda:
                    meta_input = meta_input.cuda()
                    attention_mask = attention_mask.cuda()
                    token_type_ids = token_type_ids.cuda()
                
                outputs = self.net(meta_input, token_type_ids=token_type_ids, attention_mask=attention_mask, Q=None,\
                                  e1_e2_start=e1_e2_start)
                
                matrix_product = torch.mm(outputs, outputs.T)
                closest_idx = matrix_product[-1][:-1].argmax().cpu().item()
                
                if closest_idx == meta_labels[-1].item():
                    hits += 1
                counts += 1
        
        print("Results (%d samples): %.3f %%" % (counts, (hits/counts)*100))
        return meta_input, e1_e2_start, meta_labels, outputs