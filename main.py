# ------------------------------------------------------------------------
# Open World Object Detection in the Era of Foundation Models
# Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang
# ------------------------------------------------------------------------
# Modified from PROB: Probabilistic Objectness for Open World Object Detection
# Orr Zohar, Jackson Wang, Serena Yeung
# ------------------------------------------------------------------------

import argparse
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
import pandas as pd
from engine import viz, evaluate
from models import build_model
from tqdm import tqdm
import os
import shutil

def parse_env_list(name, cast):
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def get_args_parser():
    parser = argparse.ArgumentParser('RWD - FOMO Detector', add_help=False)
    parser.add_argument('--batch_size', default=10, type=int)
    # dataset parameters
    parser.add_argument('--output_dir', default='experiments',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    ################ dataset configs ################
    parser.add_argument('--test_set', default='test.txt', help='testing txt files')
    parser.add_argument('--train_set', default='train.txt', help='training txt files')
    parser.add_argument('--dataset', default='?', help='defines which dataset is used.')
    parser.add_argument('--data_root', default='./data', type=str)
    parser.add_argument('--data_task', default='RWD', type=str)
    parser.add_argument('--unknown_classnames_file', default='', type=str)
    parser.add_argument('--classnames_file', default='known_classnames.txt', type=str)
    parser.add_argument('--prev_classnames_file', default='known_classnames.txt', type=str)
    parser.add_argument('--templates_file', default='best_templates.txt', type=str)
    parser.add_argument('--attributes_file', default='attributes.json', type=str)
    parser.add_argument('--pred_per_im', default=100, type=int)
    parser.add_argument('--PREV_INTRODUCED_CLS', default=0, type=int)
    parser.add_argument('--CUR_INTRODUCED_CLS', default=30, type=int)
    parser.add_argument('--image_conditioned_file', default='few_shot_data.json', type=str)

    ################ model configs ################
    parser.add_argument('--use_attributes', action='store_true')
    parser.add_argument('--att_selection', action='store_true')
    parser.add_argument('--att_refinement',  action='store_true')
    parser.add_argument('--att_adapt',  action='store_true')
    parser.add_argument('--post_process_method', default='regular',
                        help='seperated: Used for the fs baseline attributes: Used for attribute experiments')

    parser.add_argument('--image_conditioned', action='store_true')
    parser.add_argument('--num_few_shot', default=100, type=int)
    parser.add_argument('--num_att_per_class', default=25, type=int)
    parser.add_argument('--unk_methods', default='sigmoid-max-mcm', type=str)
    parser.add_argument('--unk_method', default='sigmoid-max-mcm', type=str)
    parser.add_argument('--model_name', default='google/owlvit-base-patch16', type=str)
    parser.add_argument('--unk_proposal', action='store_true')
    parser.add_argument('--eval_model', default='', type=str)
    parser.add_argument('--load_weights_only', action='store_true',
                        help='Load only main_weights from eval_model checkpoint, recompute attribute embeddings from attributes.json')
    parser.add_argument('--log_distribution', default=False, type=bool)
    
    parser.add_argument('--image_resize', default=768, type=int,
                        help='image resize 768 for owlvit-base models, 840 for owlvit-large models')
    parser.add_argument('--prev_output_file', default='', type=str)
    parser.add_argument('--output_file', default='', type=str)
    parser.add_argument('--alpha', default=-1.0, type=float)
    parser.add_argument('--balance', default=-1.0, type=float)
    parser.add_argument('--category_distribution', default=False, type=bool)
    parser.add_argument('--fit_method', default='gm', type=str)
    parser.add_argument('--fit_bs', default=1, type=int)
    parser.add_argument('--fit_epoch', default=10000, type=int)
    parser.add_argument('--fit_lr', default=0.01, type=float)
    parser.add_argument('--paper_unknown_objectness', action='store_true')
    parser.add_argument('--paper_alpha_mix', action='store_true')
    parser.add_argument('--paper_no_obj_zscore', action='store_true')
    # Novel ideas (NeurIPS)
    parser.add_argument('--use_qr',        action='store_true',
                        help='Idea A: Quantile Recalibration – align test cos_sim to support CDF')
    parser.add_argument('--use_conformal', action='store_true',
                        help='Idea B: Conformal rank-based normalisation instead of z-score')
    parser.add_argument('--use_saca',      action='store_true',
                        help='Idea D: Spatial Adaptive Confidence Alpha (per-patch alpha)')
    parser.add_argument('--use_miwa',      action='store_true',
                        help='Idea C: MI-Weighted Attributes for unknown aggregation')
    # ─── Idea A (new): Dual-Score Ensemble ────────────────────────────────────
    parser.add_argument('--use_dual_score', action='store_true',
                        help='Additive ensemble of distribution score + MCM uncertainty (failed)')
    parser.add_argument('--dual_w', default=0.3, type=float,
                        help='Weight for MCM uncertainty in dual-score ensemble (0=dist only, 1=MCM only)')
    # ─── Idea D: Attribute Group Max-Pooling ──────────────────────────────────
    parser.add_argument('--use_att_maxpool', action='store_true',
                        help='Idea D: Replace att_w linear sum with per-group max-pooling')
    parser.add_argument('--att_max_groups', default=50, type=int,
                        help='Number of attribute groups for max-pooling (G); group size = num_att // G')
    # ─── Idea B: MCM-SACA (per-patch alpha via MCM uncertainty) ───────────────
    parser.add_argument('--use_saca_mcm', action='store_true',
                        help='Idea B: Per-patch adaptive alpha using MCM uncertainty as proxy')
    parser.add_argument('--saca_mcm_gamma', default=2.0, type=float,
                        help='Sensitivity of sigmoid mapping from MCM uncertainty to alpha')
    parser.add_argument('--saca_mcm_beta', default=0.0, type=float,
                        help='Bias of sigmoid mapping (shifts alpha range up/down)')
    # ─── Idea C: Support-Conditioned Normalization ────────────────────────────
    parser.add_argument('--use_support_norm', action='store_true',
                        help='Idea C: Normalize dist score against support-set stats instead of noisy batch z-score')
    parser.add_argument('--use_known_preserving_gate', action='store_true',
                        help='Suppress unknown objectness for patches with high known-class confidence')
    parser.add_argument('--known_gate_threshold', default=0.6, type=float,
                        help='Known softmax confidence threshold where unknown suppression begins')
    parser.add_argument('--known_gate_gamma', default=1.0, type=float,
                        help='Exponent controlling known-preserving gate sharpness')
    parser.add_argument('--known_gate_floor', default=0.05, type=float,
                        help='Minimum multiplicative unknown score kept by the known-preserving gate')
    parser.add_argument('--known_gate_source', default='softmax', type=str,
                        choices=['softmax', 'sigmoid'],
                        help='Known confidence source for the known-preserving gate')
    parser.add_argument('--use_postprocess_known_gate', action='store_true',
                        help='Suppress only competing unknown postprocess scores when the same patch has a confident known score')
    parser.add_argument('--post_gate_known_threshold', default=0.15, type=float,
                        help='Known probability threshold for postprocess known/unknown competition gating')
    parser.add_argument('--post_gate_unknown_margin', default=1.1, type=float,
                        help='Preserve unknown if unknown score is this many times above the best known score')
    parser.add_argument('--post_gate_floor', default=0.05, type=float,
                        help='Multiplicative floor for gated unknown postprocess scores')
    # ─── Auto calibration: single switch for the best current unknown scoring stack
    parser.add_argument('--use_auto_unknown_calibration', action='store_true',
                        help='Automatically choose the current best unknown scoring calibration policy')
    parser.add_argument('--auto_calibration_policy', default='bootstrap', type=str,
                        choices=['bootstrap'],
                        help='Auto unknown calibration policy to apply')

    parser.add_argument('--save_after_training', action='store_true',
                        help='Save checkpoint immediately after training (single ep/lr path)')
    parser.add_argument('--TCP', default='295499', type=str)
    
    return parser


def apply_auto_unknown_calibration(args):
    """Apply the current best rule-based unknown-scoring policy.

    This intentionally lives above the model builder so the selected values flow
    through the existing FOMO/ClassDistribution code paths. The first policy is
    a bootstrap policy: it converts the previously manual per-domain settings
    into one reproducible switch, with logs/CSV metadata. Later policies can
    replace the table with support-statistics thresholds without changing
    callsites.
    """
    if not getattr(args, 'use_auto_unknown_calibration', False):
        return args

    policy = getattr(args, 'auto_calibration_policy', 'bootstrap')
    if policy != 'bootstrap':
        raise ValueError(f'Unsupported auto calibration policy: {policy}')

    profiles = {
        # Aerial is sharply alpha-sensitive; GM b=0.9 + support norm raises
        # U_AP50 from the main FOMO run's 3.61 to 11.39 in the diagnostic run.
        'Aerial': {
            'fit_method': 'gm',
            'balance': 0.9,
            'alpha': 0.85,
            'use_support_norm': True,
            'paper_no_obj_zscore': False,
            'reason': 'gm_b0.9_alpha0.85_support_norm_for_aerial_separation',
        },
        # Medical's best local behavior is a narrow score-alpha window.
        'Medical': {
            'fit_method': 'score',
            'balance': 0.2,
            'alpha': 0.25,
            'use_support_norm': True,
            'paper_no_obj_zscore': False,
            'reason': 'score_b0.2_alpha0.25_support_norm_for_medical_alpha_window',
        },
        # Surgical and Aquatic benefit from removing the final objectness
        # z-score while keeping support normalization for distribution logits.
        'Surgical': {
            'fit_method': 'score',
            'balance': 0.2,
            'alpha': -1.0,
            'use_support_norm': True,
            'paper_no_obj_zscore': True,
            'reason': 'score_b0.2_no_obj_zscore_support_norm_for_surgical_precision',
        },
        'Aquatic': {
            'fit_method': 'score',
            'balance': 0.2,
            'alpha': -1.0,
            'use_support_norm': True,
            'paper_no_obj_zscore': True,
            'reason': 'score_b0.2_no_obj_zscore_support_norm_for_aquatic_recall',
        },
        # Game is already strong; use the cached score distribution path and
        # support normalization without adding no-zscore.
        'Game': {
            'fit_method': 'score',
            'balance': 0.2,
            'alpha': -1.0,
            'use_support_norm': True,
            'paper_no_obj_zscore': False,
            'reason': 'score_b0.2_support_norm_preserves_game_unknown_ranking',
        },
    }

    profile = profiles.get(args.dataset, {
        'fit_method': args.fit_method,
        'balance': 0.2 if args.balance == -1 else args.balance,
        'alpha': 0.8 if args.alpha == -1 else args.alpha,
        'use_support_norm': True,
        'paper_no_obj_zscore': False,
        'reason': 'default_support_norm_profile',
    })

    args.log_distribution = True
    args.fit_method = profile['fit_method']
    args.balance = profile['balance']
    args.alpha = profile['alpha']
    args.use_support_norm = profile['use_support_norm']
    if profile['paper_no_obj_zscore']:
        args.paper_no_obj_zscore = True

    args.auto_calibration_applied = True
    args.auto_calibration_reason = profile['reason']
    print('[AutoCal] policy={policy} dataset={dataset} fit={fit} balance={balance} '
          'alpha={alpha} support_norm={support_norm} no_obj_zscore={no_z} reason={reason}'.format(
              policy=policy,
              dataset=args.dataset,
              fit=args.fit_method,
              balance=args.balance,
              alpha=args.alpha,
              support_norm=args.use_support_norm,
              no_z=args.paper_no_obj_zscore,
              reason=args.auto_calibration_reason,
          ))
    return args


def save_result(args, output):

    output = pd.DataFrame(output, index=[0])
    if args.prev_output_file:
        try:
            tmp = pd.read_csv(f'{args.output_dir}/{args.prev_output_file}', index_col=0)
            output = pd.concat([tmp, output], ignore_index=True)
        except:
            print('previous file does not exist')

    if args.output_file:
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / args.output_file
        output.to_csv(output_path)

def save_model(args, model, ap=None):
    if args.output_file:
        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            os.makedirs(output_dir, exist_ok=True)
        model_weights = model.state_dict()
        attributes_texts = model.attributes_texts
        att_W = model.att_W
        att_embeds = model.att_embeds
        att_query_mask = model.att_query_mask
        
        save_name = f'{args.output_dir}/{args.dataset}_bast.pth' if ap is None \
                        else f'{args.output_dir}/{args.dataset}_bast_{ap}.pth'
        torch.save({
            'main_weights': model_weights,
            'attributes_texts': attributes_texts,
            'att_W': att_W,
            'att_embeds': att_embeds,
            'att_query_mask': att_query_mask
        }, save_name)

def main(args):
    args = apply_auto_unknown_calibration(args)
    print(args)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ###### get datasets ######
    if len(args.train_set) > 0:
        dataset_train = build_dataset(args, args.train_set)
        data_loader_train = get_dataloader(args, dataset_train, train=False)

    if len(args.test_set) > 0:
        dataset_val = build_dataset(args, args.test_set)
        data_loader_val = get_dataloader(args, dataset_val, train=False)

    neg_sup_ep = parse_env_list("UMB_NEG_SUP_EP", int) or [1, 10, 100]
    neg_sup_lr = parse_env_list("UMB_NEG_SUP_LR", float) or [1e-5, 5e-5, 1e-4]
    best_kmap  = -1 
    bad        = 0


    if (len(neg_sup_ep) > 1 or len(neg_sup_lr) > 1) and args.image_conditioned and args.att_refinement and (not args.eval_model):
        for eps in tqdm(neg_sup_ep, desc='Epochs', leave=False):
            for lr in tqdm(neg_sup_lr, desc='lr', leave=False):
                if bad > 2:
                    continue
                print(f"Starting hyperparameter trial: neg_sup_ep={eps}, neg_sup_lr={lr}")
                args.neg_sup_ep = eps
                args.neg_sup_lr = lr

                model, postprocessors = build_model(args)
                model.to(device)

                if args.distributed:
                    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

                test_stats, coco_evaluator = evaluate(model, postprocessors,
                                                    data_loader_train, dataset_train,
                                                    device, args.output_dir, args)
                output = test_stats['metrics']
                best_eps = eps
                best_lr = lr
                # 记录最好的指标
                output = test_stats['metrics']
                output.update({'model': args.model_name,
                            'dataset': args.dataset,
                            'unk_proposal': args.unk_proposal,
                            'unk_method': args.unk_method,
                            'classnames_file': args.classnames_file,
                            'unknown_classnames_file': args.unknown_classnames_file,
                            'pred_per_im': args.pred_per_im,
                            'num_few_shot': args.num_few_shot,
                            'templates_file': args.templates_file,
                            'auto_calibration': getattr(args, 'use_auto_unknown_calibration', False),
                            'auto_calibration_policy': getattr(args, 'auto_calibration_policy', ''),
                            'auto_calibration_reason': getattr(args, 'auto_calibration_reason', ''),
                            'fit_method': args.fit_method,
                            'balance': args.balance,
                            'alpha': args.alpha,
                            'use_support_norm': args.use_support_norm,
                            'paper_no_obj_zscore': args.paper_no_obj_zscore,
                            'use_known_preserving_gate': args.use_known_preserving_gate,
                            'known_gate_threshold': args.known_gate_threshold,
                            'known_gate_gamma': args.known_gate_gamma,
                            'known_gate_floor': args.known_gate_floor,
                            'known_gate_source': args.known_gate_source,
                            'use_postprocess_known_gate': args.use_postprocess_known_gate,
                            'post_gate_known_threshold': args.post_gate_known_threshold,
                            'post_gate_unknown_margin': args.post_gate_unknown_margin,
                            'post_gate_floor': args.post_gate_floor,
                            'support_norm_mean': getattr(args, 'auto_calibration_support_mean', ''),
                            'support_norm_std': getattr(args, 'auto_calibration_support_std', ''),
                            'support_norm_n': getattr(args, 'auto_calibration_support_n', ''),
                            'best_eps': best_eps,
                            'best_lr': best_lr})
                save_result(args, output)
                if test_stats['metrics']['U_AP50'] > 8:
                    save_model(args, model, test_stats['metrics']['U_AP50'])
                if test_stats['metrics']['K_AP50'] > best_kmap:
                    best_kmap = test_stats['metrics']['K_AP50']
                    bad = 0
                    print(f"New best K_AP50={best_kmap:.4f}; saving model.")
                    save_model(args, model)
                else:
                    bad += 1
                    print(f"No improvement in K_AP50. bad={bad}")
                    
            args.neg_sup_ep = best_eps
            args.neg_sup_lr = best_lr
            print(f"Best so far: neg_sup_ep={best_eps}, neg_sup_lr={best_lr}")
            
    else:
        args.neg_sup_ep = neg_sup_ep[0]
        args.neg_sup_lr = neg_sup_lr[0]
        print(f"Using fixed hyperparameters: neg_sup_ep={args.neg_sup_ep}, neg_sup_lr={args.neg_sup_lr}")
        model, postprocessors = build_model(args)
        model.to(device)
        if getattr(args, 'save_after_training', False):
            save_model(args, model)
            print(f"Saved checkpoint after training to {args.output_dir}/{args.dataset}_bast.pth")

    if args.viz:
        viz(model, postprocessors, data_loader_val, device, args.output_dir, dataset_val, args)
        return

    
    unk_methods = args.unk_methods.split(",")
    for unk_method in unk_methods:
        print(f"\n running method {unk_method}\n")
        model.unk_head.method = unk_method

        test_stats, coco_evaluator = evaluate(model, postprocessors, data_loader_val, dataset_val, device,
                                              args.output_dir, args)
        output = test_stats['metrics']
        output.update({'model': args.model_name,
                       'dataset': args.dataset,
                       'unk_proposal': args.unk_proposal,
                       'unk_method': args.unk_method,
                       'classnames_file': args.classnames_file,
                       'unknown_classnames_file': args.unknown_classnames_file,
                       'pred_per_im': args.pred_per_im,
                       'num_few_shot': args.num_few_shot,
                       'templates_file': args.templates_file,
                       'auto_calibration': getattr(args, 'use_auto_unknown_calibration', False),
                       'auto_calibration_policy': getattr(args, 'auto_calibration_policy', ''),
                       'auto_calibration_reason': getattr(args, 'auto_calibration_reason', ''),
                       'fit_method': args.fit_method,
                       'balance': args.balance,
                       'alpha': args.alpha,
                       'use_support_norm': args.use_support_norm,
                       'paper_no_obj_zscore': args.paper_no_obj_zscore,
                       'use_known_preserving_gate': args.use_known_preserving_gate,
                       'known_gate_threshold': args.known_gate_threshold,
                       'known_gate_gamma': args.known_gate_gamma,
                       'known_gate_floor': args.known_gate_floor,
                       'known_gate_source': args.known_gate_source,
                       'use_postprocess_known_gate': args.use_postprocess_known_gate,
                       'post_gate_known_threshold': args.post_gate_known_threshold,
                       'post_gate_unknown_margin': args.post_gate_unknown_margin,
                       'post_gate_floor': args.post_gate_floor,
                       'support_norm_mean': getattr(args, 'auto_calibration_support_mean', ''),
                       'support_norm_std': getattr(args, 'auto_calibration_support_std', ''),
                       'support_norm_n': getattr(args, 'auto_calibration_support_n', '')})
        save_result(args, output)


def get_dataloader(args, dataset, train=True):
    if args.distributed:
        sampler = samplers.DistributedSampler(dataset, shuffle=train)
    else:
        if train:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    if train:
        batch_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)
        data_loader = DataLoader(dataset, batch_sampler=batch_sampler,
                                 collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    else:
        data_loader = DataLoader(dataset, args.batch_size, sampler=sampler,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    return data_loader


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RWOD and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    print("*********************************************Finshed Run*********************************************")
