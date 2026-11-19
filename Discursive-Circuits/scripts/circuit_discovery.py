import argparse
import sys
import os
import json
import torch
import logging
import transformers
logger = logging.getLogger(__name__)
from DC_config import DC_Config

PACKAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "auto-circuit-DC")
if PACKAGE_PATH not in sys.path:
    sys.path.insert(0, PACKAGE_PATH)

from pathlib import Path
from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import PruneScores
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.visualize import draw_seq_graph
from auto_circuit.types import (
    AblationType,
    CircuitOutputs,
    PatchType,
)
from auto_circuit.prune import run_circuits
from auto_circuit.metrics.prune_metrics.correct_answer_percent import measure_correct_ans_percent
from auto_circuit.metrics.prune_metrics.answer_diff import measure_answer_diff
from auto_circuit.visualize import obtain_top_edge_names, edit_prune_scores


class CircuitDiscovery:
    def __init__(self, config):    
        self.config = config
        self.desired_relation = self.config.desired_relation
        self.source_relation = self.config.source_relation
        self.dataset_name = self.config.dataset_name
        #self.dataset_name_source = self.config.dataset_name_source
        self.batch_size = self.config.batch_size
        self.N_sample = self.config.N_sample
        self.mode = self.config.mode
        self.level = self.config.level
        self.gentype = self.config.gentype
        self.seed = self.config.seed
        self.model_path = self.config.model_path
        self.model_name = self.config.model_name
        self.N_sample = self.config.N_sample
        self.thr = self.config.thr
        self.model_mode = self.config.model_mode
        self.if_mlm = self.config.if_mlm

        self.top_edges_dir = self.config.top_edges_dir
        self.top_edges = None # for now
        self.inverse = self.config.data_inverse
        #self.config_relation_dir = self.config.config_relation_dir
        #self.config_relation_fp = os.path.join(self.config_relation_dir, f'{self.dataset_name}2{self.dataset_name_source}.json')

        self.edge_series = self.config.edge_series
        postfix = 'small' if 'small' in self.model_name else 'medium'
        self.outer_folder = self.model_mode+'_'+postfix
        self.outer_folder = self.outer_folder+'_inv' if self.inverse else self.outer_folder
        self.results_dir =f'/srv/data1/abhidipbhatt/output/DC_exp/{self.outer_folder}/{self.source_relation}/{self.source_relation}_{self.model_mode}_{self.seed}/{self.desired_relation}'
        print(f'++++++++++++++++++ {self.inverse}')
        print(self.outer_folder)
        if self.source_relation == 'all':
            self.patching_data_path_train = self.config.patching_data_train_fp.format(self.model_mode,self.source_relation)
        elif self.source_relation in ['AB','A_not', 'B_not', 'not_not']:
            train_dataset_suff = self.source_relation+"_300"
            self.patching_data_path_train = self.config.patching_data_train_fp.format(self.model_mode, train_dataset_suff)
        else: #other circuits
            if self.desired_relation=='all':
                train_dataset_suff = self.desired_relation
            else:
                train_dataset_suff = self.desired_relation+"_300"
            self.patching_data_path_train = self.config.patching_data_train_fp.format(self.model_mode, train_dataset_suff)
            

        if self.desired_relation == 'all':
            self.patching_data_path_val = self.config.patching_data_val_fp.format(self.model_mode, self.desired_relation)
        else:
            val_dataset_suff = self.desired_relation+"_30"
            self.patching_data_path_val = self.config.patching_data_val_fp.format(self.model_mode, val_dataset_suff)
            
        '''
        if self.gentype == "genrel" or self.gentype == "gendataset":
            self.mode = "patch" # For the best result, we use the saved edges for generalization experiments.
        
        self.L_rel_name = None
        if self.level == 0:
            self.L_fp = 'L0_PDTB.json'
        elif self.level == 1:
            self.L_rel_name = self.source_relation.split('.')
            self.L_fp = f'L1_{self.L_rel_name[0]}.json'
        elif self.level == 2:
            self.L_rel_name = self.source_relation.split('.')
            self.L_rel_name = '.'.join(self.L_rel_name[:2])
            self.L_fp = f'L2_{self.L_rel_name}.json'
        elif self.level == 3:
            self.L_rel_name = self.source_relation
            self.L_fp = f'L3_{self.L_rel_name}.json'
        
        if self.source_relation == "ioi":
            self.L_fp = 'THE_ioi.json'
            self.patching_data_path_train = self.config.ioi_fp
        if self.source_relation == "random":
            self.L_fp = 'THE_random.json'
            self.patching_data_path_train = self.config.ioi_fp # placeholder for now. For random, we don't need to "discover" the circuit.
        '''
        
        try:
            #print("-----{}".format(self.source_relation))
            if self.source_relation in ['All_intersection', 'All_union', 'All_top','All_top_causation_ante','All_intersection_causation_ante','SUPER_All_intersection','SUPER_All_union']:
                self.top_edges_file = f'/srv/data1/abhidipbhatt/output/DC_exp/{self.outer_folder}/All_approx/{self.source_relation}/top_edges.json'
                self.results_dir = f'/srv/data1/abhidipbhatt/output/DC_exp/{self.outer_folder}/All_approx/{self.source_relation}/{self.desired_relation}'
            elif "_random_" in self.source_relation:
                folder1 = self.source_relation.split("_")[0]
                self.top_edges_file = f'/srv/data1/abhidipbhatt/output/DC_exp/{self.outer_folder}/{folder1}/{self.source_relation}/rand_1000_edges_seed_{self.seed}.json'
                self.results_dir = f'/srv/data1/abhidipbhatt/output/DC_exp/{self.outer_folder}/{folder1}/{self.source_relation}/{self.desired_relation}/{self.seed}'
            elif self.source_relation not in ['AB','A_not', 'B_not', 'not_not','all']:
                self.top_edges_file = f'/srv/data1/abhidipbhatt/output/DC_exp/{self.model_mode}/other_circuits/{self.source_relation}.json'
                self.results_dir = f'/srv/data1/abhidipbhatt/output/DC_exp/{self.outer_folder}/other_circuits/{self.source_relation}/{self.desired_relation}'
                self.results_dir = self.results_dir.replace(".","_")
            else:
                self.top_edges_file = f'/srv/data1/abhidipbhatt/output/DC_exp/{self.outer_folder}/{self.source_relation}/{self.source_relation}_{self.model_mode}_ALL_seeds/top_1000_edges.json' #os.path.join(self.top_edges_dir, self.L_fp)
            with open(self.top_edges_file, 'r') as f:
                self.top_edges = json.load(f)
            print(f"Loaded top edges from {self.top_edges_file}. There are {len(self.top_edges)} edges.")            
        except FileNotFoundError:
            print(f"Top edges file {self.top_edges_file} not found. Please check the path.")
            self.top_edges = None
            if self.mode == "patch":
                print('no top edges file found, so no patching')
                exit()
        
        print('the state of gentype is {}'.format(self.gentype))
        print('train path: {}'.format(self.patching_data_path_train))
        print('val path: {}'.format(self.patching_data_path_val))
        
        self.json_path_train = Path(self.patching_data_path_train)
        self.json_path_val = Path(self.patching_data_path_val)
        
        with open(self.json_path_train, 'r') as f:
            data_train = json.load(f)
        with open(self.json_path_val, 'r') as f:
            data_val = json.load(f)

        prompts_train = data_train.get('prompts', [])
        prompts_val = data_val.get('prompts', [])
        
        print(f"Loaded {len(prompts_train)} examples from {os.path.basename(self.patching_data_path_train)}")
        print(f"Loaded {len(prompts_val)} examples from {os.path.basename(self.patching_data_path_val)}")

        #self.results_dir = self.config.results_dir.format(self.level, self.gentype, self.dataset_name, self.N_sample, self.source_relation, self.model_mode, self.seed, self.desired_relation)
        
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"Created results directory at {self.results_dir}")
        print('=== init done, now we are ready to run the experiment ===')

    def run_eap(self):
        if torch.cuda.is_available():
            device = torch.device(f"cuda")
        else:
            device = torch.device("cpu")
        model = load_tl_model(self.model_path, device, base_name=self.model_name)
        model = model.to(device)
        print('model loaded')

        if self.if_mlm:
            mlm_head = transformers.BertForMaskedLM.from_pretrained(os.path.dirname(self.model_path)).cls.predictions
            mlm_head = mlm_head.to(device)

        TRAIN_SEED = self.seed
        print('loading train loader')
        train_loader, _ = load_datasets_from_json(
            model=model,
            path=self.json_path_train,
            device=device,
            prepend_bos=True if self.model_mode=='clm' else False,
            batch_size=self.batch_size,
            train_test_size=(self.N_sample, 1),
            random_seed = TRAIN_SEED,
            inverse = self.inverse,
        )

        TEST_SEED = self.seed + 1
        print(f'loading test loader from {self.json_path_val}: batch size: {self.batch_size}')
        test_loader, _ = load_datasets_from_json(
            model=model,
            path=self.json_path_val,
            device=device,
            prepend_bos=True if self.model_mode=='clm' else False,
            batch_size=self.batch_size,
            train_test_size=(self.N_sample, 1),
            random_seed = TEST_SEED,
            inverse = self.inverse,
        )
        #print(f'len {len(test_loader)}')
        model = patchable_model(
            model,
            factorized=True,
            slice_output="last_seq",
            separate_qkv=True,
            device=device,
        )

        print('starting to perform edge attribution patching')

        # This is equivalent to the edge attribution patching method, as the author of auto-circuit explained. 
        attribution_scores: PruneScores = mask_gradient_prune_scores(
            model=model,
            dataloader=train_loader,
            grad_function="logit",
            answer_function="avg_diff",
            mask_val=0.0,
            official_edges=None,
            if_mlm= self.if_mlm,
            #mlm_head = mlm_head,
        )
        
        thr = self.thr
        
        #fig_eap = draw_seq_graph(
        #    model, attribution_scores, thr, layer_spacing=True, orientation="h", seq_labels=train_loader.seq_labels
        #)
        #fp = os.path.join(self.results_dir, "circuit-{}-h.pdf".format(self.desired_relation))
        #fig_eap.write_image(fp, scale=4, format='pdf')
        #print(f"Saved EAP figure to {fp}")

        top_edge_names, top_edge_names_raw = obtain_top_edge_names(
            model=model,
            seq_edges=set(model.edges),
            prune_scores=attribution_scores,
            orientation="h"
        )

        top_edges_dict_raw = {
            f"{name}": score 
            for name, score in top_edge_names_raw
        }

        top_1000_edges_raw = dict(sorted(top_edges_dict_raw.items(), 
                                   key=lambda x: abs(x[1]), 
                                   reverse=True)[:1000])

        if self.gentype == "own" and self.mode != "patch":
            edge_names_file_raw = os.path.join(self.results_dir, f'top_edge_names_raw_{self.desired_relation}.json')
            with open(edge_names_file_raw, 'w') as f:
                json.dump(top_edges_dict_raw, f, indent=4) #top_1000_edges_raw
            print(f'Saved top edge names to {edge_names_file_raw}')
        
        attribution_shapes = {
            key: tensor.shape 
            for key, tensor in attribution_scores.items()
        }
        print(f"Number of attribution tensors: {len(attribution_shapes)}")

        if self.mode == "learn":
            circuit_outs: CircuitOutputs = run_circuits(
                model=model,
                dataloader=test_loader,
                test_edge_counts=self.edge_series,
                prune_scores=attribution_scores,
                if_mlm= self.if_mlm,
                #mlm_head=mlm_head,
            )
        elif self.mode == "patch":
            print("---patching-----")
            #print(f'len {len(test_loader)}')
            patch_edges = self.top_edges
            prune_scores_patched = edit_prune_scores(model, set(model.edges), attribution_scores, patch_edges)
            circuit_outs: CircuitOutputs = run_circuits(
                model=model,
                dataloader=test_loader,
                test_edge_counts=self.edge_series,
                prune_scores=prune_scores_patched,
                if_mlm= self.if_mlm,
                #mlm_head=mlm_head,
            )

        
        correct_ans_percent_measurements = measure_correct_ans_percent(
            model=model,
            dataloader=test_loader,
            pruned_outs=circuit_outs,
            out_of_correct_and_incorrect_answers=True
        )

        correct_ans_percent_results = {
            edge_count: score
            for edge_count, score in correct_ans_percent_measurements
        }

        with open(os.path.join(self.results_dir, f"correct_ans_percent_results_{self.desired_relation}.json"), 'w') as f:
            json.dump(correct_ans_percent_results, f, indent=4)

        print("\nCorrect Answer Percent Results:")
        for item in correct_ans_percent_measurements:
            edge_count, score = item
            print(f"Edges: {edge_count:,}, Score: {score:.4f}")

        answer_diff_measurements = measure_answer_diff(
            model=model,
            test_loader=test_loader,
            circuit_outs=circuit_outs,
            prob_func="logits"
        )

        print("\nAnswer Difference Results:") 
        for item in answer_diff_measurements:
            edge_count, score = item
            print(f"Edges: {edge_count:,}, Score: {score:.4f}")

        edge_counts = []
        scores = []
        for edge_count, score in answer_diff_measurements:
            edge_counts.append(edge_count)
            scores.append(score)
        
        answer_diff_results = {
            edge_count: score
            for edge_count, score in answer_diff_measurements
        }

        with open(os.path.join(self.results_dir, f"answer_diff_results_{self.desired_relation}.json"), 'w') as f:
            json.dump(answer_diff_results, f, indent=4)
        
        falthfulness_score = dict()
        for edge_count, score in answer_diff_results.items():
            falthfulness_score[edge_count] = score
            final_score = score
        
        for edge_count, score in falthfulness_score.items():
            if final_score > 0:
                falthfulness_score[edge_count] = score / final_score * 100
            else:
                falthfulness_score[edge_count] = None
        
        with open(os.path.join(self.results_dir, f"falthfulness_score_{self.desired_relation}.json"), 'w') as f:
            json.dump(falthfulness_score, f, indent=4)
        print(f"Saved falthfulness score to {os.path.join(self.results_dir, f'falthfulness_score_{self.desired_relation}.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Discover discursive circuits for causation vs antethesis.')
    parser.add_argument('--relation', type=str, default='all',
                      help='all|AB|A_not|B_not|not_not')
    parser.add_argument('--source_relation', type=str, default='all',
help='all|AB|A_not|B_not|not_not|L1_Comparison.json|L1_Contingency.json|L3_Comparison.Contrast.json|L3_Contingency.Cause.Reason.json|L3_Contingency.Cause.Result.json')
    parser.add_argument('--dataset_name', type=str, default='RST',
                      help='The (target) dataset name to use for the experiment.')
    
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for the experiment. Warning: The experiment is very memory-intensive, so it is recommended to use a small batch size.')
    parser.add_argument('--N_sample', type=int, default=32,
                      help='Number of samples to use for the experiment')
    parser.add_argument('--mode', type=str, default='learn', choices=['learn', 'patch'],
                      help='learn (the first time) or patch (using a learned circuit)')
    parser.add_argument('--level', type=int, default=3, choices=[0, 1, 2, 3],
                      help='Level of the circuit to use, L0, L1, or L3. Only a few relation has L2, hence skipped.')
    parser.add_argument('--gentype', type=str, default="own", choices=['own', 'gendataset', 'genrel'],
                      help='Choose which type of generalization we want to test. First option is "own", where we use the in-framework own data to discovery the circuit. Second option is "gendataset", where we use a different dataset to test the circuit. Third option is "genrel", where we use a different relation, but within the same dataset.')
    parser.add_argument('--seed', type=int, default=42,
                      help='Seed for the experiment.')
    parser.add_argument('--model_mode', type=str, default="mlm", choices=['clm', 'mlm'],
                      help='Choose causal LM or masked LM')
    parser.add_argument('--data_inverse', action='store_true',
                      help='Whether to flip clean and corrupt')
    
    args = parser.parse_args()
    print('=++====++++===={}'.format(args.data_inverse))
    new_config = DC_Config(args.relation, args.source_relation, args.dataset_name, args.batch_size, args.N_sample, args.mode, args.level, args.gentype, args.seed, args.model_mode, args.data_inverse)

    new_circuit_discovery = CircuitDiscovery(new_config)
    new_circuit_discovery.run_eap()
