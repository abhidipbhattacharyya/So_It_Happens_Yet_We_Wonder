# This file contains the configuration for the circuit discovery experiment.

class DC_Config:
    def __init__(self, desired_relation, source_relation, dataset_name,  batch_size, N_sample, mode, level, gentype, seed, model_mode, data_inverse):
        self.desired_relation = desired_relation
        self.source_relation = source_relation
        self.dataset_name = dataset_name
        #assert self.source_relation in ['all', 'AB', 'A_not', 'B_not', 'not_not']
        #assert self.desired_relation in ['all', 'AB', 'A_not', 'B_not', 'not_not']
        
        #self.dataset_name_source = dataset_name_source
        self.batch_size = batch_size
        self.N_sample = N_sample
        self.mode = mode
        self.level = level
        self.gentype = gentype
        self.seed = seed
        self.top_edges_dir = 'data/L_edges/'
        
        #self.config_relation_dir = 'patching_experiments/config/'
        #self.ioi_fp = 'data/misc/ioi_vanilla_template_prompts.json'
        self.model_mode = model_mode#'clm' if 'gpt2' in self.model_path else 'mlm'
        if self.model_mode == 'clm':
            self.model_name = 'gpt2-small' #gpt2-medium
            self.model_path = f'/srv/data1/abhidipbhatt/models/GPT2/{self.model_name}-FT_32B/model.pth' # Path to the model checkpoint.
            self.if_mlm = False
        else:
            self.model_name = 'bert-large-uncased'
            self.model_path ="/srv/data1/abhidipbhatt/models/bert-large/bert_ft_pooler/model.pth"
            self.if_mlm= True 
            
        self.patching_data_train_fp = 'data/training_data_{}_processed4DC/sentence_{}.json'
        self.patching_data_val_fp = 'data/val_data_{}_processed4DC/sentence_{}.json'
        self.results_dir = "patching_experiments/data/results-L{}-{}/results_{}_{}_{}_{}_{}_{}/{}/"

        # Following parameters are up to the user's choice.
        self.thr = 6 # Threshold for the EAP plot (the higher the threshold, the fewer edges are considered).
        
        #self.model_name = 'gpt2-medium' # Name of the model.
        if 'small' in self.model_name :
            self.edge_series = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 32491]   
        elif 'medium' in self.model_name:
            self.edge_series = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000, 100000, 200000, 231877] # Edge series for the EAP plot (the higher the edge count, the more edges are considered).

        self.data_inverse = data_inverse
