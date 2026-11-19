# Official Implementation of "Discursive Circuits: How Do Language Models Understand Discourse Relations?"

Official implementation for our paper: Discursive Circuits: How Do Language Models Understand Discourse Relations? EMNLP 2025. Yisong Miao, Min-Yen Kan. \
Paper PDF: [https://yisong.me/publications/emnlp25-DC-CR.pdf](https://yisong.me/publications/emnlp25-DC-CR.pdf) \
Slides: [https://yisong.me/publications/emnlp25-DC-Slides.pdf](https://yisong.me/publications/emnlp25-DC-Slides.pdf) \
Poster: [https://yisong.me/publications/emnlp25-DC-Poster.pdf](https://yisong.me/publications/emnlp25-DC-Poster.pdf)


# Installation 📀💻
We recommend using an anaconda environment for the installation:
```
git clone git@github.com:YisongMiao/Discursive-Circuits.git
conda create DC python=3.10
conda activate DC
cd Discursive-Circuits
cd scripts
pip install -r requirements.txt
```

It could take ~20 minutes or more to install the dependencies.

Aside from that, the (recently updated) Kaleido package requires further installations of Chrome and some other dependencies to be installed. Consider installing it by running the following command:

```
plotly_get_chrome
```

and

```
sudo apt update && sudo apt-get install libnss3 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2
```

## Downloads 🏕️💾

Download 1: `data.zip` \
Google Drive: https://drive.google.com/file/d/1anAq2gp1nXgst0W4to2j-JJrU64S-8YO/view?usp=share_link \

Download the file, unzip it and replace the `data` directory within this repository. Before using this dataset, we request you agree to the following:
- The dataset will not be used for commercial purposes.
- You will not distribute the raw textual data online (the data is currently zipped). This is to limit the easy availability of the data for training future AI models.

Download 2: `model_path/gpt2-medium_ft.pth` \
Google Drive: https://drive.google.com/file/d/1Yf_Xq2M--CNwSSwbVA-sMzONCJVvllQv/view?usp=share_link \

After downloading the file, please copy the `gpt2-medium_ft.pth` file to the `model_path` directory of the repository.

## Specifications 🧑‍🍳🍳
Our code is built upon the ⭐️ stellar ⭐️ open-source circuit discovery codebase [auto-circuit](https://github.com/UFO-101/auto-circuit/). Please refer to their repository for more details. Our IOI data in the `data/misc/ioi_vanilla_template_prompts.json` is also borrowed from their repository.

We have made a few customizations to the original `auto-circuit` codebase to fit our CuDR task in the `auto-circuit-DC` directory. In our `circuit_discovery.py` script, we have specified using our customized package `auto-circuit-DC`. Please directly run our script (no action required from our users). 

Please edit the following parameters in the `scripts/DC_config.py` file to customize your experiment:
```
thr = 6 # Threshold for the EAP plot (the higher the threshold, the fewer edges are considered).
model_path = 'model_path/gpt2-medium_ft.pth' # Path to the model checkpoint.
model_name = 'gpt2-medium' # Name of the model.
edge_series = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000, 100000, 200000, 231877] # Edge series for patching experiments (the higher the edge count, the more edges are considered).
```

## Hardware recommendation 💻💻
We recommend using one single GPU with at least 48 GB of memory:
- It is not trivial to customize the code to run multiple GPUs. This is because [the TransformerLens package](https://github.com/TransformerLensOrg/TransformerLens) (the backbone of auto-circuit) has not yet fully supported multiple GPUs.
- We recommend a batch_size of 1 for circuit discovery experiments. This is because the CuDR task has longer inputs (consisting of two arg2 and one arg1). In our practice, batch_size = 1 is already the maximum for our local GPU (48 GB memory).


# Discover the circuit for one discourse relation with one line command 🏃🧪🔬

Would you like to discover the circuit for any discourse relation? You are welcome to use this one line command!

![DC Abstract](media/dc-abstract.png)

**Command line:**
```
cd Discursive-Circuits
python scripts/circuit_discovery.py --relation <relation> --source_relation <source_relation> --dataset_name <dataset_name> --dataset_name_source <dataset_name_source> --batch_size <batch_size> --N_sample <N_sample> --mode <mode> --level <level> --gentype <gentype> --seed <seed>
```

- `relation`: The (target) relation type to analyze. Options: e.g. Contingency.Cause.Result, Comparison.Concession.Arg2-as-denier, Expansion.Instantiation.Arg2-as-instance, Temporal.Asynchronous.Precedence, and other relations in PDTB / GDTB / RST / SDRT.
- `source_relation`: The source relation (whose circuit is used on the target relation). For circuit discovery (i.e. `mode = learn`), the source relation is the same as the target relation. For circuit patching (i.e. `mode = patch`), the source relation is the relation whose circuit is used on the target relation.
- `dataset_name`: The (target) dataset name to use for the experiment. Options: e.g. PDTB, GDTB, RST, SDRT.
- `dataset_name_source`: The (source) dataset name, the circuit is already learned on this dataset. Options: e.g. PDTB, GDTB, RST, SDRT.
- `batch_size`: The batch size for the experiment.
- `N_sample`: The number of samples to use for the experiment.
- `mode`: The mode for the experiment. Options: e.g. learn, patch.
- `level`: The level of the circuit to use. Options: e.g. 0, 1, 3.
- `gentype`: The type of generalization to use. Options: e.g. own, gendataset, genrel.
- `seed`: The seed for the experiment. 


For example, to discover the circuit for the relation `Temporal.Asynchronous.Precedence`, run the following command (the --gentype is `own`):

```
cd Discursive-Circuits
python scripts/circuit_discovery.py --relation Temporal.Asynchronous.Precedence --source_relation Temporal.Asynchronous.Precedence --dataset_name PDTB --dataset_name_source PDTB --batch_size 1 --N_sample 32 --mode learn --level 3 --gentype own --seed 42
```

The results will be saved at `patching_experiments/data/results-L3-own/results_PDTB_32_PDTB_Temporal.Asynchronous.Precedence_learn_42/Temporal.Asynchronous.Precedence` (the name of the directory will store the relavant parameters for the experiment). 

We have produced the following files under the directory:
- `top_edge_names_raw_{relation}.json`: The top edges of the circuit for the relation, sorted by the decending order of the absolute value of the scores. The key is the edge name and the value is the score of the edge.
```json
{
    "A20.2->Resid End": 41.37675476074219,
    "A21.1->Resid End": 34.7537727355957,
    "A19.2->Resid End": 31.59067153930664,
    "A20.9->Resid End": 22.520381927490234,
    "MLP 23->Resid End": -21.701507568359375,
    "MLP 21->Resid End": 21.353221893310547,
    "MLP 19->Resid End": 19.05948257446289,
    "MLP 22->MLP 23": -13.985919952392578,
    "MLP 0->MLP 1": 13.15355110168457,
    "A17.13->Resid End": 12.307862281799316, 
    // ... (other edges)
}
```

- `correct_ans_percent_results_{relation}.json`: The results of the correct answer percentage.
- `answer_diff_results_{relation}.json`: The results of the logits differences between the correct answer and the incorrect answer.
- `falthfulness_score_{relation}.json`: The falthfulness scores of the circuit, which is the normalized logits differences between the correct answer and the incorrect answer (by the full model's logits differences). 
As we can see from the json file below, the falthfulness score starts from a negative value (this is because the model has been steered towards the counterfactual answer) and increases to 100.0 when the number of edges is 231877. This means that the circuit is fully restored back to the original answer.

```json
{
    "0": -53.91910639612152,
    "1": -42.23724833916364,
    "2": -30.602846961202,
    "3": -20.547302787173344,
    "4": -17.240843704548713,
    "5": -20.917571819668158,
    "6": -20.844045726508234,
    "7": -11.70630332544631,
    "8": -11.706306005554685,
    "9": 1.3107673032352873,
    "10": 3.391534432799552,
    "30": 53.45025359721466,
    "50": 91.55501067165552,
    "100": 95.50375370618787,
    "200": 99.73672759410638,
    "300": 94.69382495526095,
    "500": 102.92360158109241,
    "1000": 99.16021484177551,
    "2000": 98.2235491260113,
    "3000": 97.18684032482484,
    "5000": 97.94667249000615,
    "10000": 99.73836782043189,
    "20000": 97.62200416145788,
    "30000": 97.52985131509165,
    "50000": 98.65659031681132,
    "100000": 98.57903870087213,
    "200000": 100.01208192855454,
    "231877": 100.0
}
```
- `circuit-{relation}-h.pdf`: Finally, we also store the PDF format (as a vector graphic) for the visualization of the circuit for a given relation.

## Patching the circuit with one line command 🏓🏀⚽️

After the circuit discovery is complete, we can patch the circuit to solve the CuDR task for other relations / frameworks. 

We support cross-framework patching. For example, we can use the L3 circuits of `Contingency.Cause.Result` (of PDTB) to solve the CuDR task for the relation `causal-result_r` (of RST). Run the following command (the --gentype is `gendataset`):
```
cd Discursive-Circuits
python scripts/circuit_discovery.py --relation causal-result_r --source_relation Contingency.Cause.Result --dataset_name RST --dataset_name_source PDTB --batch_size 1 --N_sample 32 --mode patch --level 3 --gentype gendataset --seed 42
```

We also support intra-framework patching. For example, we can use the L3 circuits of `Contingency.Cause.Result` (of PDTB) to solve the CuDR task for the relation `Expansion.Conjunction` (of PDTB). Run the following command (the --gentype is `genrel`):
```
cd Discursive-Circuits
python scripts/circuit_discovery.py --relation Expansion.Conjunction --source_relation Contingency.Cause.Result --dataset_name PDTB --dataset_name_source PDTB --batch_size 1 --N_sample 32 --mode patch --level 3 --gentype genrel --seed 42
```

The output files are similar to the previous section. Aside, users can explore different configs on the `--level` (supports 0, 1, 3 for the patching experiments) and altering random seeds for different samples. 


# Introduction to the CuDR Dataset 🔡🔠
We'd first acknowledge earlier efforts by the [PDTB corpus](https://catalog.ldc.upenn.edu/LDC2019T05); [DISRPT shared task](https://github.com/disrpt/sharedtask2023); [GDTB](https://github.com/gucorpling/gum2pdtb); [RST (GUM corpus)](https://github.com/amir-zeldes/gum); [SDRT corpus](https://github.com/disrpt/sharedtask2023). Our work builds upon theses excellent prior efforts on discourse language resources. 

The file structure of the CuDR dataset is as follows:

```
data/CuDR_data/
├── CuDR_data_GDTB/
├── CuDR_data_PDTB/
├── CuDR_data_RST/
└── CuDR_data_SDRT/
```

Under each directory, there are the following files:

```
├── rel1_train.json
├── rel1_val.json
├── rel2_train.json
├── rel2_val.json
├── ...
├── relN_train.json
├── relN_val.json
```

where `relN` is the N-th relation in the corresponding corpus. 

The data structure of all json files are the same as the following:
```
{
    "prompts": [
        {
            "clean": "Complete the sentence using one of the two provided options verbatim, ensuring coherence.\n\nthey prefer clear justifications for financial decisions to minimize uncertainty\nor\nthe obvious reason was (for British Air) to buy a stake in United Airlines\n\nSentence to complete: \"Fund managers don't like to have rights issues that don't have an obvious reason, but <|ANSWER|>",
            "corrupt": "Complete the sentence using one of the two provided options verbatim, ensuring coherence.\n\nthey prefer clear justifications for financial decisions to minimize uncertainty\nor\nthe obvious reason was (for British Air) to buy a stake in United Airlines\n\nSentence to complete: \"Fund managers don't like to have rights issues that don't have an obvious reason, because <|ANSWER|>",
            "answers": [
                " the"
            ],
            "wrong_answers": [
                " they"
            ]
        }
        ...
    ]
}
```
Our prompts are in multiple-choice format. Specifically,
- The clean prompt is structured as `<Instruction> <option 1> or <option 2> Sentence to complete: <Arg1> <Connective original> ... `
- The corrupt prompt is structured as `<Instruction> <option 1> or <option 2> Sentence to complete: <Arg1> <Connective corrupt> ... `
- `Arg2` and `Arg2'` are randomly ordered to fill in `<option 1>` and `<option 2>` respectively.

**Dataset Statistics:**

**PDTB:**
| Relation | Train | Val | Total |
|----------|-------|-----|-------|
| Comparison.Concession.Arg2-as-denier | 234 | 239 | 473 |
| Comparison.Contrast | 240 | 245 | 485 |
| Contingency.Cause.Reason | 241 | 246 | 487 |
| Expansion.Level-of-detail.Arg1-as-detail | 217 | 223 | 440 |
| Expansion.Level-of-detail.Arg2-as-detail | 239 | 244 | 483 |
| Contingency.Cause.Result | 240 | 246 | 486 |
| Expansion.Conjunction | 237 | 243 | 480 |
| Expansion.Instantiation.Arg2-as-instance | 238 | 244 | 482 |
| Temporal.Asynchronous.Precedence | 230 | 235 | 465 |
| Temporal.Synchronous | 239 | 244 | 483 |
| Expansion.Equivalence | 241 | 246 | 487 |
| Expansion.Substitution.Arg2-as-subst | 212 | 218 | 430 |
| Temporal.Asynchronous.Succession | 163 | 169 | 332 |
| Total | 2971 | 3042 | 6013 |

**GDTB:**
| Relation | Train | Val | Total |
|----------|-------|-----|-------|
| Comparison.Concession.Arg2-as-denier | 235 | 240 | 475 |
| Comparison.Contrast | 224 | 229 | 453 |
| Contingency.Cause.Reason | 234 | 239 | 473 |
| Expansion.Level-of-detail.Arg2-as-detail | 231 | 236 | 467 |
| Contingency.Cause.Result | 227 | 232 | 459 |
| Expansion.Conjunction | 218 | 224 | 442 |
| Expansion.Instantiation.Arg2-as-instance | 218 | 224 | 442 |
| Temporal.Asynchronous.Precedence | 230 | 235 | 465 |
| Temporal.Synchronous | 234 | 239 | 473 |
| Expansion.Equivalence | 186 | 192 | 378 |
| Expansion.Substitution.Arg2-as-subst | 110 | 116 | 226 |
| Temporal.Asynchronous.Succession | 217 | 223 | 440 |
| Total | 2564 | 2629 | 5193 |

**RST:**
| Relation | Train | Val | Total |
|----------|-------|-----|-------|
| joint-list_m | 240 | 246 | 486 |
| joint-sequence_m | 231 | 236 | 467 |
| elaboration-additional_r | 229 | 235 | 464 |
| context-circumstance_r | 232 | 238 | 470 |
| adversative-concession_r | 235 | 240 | 475 |
| causal-cause_r | 239 | 244 | 483 |
| causal-result_r | 231 | 236 | 467 |
| adversative-contrast_m | 239 | 244 | 483 |
| explanation-justify_r | 234 | 239 | 473 |
| context-background_r | 230 | 236 | 466 |
| joint-other_m | 220 | 225 | 445 |
| adversative-antithesis_r | 152 | 157 | 309 |
| explanation-evidence_r | 180 | 185 | 365 |
| evaluation-comment_r | 95 | 100 | 195 |
| explanation-motivation_r | 136 | 141 | 277 |
| restatement-repetition_m | 126 | 131 | 257 |
| joint-sequence_r | 66 | 72 | 138 |
| Total | 3315 | 3405 | 6720 |

**SDRT:**
| Relation | Train | Val | Total |
|----------|-------|-----|-------|
| Acknowledgement | 224 | 229 | 453 |
| Comment | 240 | 245 | 485 |
| Continuation | 236 | 242 | 478 |
| Contrast | 242 | 248 | 490 |
| Correction | 69 | 74 | 143 |
| Elaboration | 242 | 248 | 490 |
| Explanation | 243 | 249 | 492 |
| Narration | 55 | 60 | 115 |
| Parallel | 79 | 85 | 164 |
| Result | 244 | 249 | 493 |
| Total | 1874 | 1929 | 3803 |

Note: We used a held-out portion of the PDTB CuDR dataset (approximately half of the full PDTB data) to warm up the model for the CuDR task. Stats on this portion are not reported here. \
In this release version, we also excluded five data instances per relation from the complete dataset to ensure that the training and validation splits are non-overlapping (because each original Arg1 has five possible counterfactual Arg2 instances).


## Future Enhancements 🔮🔮
(We will try to integrate these enhancements items (probably) before the end of this year.)
- Include bash scripts for running experiments in batch mode. 
- Our code structure now only allows existing discourse relations / frameworks. We will update the code structure to allow customizing for new discourse relations / frameworks with a user-friendly pipeline. Hopefully the users only have to prepare their own CuDR data in the correct format and they can run the experiments with one line command.


# Citations 📜📝
If you find our work interesting, you are most welcome to try our dataset/codebase. \
Please kindly cite our research if you have used our dataset/codebase:

```
@inproceedings{miao-kan-2025-discursive,
    title = "Discursive Circuits: How Do Language Models Understand Discourse Relations?",
    author = "Miao, Yisong  and
      Kan, Min-Yen",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1657/",
    pages = "32558--32577",
    ISBN = "979-8-89176-332-6",
    abstract = "Which components in transformer language models are responsible for discourse understanding? We hypothesize that sparse computational graphs, termed as discursive circuits, control how models process discourse relations. Unlike simpler tasks, discourse relations involve longer spans and complex reasoning. To make circuit discovery feasible, we introduce a task called Completion under Discourse Relation (CuDR), where a model completes a discourse given a specified relation. To support this task, we construct a corpus of minimal contrastive pairs tailored for activation patching in circuit discovery. Experiments show that sparse circuits ({\ensuremath{\approx}}0.2{\%} of a full GPT-2 model) recover discourse understanding in the English PDTB-based CuDR task. These circuits generalize well to unseen discourse frameworks such as RST and SDRT. Further analysis shows lower layers capture linguistic features such as lexical semantics and coreference, while upper layers encode discourse-level abstractions. Feature utility is consistent across frameworks (e.g., coreference supports Expansion-like relations)."
}
```

## Contact 📤📥
If you have questions or bug reports, please raise an issue or contact us directly via the email:\
Email address: 🐿@🐰\
where 🐿️=`yisong`, 🐰=`comp.nus.edu.sg`

## License 🪪🪪
For the open-sourced prior work that we build upon, please refer to the license of the original repositories. \
Our Discursive Circuits software and data has a CC-BY 4.0 license.
