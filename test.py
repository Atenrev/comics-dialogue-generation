import argparse
import yaml
import random
import torch
import numpy as np

from src.models.modeling_vlt5 import VLT5
from src.tokenizers.vlt5_tokenizers import VLT5TokenizerFast


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def config_str(self):
        return str(self.__dict__)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += self.config_str
        return config_str

    def save(self, path):
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            kwargs = yaml.load(f)

        return Config(**kwargs)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_optimizer(optim, verbose=False):
    # Bind the optimizer
    if optim == 'rms':
        if verbose:
            print("Optimizer: Using RMSProp")
        optimizer = torch.optim.RMSprop
    elif optim == 'adam':
        if verbose:
            print("Optimizer: Using Adam")
        optimizer = torch.optim.Adam
    elif optim == 'adamw':
        if verbose:
            print("Optimizer: Using AdamW")
        # optimizer = torch.optim.AdamW
        optimizer = 'adamw'
    elif optim == 'adamax':
        if verbose:
            print("Optimizer: Using Adamax")
        optimizer = torch.optim.Adamax
    elif optim == 'sgd':
        if verbose:
            print("Optimizer: SGD")
        optimizer = torch.optim.SGD
    else:
        assert False, "Please add your optimizer %s in the list." % optim

    return optimizer


def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=9595, help='random seed')

    # Data Splits
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default=None)
    parser.add_argument('--test_only', action='store_true')

    parser.add_argument('--submit', action='store_true')

    # Quick experiments
    parser.add_argument('--train_topk', type=int, default=-1)
    parser.add_argument('--valid_topk', type=int, default=-1)

    # Checkpoint
    parser.add_argument('--output', type=str, default='snap/test')
    parser.add_argument('--load', type=str, default=None, help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--from_scratch', action='store_true')

    # CPU/GPU
    parser.add_argument("--multiGPU", action='store_const', default=False, const=True)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument('--local_rank', type=int, default=-1)

    # Model Config
    parser.add_argument('--backbone', type=str, default='t5-base')
    parser.add_argument('--tokenizer', type=str, default=None)

    parser.add_argument('--feat_dim', type=float, default=2048)
    parser.add_argument('--pos_dim', type=float, default=4)

    parser.add_argument('--use_vision', default=True, type=str2bool)
    parser.add_argument('--use_vis_order_embedding', default=True, type=str2bool)
    parser.add_argument('--use_vis_layer_norm', default=True, type=str2bool)
    parser.add_argument('--individual_vis_layer_norm', default=True, type=str2bool)
    parser.add_argument('--share_vis_lang_layer_norm', action='store_true')

    parser.add_argument('--n_boxes', type=int, default=36)
    parser.add_argument('--max_n_boxes', type=int, default=36)
    parser.add_argument('--max_text_length', type=int, default=100)

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--valid_batch_size', type=int, default=None)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument("--losses", default='lm,obj,attr,feat', type=str)

    parser.add_argument('--log_train_accuracy', action='store_true')

    parser.add_argument('--n_ground', type=int, default=1)
    parser.add_argument("--wordMaskRate", dest='word_mask_rate', default=0.15, type=float)
    parser.add_argument("--objMaskRate", dest='obj_mask_rate',default=0.15, type=float)

    # Inference
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--gen_max_length', type=int, default=20)

    # Data
    parser.add_argument('--caption_only', action='store_true')
    parser.add_argument('--coco_only', action='store_true')
    parser.add_argument('--caption_cocoonly', default=True, type=str2bool)

    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--oscar_tags', action='store_true')

    parser.add_argument('--prefix', type=str, default=None)

    # Pretraining
    parser.add_argument('--ground_upsample', type=int, default=1)
    parser.add_argument('--ground_weight', type=int, default=1)
    parser.add_argument('--itm_cocoonly', default=True, type=str2bool)
    parser.add_argument('--single_vqa_prefix', action='store_true')

    # COCO Caption
    parser.add_argument('--no_prefix', action='store_true')

    # VQA
    parser.add_argument("--raw_label", action='store_true')
    parser.add_argument("--answer_normalize", action='store_true')
    parser.add_argument("--classifier", action='store_true')
    parser.add_argument("--test_answerable", action='store_true')

    # RefCOCOg
    parser.add_argument('--RefCOCO_GT', action='store_true')
    parser.add_argument('--RefCOCO_BUTD', action='store_true')
    parser.add_argument("--shuffle_boxes", action='store_true')

    # Multitask
    parser.add_argument("--multitask_sampling", type=str, default='roundrobin')
    parser.add_argument("--tasks", type=str, default='')

    # Etc.
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument("--dry", action='store_true')

    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    # For interative engironmnet (ex. jupyter)
    else:
        args = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(args)
    kwargs.update(optional_kwargs)

    args = Config(**kwargs)

    # Bind optimizer class.
    verbose = False
    args.optimizer = get_optimizer(args.optim, verbose=verbose)

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


def create_config(args):
    from transformers import T5Config

    config = T5Config.from_pretrained(args.backbone)

    config.feat_dim = args.feat_dim
    config.pos_dim = args.pos_dim
    config.n_images = 2
    config.vocab_size = 32200

    config.use_vis_order_embedding = args.use_vis_order_embedding

    config.dropout_rate = args.dropout
    config.dropout = args.dropout
    config.attention_dropout = args.dropout
    config.activation_dropout = args.dropout

    config.use_vis_layer_norm = args.use_vis_layer_norm
    config.individual_vis_layer_norm = args.individual_vis_layer_norm
    config.losses = args.losses

    config.share_vis_lang_layer_norm = args.share_vis_lang_layer_norm
    config.classifier = args.classifier

    return config


# Parse configuration
args = parse_args(
    backbone='t5-base', # Backbone architecture
    load='./pretrained_weights/vlt5_epoch30.pth', # Pretrained checkpoint
    parse=False, # False for interactive env (ex. jupyter)
)
# Assign GPU
args.gpu = 0

config = create_config(args)
model = VLT5(config)

def load_state_dict(state_dict_path, loc='cpu'):
    state_dict = torch.load(state_dict_path, map_location=loc)
    # Change Multi GPU to single GPU
    original_keys = list(state_dict.keys())
    for key in original_keys:
        if key.startswith("module."):
            new_key = key[len("module."):]
            state_dict[new_key] = state_dict.pop(key)
    return state_dict

state_dict = load_state_dict(args.load, 'cpu')

original_keys = list(state_dict.keys())
for key in original_keys:
    if key.startswith("vis_encoder."):
        new_key = 'encoder.' + key[len("vis_encoder."):]
        state_dict[new_key] = state_dict.pop(key)

    if key.startswith("model.vis_encoder."):
        new_key = 'model.encoder.' + key[len("model.vis_encoder."):]
        state_dict[new_key] = state_dict.pop(key)

results = model.load_state_dict(state_dict, strict=False)
print('Model loaded from ', args.load)
print(results)

tokenizer = VLT5TokenizerFast.from_pretrained(
            args.backbone,
            max_length=args.max_text_length,
            do_lower_case=args.do_lower_case,
            )

input_ids = tokenizer.encode('Hello, world!')

# Extract features with https://github.com/j-min/VL-T5/blob/main/feature_extraction/coco_proposal.py

output = model(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            return_dict=True
        )