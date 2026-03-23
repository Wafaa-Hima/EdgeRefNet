from argparse import ArgumentParser
import torch
import utils 
from models.trainer import *

print(torch.cuda.is_available())

"""
the main function for training the CD networks
"""

def train(args):
   
    from custom_models.trainer import CDTrainer
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()

def test(args):
    from custom_models.evaluator import CDEvaluator

    test_dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                   batch_size=args.batch_size, is_train=False,
                                   split='test', dataset_name=args.dataset) 
    
    evaluator = CDEvaluator(args=args, dataloader=test_dataloader) 
    evaluator.eval_model(checkpoint_name='best_ckpt.pt', 
                         use_tta=args.use_tta if hasattr(args, 'use_tta') else False, 
                         use_multi_scale=args.use_multi_scale_eval if hasattr(args, 'use_multi_scale_eval') else False)


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='EdgeRefNet_LEVIR_CD', type=str) 
    parser.add_argument('--checkpoint_root', default='/checkpoints', type=str) 
    parser.add_argument('--vis_root', default='/vis', type=str) 
    parser.add_argument('--output_folder', default='/samples_LEVIR/predict_CD_EdgeRefNet', type=str, help='folder to save prediction images')

    # data
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str, help="Dataset type, e.g., CDDataset")
    parser.add_argument('--data_name', default='LEVIR-CD-256', type=str, help="Specific dataset name, e.g., LEVIR-CD-256")

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--shuffle_AB', default=False, type=str) 
    


    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=32, type=int) 
    parser.add_argument('--embed_dim_transformer', default=256, type=int, help="Embedding dimension for transformer components in EdgeRefNet")
    parser.add_argument('--pretrain_path', default=None, type=str, help="Path to pretrained backbone weights if not using torchvision's") 
    parser.add_argument('--multi_scale_train', default=False, type=str) 
    parser.add_argument('--multi_scale_infer', default=False, type=str) 
    parser.add_argument('--multi_pred_weights', nargs='+', type=float, default=[0.5, 0.5, 0.5, 0.8, 1.0]) 

    parser.add_argument('--net_G', default='EdgeRefNet', type=str,
                        help='EdgeRefNet | BIT | EGCTNet | ICIF_Net | ChangeFormer | SNUNet | SiamUnet_diff | SiamUnet_conc | DTCDSCN | Unet ')
    parser.add_argument('--mode', default='train_test', type=str, help='train_test | test_only')

    parser.add_argument('--loss', default='eas', type=str, help="Loss function type")

    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str, choices=['adamw', 'adam', 'sgd'])
    parser.add_argument('--lr', default=1e-4, type=float) # 1e-4
    parser.add_argument('--weight_decay', default=1e-4, type=float, help="Weight decay for optimizer. Try 1e-4, 3e-3 for AdamW.")
    parser.add_argument('--max_epochs', default=400, type=int)
    parser.add_argument('--lr_policy', default='cosine', type=str, 
                    help='linear | step | plateau | cosine')

    parser.add_argument('--lr_decay_iters', default=100, type=int, help="Parameter for 'step' policy or other custom policies")
    parser.add_argument('--lr_plateau_patience', default=10, type=int, help="Patience for ReduceLROnPlateau scheduler")



    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)
    test(args)
    # test2(args)