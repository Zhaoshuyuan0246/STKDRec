import os
import time
import torch
import argparse
import sys
os.chdir(sys.path[0])
import torch.nn as nn
import torch.nn.functional as F
from model import SASRec
from utils import *
from tqdm import tqdm
from EarlyStopping import EarlyStopping
from model_gnn import GCL4SR

def set_seed(seed=42):
    random.seed(seed)  # Python内置随机数生成器的种子
    np.random.seed(seed)  # NumPy随机数生成器的种子
    torch.manual_seed(seed)  # PyTorch的CPU随机数生成器的种子
    torch.cuda.manual_seed(seed)  # 如果使用GPU，则设置CUDA随机数生成器的种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，则为所有GPU设置随机数生成器的种子
    torch.backends.cudnn.deterministic = True  # 确保卷积算法的确定性
    torch.backends.cudnn.benchmark = False  # 禁用自动优化，确保结果可复现

def find_first_file_with_prefix(directory, prefixes):
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.startswith(prefix) and file.endswith('.pt') for prefix in prefixes):
                return os.path.join(root, file)

parser = argparse.ArgumentParser()
# Dataset
parser.add_argument('--city', default='sanya', type=str) 
parser.add_argument('--dataset', default='elm_', type=str) # elm_sanya, elm_small

# Student Model
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--train_dir', default='default', type=str)
parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=128, type=int)
parser.add_argument('--hidden_units', default=256, type=int)    # 32, 64, 128, 256
parser.add_argument('--num_blocks', default=2, type=int)    # 2, 4, 6
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=2, type=int)    # 1, 2, 4
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.01, type=float)
parser.add_argument('--device', default='cuda:7', type=str)

# Spatial-Temporal Knowledge Distllition
parser.add_argument('--geo_hash', default=False, choices=[True, False], type=str) 
parser.add_argument('--distances', default=False, choices=[True, False], type=str) 
parser.add_argument('--sptia', default=True, choices=[True, False], type=str) 
parser.add_argument('--kd', default='pos',choices=['pos', 'all', None], type=str) 
parser.add_argument('--loss_type', default='bce', choices=['bce', 'cross'], type=str)
parser.add_argument('--only_teacher', default=False, type=str)
parser.add_argument('--lamada', default=0.2, type=float)


# Teacher Model GNNs
parser.add_argument("--pre_train_path",default='./pre_train/', type=str)
parser.add_argument("--gnn_dataset", default='./datasets/', type=str)
parser.add_argument('--gnn_hidden_units', default=256, type=int)    # 32, 64, 128, 256
parser.add_argument("--use_renorm", type=bool, default=True, help="use re-normalize when build witg")
parser.add_argument("--use_scale", type=bool, default=False, help="use scale when build witg")
parser.add_argument("--fast_run", type=bool, default=True, help="can reduce training time and memory")
parser.add_argument("--sample_size", default=[20, 20], type=int, nargs='+', help='gnn sample')
parser.add_argument("--sample_type", default='sparse', choices=['dense', 'sparse'], type=str)
# Teacher Model Transformer
parser.add_argument("--num_attention_heads", default=2, type=int, help="number of heads")
parser.add_argument("--hidden_act", default="gelu", type=str, help="activation function")

# Optimizer
# parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
parser.add_argument("--lr_dc", type=float, default=0.7, help='learning rate decay.')
parser.add_argument("--lr_dc_step", type=int, default=5,
                        help='the number of steps after which the learning rate decay.')
parser.add_argument("--weight_decay", type=float, default=5e-5, help="weight_decay of adam")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

# Test
parser.add_argument('--seed', default=24, type=str) 

parser.add_argument('--fus', default='add', choices=['kd', 'add', 'cat', 'plus', 'None'], type=str) 
parser.add_argument('--inference_only', default=False, type=str)
parser.add_argument('--test', default=True, type=str) 
parser.add_argument('--log', default='log_3.txt', type=str) 


args = parser.parse_args()

args.save_path = args.dataset + args.city
args.gnn_dataset = args.gnn_dataset + args.city
args.pre_train_path = args.pre_train_path

set_seed(args.seed)

if not os.path.isdir(args.save_path + '_' + args.train_dir):
    os.makedirs(args.save_path + '_' + args.train_dir)
with open(os.path.join(args.save_path + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # global dataset
    dataset = data_partition(args)

    [u, user_train, user_valid, user_test, geo_train, geo_val, geo_test, dis_train, dis_val, dis_test, usernum, itemnum, geonum, disnum] = dataset

    args.item_size = itemnum
    args.user_size = usernum

    if args.only_teacher == True:
        global_graph = torch.load(args.gnn_dataset + '/' + args.city + '_witg.pt')
        gcn_model = GCL4SR(args=args, global_graph=global_graph)
        print('Load GCN Model')
        args.pre_train_model_path = args.pre_train_path + '/GCL4SR-' + args.city + str(args.sample_size) + '_'
        
        args.pre_train_model_path = find_first_file_with_prefix(args.pre_train_path, args.pre_train_model_path)
        if os.path.exists(args.pre_train_path):
            gcn_model.load_state_dict(torch.load(args.pre_train_model_path))
        else:
            for name, param in gcn_model.named_parameters():
                try:
                    torch.nn.init.xavier_normal_(param.data)
                except:
                    pass # just ignore those failed init layers
        gcn_model.to(args.device)
        print('Load GCN Model:' + str(args.pre_train_model_path))
    
    # Process GCN
    if args.kd != None:
        global_graph = torch.load(args.gnn_dataset + '/' + args.city + '_witg.pt')
        gcn_model = GCL4SR(args=args, global_graph=global_graph)
        print('Load GCN Model')
        args.pre_train_model_path = args.pre_train_path + '/GCL4SR-' + args.city
        # print(args.pre_train_model_path)
        # args.pre_train_model_path = find_first_file_with_prefix(args.pre_train_path, args.pre_train_model_path)
        args.pre_train_model_path = args.pre_train_model_path + ".pt"
        gcn_model.load_state_dict(torch.load(args.pre_train_model_path, map_location=args.device))

        # expanded_embedding = nn.Embedding(args.item_size+1, args.gnn_hidden_units, padding_idx=0)
        # with torch.no_grad():
        #     expanded_embedding.weight[:args.item_size, :] = gcn_model.item_embeddings.weight
        # gcn_model.item_embeddings = expanded_embedding
        # 冻结教师模型参数
        for param in gcn_model.parameters():
            param.requires_grad = False
        gcn_model.to(args.device)
        print('Load GCN Model:' + str(args.pre_train_model_path))


    num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)
    cc = 0.0
    for i in range(len(user_train)):
        cc += len(user_train[i])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    # f = open(os.path.join(args.save_path + '_' + args.train_dir, 'log_3.txt'), 'w')
    f = open(os.path.join(args.save_path + '_' + args.train_dir + '_' + args.fus +'+' +args.log), 'w')

    
    # Process Attention
    sampler = WarpSampler(u, user_train, geo_train, dis_train, usernum, itemnum, geonum, disnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=1, seed=args.seed)
    model = SASRec(usernum, itemnum, geonum, disnum, args).to(args.device) # no ReLU activation in original SASRec implementation?
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total update params: %.2f' % total_params)

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass # just ignore those failed init layers
    
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)
    
    model.train() # enable model training
    
    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)), strict=False)
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb; pdb.set_trace()
            
    
    if args.inference_only:
        model.eval()
        t_test = evaluate(model, dataset, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
    
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    kd_criterion = torch.nn.KLDivLoss(log_target=False, reduction='batchmean')
    cross_criterion = nn.CrossEntropyLoss()

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    
    lamada = args.lamada
    T = 0.0
    t0 = time.time()
    temp = 7    # 1， 3， 5， 7， 9

    if args.city == 'taiyuan':
        save_path = 'elm_taiyuan_default/elm__date=2024-12-02_12-30-26.epoch=200.lr=0.001.layer=2.head=16.hidden=256.maxlen=128.pth'
    if args.city == 'sanya':
        save_path = 'elm_sanya_default/elm__date=2024-12-02_08-58-01.epoch=200.lr=0.001.layer=2.head=16.hidden=256.maxlen=128.pth'
    if args.city == 'wuhan':
        save_path = 'elm_wuhan_default/elm__date=2024-12-02_04-28-49.epoch=200.lr=0.001.layer=2.head=16.hidden=256.maxlen=128.pth'


    # 加载最佳模型权重
    model.load_state_dict(torch.load(save_path, map_location=args.device))
    print(f"Model loaded from {save_path}")

    # 设置模型为评估模式
    model.eval()
    print('Final test', end='')
    NDCG_test, HR_test = evaluate(model, gcn_model, dataset, args)
    print('test (NDCG@1: %.4f, Recall@1: %.4f; NDCG@5: %.4f, Recall@5: %.4f; NDCG@10: %.4f, Recall@10: %.4f; NDCG@20: %.4f, Recall@20: %.4f; NDCG@50: %.4f, Recall@50: %.4f)'
            % (NDCG_test[0], HR_test[0], NDCG_test[1], HR_test[1], NDCG_test[2], HR_test[2], NDCG_test[3], HR_test[3], NDCG_test[4], HR_test[4]))
