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
parser.add_argument('--city', default='taiyuan', type=str) 
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
parser.add_argument('--num_heads', default=32, type=int)    # 1, 2, 4
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.01, type=float)
parser.add_argument('--device', default='cuda:5', type=str)

# Spatial-Temporal Knowledge Distllition
parser.add_argument('--geo_hash', default=False, choices=[True, False], type=bool) 
parser.add_argument('--distances', default=False, choices=[True, False], type=bool) 
parser.add_argument('--sptia', default=True, choices=[True, False], type=bool) 
parser.add_argument('--kd', default='all',choices=['pos', 'all', None], type=str) 
parser.add_argument('--loss_type', default='bce', choices=['bce', 'cross'], type=str)
parser.add_argument('--only_teacher', default=False, type=str)
parser.add_argument('--lamada', default=0.5, type=float)


# Teacher Model GNNs
parser.add_argument("--pre_train_path",default='/data/ZhaoShuyuan/Zhaoshuyuan/ELEME/Our_model_final/STKD/pre_train/', type=str)
parser.add_argument("--gnn_dataset", default='/data/ZhaoShuyuan/Zhaoshuyuan/ELEME/Our_model_final/STKD/datasets/', type=str)
parser.add_argument('--gnn_hidden_units', default=256, type=int)    # 32, 64, 128, 256
parser.add_argument("--use_renorm", type=bool, default=True, help="use re-normalize when build witg")
parser.add_argument("--use_scale", type=bool, default=False, help="use scale when build witg")
parser.add_argument("--fast_run", type=bool, default=True, help="can reduce training time and memory")
parser.add_argument("--sample_size", default=[10, 10], type=int, nargs='+', help='gnn sample')
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
# parser.add_argument('--seed', default=3407, type=int)
parser.add_argument('--seed', default=2024, type=int)
# parser.add_argument('--seed', default=42, type=str) 
# parser.add_argument('--seed', default=114514, type=str) 
# parser.add_argument('--seed', default=24, type=str) 

parser.add_argument('--fus', default='kd', choices=['kd', 'add', 'cat', 'plus', 'None'], type=str) 
parser.add_argument('--inference_only', default=True, type=str)
parser.add_argument('--test', default=False, type=str) 
parser.add_argument('--log', default='log_8.txt', type=str) 


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
    
    # Process GCN
    if args.kd != None:
        global_graph = torch.load(args.gnn_dataset + '/' + args.city + '_witg.pt')
        gcn_model = GCL4SR(args=args, global_graph=global_graph)
        print('Load GCN Model')
        args.pre_train_model_path = args.pre_train_path + '/GCL4SR-' + args.city
        # print(args.pre_train_model_path)
        # args.pre_train_model_path = find_first_file_with_prefix(args.pre_train_path + '/' + args.city, args.pre_train_model_path)
        args.pre_train_model_path = args.pre_train_model_path + ".pt"
        # args.pre_train_model_path = '/data/ZhaoShuyuan/Zhaoshuyuan/ELEME/Our_model_final/STKD/pre_train/wuhan/GCL4SR-wuhan-[5, 5]_21-01-00.pt'
        # args.pre_train_model_path = '/data/ZhaoShuyuan/Zhaoshuyuan/ELEME/Our_model_final/STKD/pre_train/wuhan/GCL4SR-wuhan-[5, 5]_21-00-12.pt'
        
        # args.pre_train_model_path = '/data/ZhaoShuyuan/Zhaoshuyuan/ELEME/Our_model_final/STKD/pre_train/GCL4SR-sanya-[20, 20]_21-01-07.pt'
        # args.pre_train_model_path = '/data/ZhaoShuyuan/Zhaoshuyuan/ELEME/Our_model_final/STKD/pre_train/GCL4SR-sanya-[20, 20]_21-00-30.pt'
 
        # args.pre_train_model_path = '/data/ZhaoShuyuan/Zhaoshuyuan/ELEME/Our_model_final/STKD/pre_train/GCL4SR-taiyuan-[10, 10]_21-00-23.pt'
        # args.pre_train_model_path = '/data/ZhaoShuyuan/Zhaoshuyuan/ELEME/Our_model_final/STKD/pre_train/GCL4SR-taiyuan-[10, 10]_21-01-08.pt'

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
        t_test = evaluate(model, gcn_model, dataset, args)
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
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)  # 根据需要调整 patience 和 min_delta


    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break # just to decrease identition
        # for step in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):

            adam_optimizer.zero_grad()

            u, seq, pos, neg, geo, geo_pos, dis, dis_pos = sampler.next_batch() # tuples to ndarray

            pos_ind = np.array(pos)
            
            u = torch.LongTensor(u).to(args.device)
            seq = torch.LongTensor(seq).to(args.device)
            pos = torch.LongTensor(pos).to(args.device)
            neg = torch.LongTensor(neg).to(args.device)
            geo = torch.LongTensor(geo).to(args.device)
            geo_pos = torch.LongTensor(geo_pos).to(args.device)
            dis = torch.LongTensor(dis).to(args.device)
            dis_pos = torch.LongTensor(dis_pos).to(args.device)

            if args.fus == 'kd' or args.fus == 'None':
                # student Model
                pos_logits, neg_logits = model(u, seq, pos, neg, geo, geo_pos, dis, dis_pos)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
                indices = np.where(pos_ind != 0)
                rec_loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                rec_loss += bce_criterion(neg_logits[indices], neg_labels[indices])

                # Teacher Model
                if args.kd == 'all':
                    if args.loss_type == 'bce':
                        with torch.no_grad():   
                            teacher_pos_logits, teacher_neg_logits = gcn_model(u, seq, pos, neg, args) 
                        soft_loss = kd_criterion(F.log_softmax(pos_logits/temp, dim=-1), F.softmax(teacher_pos_logits/temp, dim=1)) + kd_criterion(F.log_softmax(neg_logits/temp, dim=-1), F.softmax(teacher_neg_logits/temp, dim=1))
                        # soft_loss = kd_criterion(F.log_softmax(teacher_pos_logits/temp, dim=-1), F.softmax(pos_logits/temp, dim=1)) + kd_criterion(F.log_softmax(teacher_neg_logits/temp, dim=-1), F.softmax(neg_logits/temp, dim=1))
                        # soft_loss = kd_criterion(F.log_softmax(pos_logits[indices]/temp, dim=-1), F.softmax(teacher_pos_logits[indices]/temp, dim=-1)) + kd_criterion(F.log_softmax(neg_logits[indices]/temp, dim=-1), F.softmax(teacher_neg_logits[indices]/temp, dim=-1))
                    else:
                        with torch.no_grad():   
                            teacher_logits = gcn_model(u, seq, pos, neg, args)  
                            teacher_pos_logits = torch.gather(teacher_logits, dim=2, index=pos.unsqueeze(-1)).squeeze(-1)
                            teacher_neg_logits = torch.gather(teacher_logits, dim=2, index=neg.unsqueeze(-1)).squeeze(-1)    
                        soft_loss = kd_criterion(F.log_softmax(pos_logits/temp, dim=-1), F.softmax(teacher_pos_logits/temp, dim=-1)) + kd_criterion(F.log_softmax(neg_logits/temp, dim=-1), F.softmax(teacher_neg_logits/temp, dim=-1))
                
                if args.kd == 'pos':
                    if args.loss_type == 'bce':
                        with torch.no_grad():   
                            teacher_pos_logits, teacher_neg_logits = gcn_model(u, seq, pos, neg, args) 
                        soft_loss = kd_criterion(F.log_softmax(pos_logits/temp, dim=-1), F.softmax(teacher_pos_logits/temp, dim=-1))
                    else:
                        with torch.no_grad():   
                            teacher_logits = gcn_model(u, seq, pos, neg, args)  
                            teacher_pos_logits = torch.gather(teacher_logits, dim=2, index=pos.unsqueeze(-1)).squeeze(-1) 
                        soft_loss = kd_criterion(F.log_softmax(pos_logits/temp, dim=-1), F.softmax(teacher_pos_logits/temp, dim=-1))

                if args.kd != None:
                    loss = lamada * soft_loss + (1 - lamada) * rec_loss
                    # loss = soft_loss + rec_loss
                else:
                    loss = rec_loss

            elif args.fus == 'add':
                # Student Model
                log_feats, pos_embs, neg_embs = model(u, seq, pos, neg, geo, geo_pos, dis, dis_pos)
                # Teacher Model
                with torch.no_grad():   
                    seq_out = gcn_model(u, seq, pos, neg, args) 
                log_feats = log_feats + seq_out
                
                pos_logits = (log_feats * pos_embs).sum(dim=-1)
                neg_logits = (log_feats * neg_embs).sum(dim=-1)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
                indices = np.where(pos_ind != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            elif args.fus == 'cat':
                log_feats, pos_embs, neg_embs = model(u, seq, pos, neg, geo, geo_pos, dis, dis_pos)
                with torch.no_grad():   
                    seq_out = gcn_model(u, seq, pos, neg, args) 
                log_feats = torch.cat((log_feats, seq_out), dim = -1)
                log_feats = model.fus_linear(log_feats)
                
                pos_logits = (log_feats * pos_embs).sum(dim=-1)
                neg_logits = (log_feats * neg_embs).sum(dim=-1)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
                indices = np.where(pos_ind != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            elif args.fus == 'plus':
                log_feats, pos_embs, neg_embs = model(u, seq, pos, neg, geo, geo_pos, dis, dis_pos)
                with torch.no_grad():   
                    seq_out = gcn_model(u, seq, pos, neg, args) 
                out = log_feats * seq_out
                
                pos_logits = (out * pos_embs).sum(dim=-1)
                neg_logits = (out * neg_embs).sum(dim=-1)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
                indices = np.where(pos_ind != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])

            loss.backward()
            adam_optimizer.step()

            if args.test == True:
                break

        print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs
    
        if epoch % 1 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')

            NDCG_valid, HR_valid = evaluate_valid(model, gcn_model, dataset, args)
            
            print('epoch:%d, time: %f(s),\n valid (NDCG@1: %.4f, Recall@1: %.4f; NDCG@5: %.4f, Recall@5: %.4f; NDCG@10: %.4f, Recall@10: %.4f; NDCG@20: %.4f, Recall@20: %.4f; NDCG@50: %.4f, Recall@50: %.4f))' % (epoch, T, NDCG_valid[0], HR_valid[0], NDCG_valid[1], HR_valid[1], NDCG_valid[2], HR_valid[2], NDCG_valid[3], HR_valid[3], NDCG_valid[4], HR_valid[4]))
            
            f.write(str(NDCG_valid) + ' ' + str(HR_valid) + '\n')
            f.flush()

            # 检查早停机制并保存最好的模型状态
            early_stopping(NDCG_valid[2], model)  # 使用 NDCG@10 作为早停指标
            if early_stopping.early_stop:
                print("Early stopping")
                break

            t0 = time.time()
            model.train()

    f.close()
    sampler.close()
    print("Done")

    # 在训练结束后保存最好的模型
    folder = args.save_path + '_' + args.train_dir
    current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fname = '{}_date={}.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
    fname = fname.format(args.dataset, current_date, args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
    save_path = os.path.join(folder, fname)
    torch.save(early_stopping.best_model_weights, save_path)
    print(f"Best model saved to {save_path}")

    # 加载最佳模型权重
    model.load_state_dict(torch.load(save_path, map_location=args.device))
    print(f"Model loaded from {save_path}")

    # 设置模型为评估模式
    model.eval()
    print('Final test', end='')
    NDCG_test, HR_test = evaluate(model, gcn_model, dataset, args)
    print('test (NDCG@1: %.4f, Recall@1: %.4f; NDCG@5: %.4f, Recall@5: %.4f; NDCG@10: %.4f, Recall@10: %.4f; NDCG@20: %.4f, Recall@20: %.4f; NDCG@50: %.4f, Recall@50: %.4f)'
            % (NDCG_test[0], HR_test[0], NDCG_test[1], HR_test[1], NDCG_test[2], HR_test[2], NDCG_test[3], HR_test[3], NDCG_test[4], HR_test[4]))
