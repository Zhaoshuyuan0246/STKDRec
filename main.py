import os
import sys
import time
import random
import argparse
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

# Import local modules
from model import STTransformer
from model_gnn import STKGEncoder
from utils import data_partition, WarpSampler, evaluate, evaluate_valid
from EarlyStopping import EarlyStopping

# --- Helper Functions ---

def set_seed(seed):
    """Set random seeds to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Data & Path Settings ---
    g_data = parser.add_argument_group('Data & Path Settings')
    g_data.add_argument('--city', default='wuhan', type=str, help='The city of the dataset')
    g_data.add_argument('--dataset_dir', default='./data/', type=str, help='Root directory of the dataset')
    g_data.add_argument('--checkpoint_dir', default='./checkpoint/model/', type=str, help='Directory to save model checkpoints')
    g_data.add_argument('--pretrain_dir', default='./checkpoint/pre_train_model', type=str, help='Directory of the pre-trained teacher model')
    g_data.add_argument('--train_dir', default='default', type=str, help='Specific subdirectory name for this training run')
    g_data.add_argument('--log_file', default='log.txt', type=str, help='Log file name')

    # --- Student Model (STTransformer) ---
    g_student = parser.add_argument_group('Student Model (STTransformer)')
    g_student.add_argument('--state_dict_path', default=None, type=str, help='Path to student model weights file (for resuming training)')
    g_student.add_argument('--maxlen', default=128, type=int, help='Maximum sequence length')
    g_student.add_argument('--hidden_units', default=256, type=int, help='Dimension of hidden units in the model')
    g_student.add_argument('--num_blocks', default=2, type=int, help='Number of Transformer Blocks')
    g_student.add_argument('--num_heads', default=16, type=int, help='Number of heads for multi-head attention')
    g_student.add_argument('--dropout_rate', default=0.5, type=float, help='Dropout rate')

    # --- Teacher Model (GNN) ---
    g_teacher = parser.add_argument_group('Teacher Model (STKGEncoder)')
    g_teacher.add_argument('--gnn_hidden_units', default=256, type=int, help='Dimension of hidden units in GNN')
    g_teacher.add_argument('--sample_size', default=[5, 5], type=int, nargs='+', help='Number of neighbors to sample for GNN')
    g_teacher.add_argument("--sample_type", default='sparse', choices=['dense', 'sparse'], type=str)

    # --- Knowledge Distillation ---
    g_kd = parser.add_argument_group('Knowledge Distillation Settings')
    g_kd.add_argument('--lamada', default=0.01, type=float, help='Balance factor between distillation loss and recommendation loss')
    g_kd.add_argument('--temperature', default=7.0, type=float, help='Temperature coefficient for knowledge distillation')
    g_kd.add_argument('--geo_hash', default=False, choices=[True, False], type=bool) 
    g_kd.add_argument('--distances', default=False, choices=[True, False], type=bool) 
    g_kd.add_argument('--sptia', default=True, choices=[True, False], type=bool) 

    # --- Training & Optimization ---
    g_train = parser.add_argument_group('Training & Optimization')
    g_train.add_argument('--num_epochs', default=200, type=int, help='Number of training epochs')
    g_train.add_argument('--batch_size', default=50, type=int, help='Batch size')
    g_train.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    g_train.add_argument('--l2_emb', default=0.01, type=float, help='L2 regularization coefficient for embeddings')
    g_train.add_argument('--device', default='cuda:0', type=str, help='Training device (e.g., "cpu", "cuda:0")')
    g_train.add_argument('--seed', default=2024, type=int, help='Random seed')
    g_train.add_argument('--eval_freq', default=1, type=int, help='Evaluate every N epochs')
    g_train.add_argument("--lr_dc", type=float, default=0.7, help='learning rate decay.')
    g_train.add_argument("--lr_dc_step", type=int, default=5,
                            help='the number of steps after which the learning rate decay.')
    g_train.add_argument("--weight_decay", type=float, default=5e-5, help="weight_decay of adam")
    g_train.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    g_train.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    # --- Execution Control ---
    g_exec = parser.add_argument_group('Execution Control')
    g_exec.add_argument('--inference_only', action='store_true', help='Only run inference without training')

    args = parser.parse_args()
    return args

def initialize_weights(model):
    """Initialize model weights using Xavier Normal initialization."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            try:
                torch.nn.init.xavier_normal_(param.data)
            except ValueError:
                # Ignore layers that cannot be initialized (e.g., Embedding)
                pass

def log_and_print_metrics(epoch, T, metrics_ndcg, metrics_hr, header, log_file=None):
    """Format, print, and log evaluation metrics."""
    metrics_str = (
        f"epoch:{epoch if isinstance(epoch, int) else 'N/A':>3s}, time: {T:.1f}s, {header} ("
        f"NDCG@10: {metrics_ndcg[2]:.4f}, HR@10: {metrics_hr[2]:.4f} | "
        f"NDCG@1: {metrics_ndcg[0]:.4f}, HR@1: {metrics_hr[0]:.4f}; "
        f"NDCG@5: {metrics_ndcg[1]:.4f}, HR@5: {metrics_hr[1]:.4f}; "
        f"NDCG@20: {metrics_ndcg[3]:.4f}, HR@20: {metrics_hr[3]:.4f}; "
        f"NDCG@50: {metrics_ndcg[4]:.4f}, HR@50: {metrics_hr[4]:.4f})"
    )
    print(metrics_str)
    if log_file:
        log_file.write(f"{metrics_ndcg} {metrics_hr}\n")
        log_file.flush()

# --- Main Function ---

def main(args):
    # 1. Setup Environment and Paths
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    
    save_dir = os.path.join(args.checkpoint_dir, f"{args.city}_{args.train_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([f"{k},{v}" for k, v in sorted(vars(args).items())]))

    log_path = os.path.join(save_dir, args.log_file)
    log_file = open(log_path, 'w')

    # 2. Load Data
    args.gnn_dataset_path = os.path.join(args.dataset_dir, args.city)
    dataset = data_partition(args)
    [u, user_train, _, _, geo_train, _, _, dis_train, _, _, usernum, itemnum, geonum, disnum] = dataset
    args.item_size = itemnum
    args.user_size = usernum

    print(f"Dataset for '{args.city}' loaded. Users: {usernum}, Items: {itemnum}")

    # 3. Load Teacher Model (GNN)
    graph_path = os.path.join(args.gnn_dataset_path, f"{args.city}_witg.pt")
    global_graph = torch.load(graph_path, map_location=device)
    teacher_model = STKGEncoder(args=args, global_graph=global_graph)

    teacher_model_path = os.path.join(args.pretrain_dir, args.city, f"STKGEncoder-{args.city}.pt")
    teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
    
    # Freeze teacher model parameters
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.to(device)
    teacher_model.eval()
    print(f"Teacher model loaded from: {teacher_model_path}")

    # 4. Create Student Model (Transformer)
    student_model = STTransformer(usernum, itemnum, geonum, disnum, args).to(device)
    initialize_weights(student_model)
    total_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f'Student model created. Total trainable params: {total_params / 1e6:.2f}M')

    epoch_start_idx = 1
    if args.state_dict_path:
        try:
            student_model.load_state_dict(torch.load(args.state_dict_path, map_location=device), strict=False)
            print(f"Student model state loaded from: {args.state_dict_path}")
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except Exception as e:
            print(f"Failed to load state_dict: {e}. Training from scratch.")

    # 5. Handle Inference-Only Mode
    if args.inference_only:
        print("Running in inference-only mode...")
        student_model.eval()
        ndcg_test, hr_test = evaluate(student_model, teacher_model, dataset, args)
        log_and_print_metrics('N/A', 0, ndcg_test, hr_test, header='Final Test')
        log_file.close()
        return

    # 6. Prepare for Training
    sampler = WarpSampler(u, user_train, geo_train, dis_train, usernum, itemnum, geonum, disnum, 
                          batch_size=args.batch_size, maxlen=args.maxlen, n_workers=1, seed=args.seed)
    
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    kd_criterion = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    num_batch = len(user_train) // args.batch_size
    T_total = 0.0
    t0 = time.time()

    # 7. Training Loop
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        student_model.train()
        epoch_loss = 0.0
        
        for _ in tqdm(range(num_batch), total=num_batch, ncols=80, desc=f"Epoch {epoch}/{args.num_epochs}", leave=False):
            optimizer.zero_grad()

            # Sample and convert data
            batch_data = sampler.next_batch()
            # u, seq, pos, neg, geo, geo_pos, dis, dis_pos
            tensors = [torch.LongTensor(arr).to(device) for arr in batch_data]
            u_t, seq_t, pos_t, neg_t, geo_t, geo_pos_t, dis_t, dis_pos_t = tensors
            
            # Forward pass for the student model
            pos_logits, neg_logits = student_model(u_t, seq_t, pos_t, neg_t, geo_t, geo_pos_t, dis_t, dis_pos_t)
            
            # Calculate recommendation loss
            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)
            indices = torch.where(pos_t != 0)
            
            rec_loss = bce_criterion(pos_logits[indices], pos_labels[indices]) + \
                       bce_criterion(neg_logits[indices], neg_labels[indices])

            # Forward pass for the teacher model (to get soft labels)
            with torch.no_grad():
                teacher_pos_logits, teacher_neg_logits = teacher_model(u_t, seq_t, pos_t, neg_t, args)
            
            # Calculate knowledge distillation loss
            soft_loss = kd_criterion(
                F.log_softmax(pos_logits / args.temperature, dim=-1),
                F.softmax(teacher_pos_logits / args.temperature, dim=-1)
            ) + kd_criterion(
                F.log_softmax(neg_logits / args.temperature, dim=-1),
                F.softmax(teacher_neg_logits / args.temperature, dim=-1)
            )

            # Combine losses
            loss = (1 - args.lamada) * rec_loss + args.lamada * soft_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch} avg loss: {epoch_loss / num_batch:.4f}")

        # 8. Evaluation and Early Stopping
        if epoch % args.eval_freq == 0:
            student_model.eval()
            t1 = time.time() - t0
            T_total += t1
            
            ndcg_valid, hr_valid = evaluate_valid(student_model, teacher_model, dataset, args)
            log_and_print_metrics(epoch, T_total, ndcg_valid, hr_valid, 'Validation', log_file)
            
            # Use NDCG@10 as the early stopping metric
            early_stopping(ndcg_valid[2], student_model) 
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
            
            t0 = time.time()

    # 9. Post-Training Procedures
    log_file.close()
    sampler.close()
    print("Training finished.")

    # Save the best model
    current_date = datetime.now().strftime('%Y-%m-%d')
    fname = f"city={args.city}_date={current_date}_epoch={early_stopping.best_epoch}.pth"
    best_model_path = os.path.join(save_dir, fname)
    torch.save(early_stopping.best_model_weights, best_model_path)
    print(f"Best model saved to {best_model_path}")

    # 10. Final Test with the Best Model
    print("Loading best model for final testing...")
    student_model.load_state_dict(torch.load(best_model_path, map_location=device))
    student_model.eval()
    
    ndcg_test, hr_test = evaluate(student_model, teacher_model, dataset, args)
    log_and_print_metrics(early_stopping.best_epoch, T_total, ndcg_test, hr_test, header='Final Test')


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    main(args)
