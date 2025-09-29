# @Time    : 13/1/25
# @Author  : Minghui Zhao
# @Affiliation  : Southeast university

import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np
from data_io_attention import ReadList,read_conf,str_to_bool
import tqdm
import random
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from torch_geometric.loader import DataLoader as DataLoader_gnn
import gnn_model
from torch_geometric.data import Batch

# PyTorch 2.6 默认 torch.load(weights_only=True) 会阻止直接反序列化自定义/第三方对象(Data, DataEdgeAttr)
# 我们保存的 .pt 是包含 torch_geometric.data.Data 对象的 list，需要关闭 weights_only。
def load_graph_list(path: str):
    import torch.serialization as _ser
    try:
        # 直接尝试强制 weights_only=False
        return torch.load(path, weights_only=False)
    except TypeError:
        # 旧版本 torch.load 没有 weights_only 形参
        return torch.load(path)
    except Exception:
        # 尝试 allowlist 相关类再加载
        try:
            from torch_geometric.data.data import DataEdgeAttr
            from torch_geometric.data import Data
            _ser.add_safe_globals([DataEdgeAttr, Data])
        except Exception:
            pass
        return torch.load(path, weights_only=False)

# Safe load for our own saved model checkpoints (state_dict + simple metrics)
def load_ckpt(path: str, map_location=None):
    import torch.serialization as _ser
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # older torch without weights_only
        return torch.load(path, map_location=map_location)
    except Exception:
        try:
            import numpy as _np  # ensure module imported
            from numpy.core.multiarray import scalar as _np_scalar  # type: ignore
            _ser.add_safe_globals([_np_scalar])
        except Exception:
            pass
        return torch.load(path, map_location=map_location, weights_only=False)

# Reading cfg file
options = read_conf('cfg/gnn_5fold_100person.cfg')

#[data]
options.name = options.name
options.tr_lst = options.tr_lst
options.te_lst = options.te_lst
options.pt_file = options.pt_file
options.class_dict_file = options.lab_dict
options.data_folder = options.data_folder+'/'
options.output_folder = options.output_folder

#[windowing]
options.fs = int(options.fs)
options.cw_len = int(options.cw_len)
options.cw_shift = int(options.cw_shift)

#[cnn]
options.cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
options.cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
options.cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
options.cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
options.cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
options.cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
options.cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
options.cnn_act = list(map(str, options.cnn_act.split(',')))
options.cnn_drop = list(map(float, options.cnn_drop.split(',')))
options.mulhead_num_hiddens = int(options.mulhead_num_hiddens)
options.mulhead_num_heads = int(options.mulhead_num_heads)
options.mulhead_num_query = int(options.mulhead_num_query)
options.dropout_fc = float(options.dropout_fc)
options.hidden_dims_fc = int(options.hidden_dims_fc)
options.num_classes = int(options.num_classes)

#[gnn]
options.num_node_features = int(options.num_node_features)
options.hidden_channels = int(options.hidden_channels)
options.num_pause_input = int(options.num_pause_input)
options.num_att_features = int(options.num_att_features)
options.num_emotion_features = int(options.num_emotion_features)
options.num_enegy_features = int(options.num_enegy_features)
options.num_tromer_features = int(options.num_tromer_features)

#[optimization]
options.lr = float(options.lr)
options.batch_size = int(options.batch_size)
options.N_epochs = int(options.N_epochs)
options.N_batches = int(options.N_batches)
options.N_eval_epoch = int(options.N_eval_epoch)
options.seed = int(options.seed)
options.fold = int(options.fold)
options.patience = int(options.patience)

# ================= Logging control (to avoid confusing duplicated outputs) =================
# 如果每次运行都希望重新生成日志而不是在旧文件后面追加，将 CLEAR_OLD_LOG 置为 True
CLEAR_OLD_LOG = True
LOG_FILE_NAME = "res.res"
if CLEAR_OLD_LOG:
    try:
        os.makedirs(options.output_folder, exist_ok=True)
        log_path_try = os.path.join(options.output_folder, LOG_FILE_NAME)
        with open(log_path_try, 'w', encoding='utf-8') as _f:
            _f.write('# epoch,fold,loss_tr,err_tr,loss_val,err_val,acc,precision_macro,recall_macro,f1_macro,'
                     'sensitivity,specificity,auc,best_f1_selector\n')
        print(f'[INFO] 已清空旧日志并写入表头: {log_path_try}')
    except Exception as e:
        print(f'[WARN] 日志初始化失败: {e}')

def diff_list_gen(path_all):
    audio_list = ReadList(path_all)
    person_old = 'xx'
    train_f = []
    val_f = []
    for audio in audio_list:
        person = audio.split('_')[0]
        if person == person_old:
            save_path.append(audio)
        else:
            num = random.random()
            if num <= 0.7:
                save_path = train_f
            else:
                save_path = val_f
            person_old = person
            save_path.append(audio)
    return train_f, val_f


def create_fully_connected_edge_index_single(num_nodes, include_self_loops=False):
    src_nodes = []
    tgt_nodes = []
    for src in range(num_nodes):
        for tgt in range(num_nodes):
            if src != tgt or include_self_loops:
                src_nodes.append(src)
                tgt_nodes.append(tgt)

    edge_index = torch.tensor([src_nodes, tgt_nodes], dtype=torch.long)
    return edge_index

def create_fully_connected_edge_index_batch(batch_size, num_nodes, include_self_loops=False):
    all_edge_indices = []
    for graph_idx in range(batch_size):
        node_offset = graph_idx * num_nodes
        single_edge_index = create_fully_connected_edge_index_single(num_nodes, include_self_loops)
        single_edge_index += node_offset
        all_edge_indices.append(single_edge_index)

    edge_index = torch.cat(all_edge_indices, dim=1)
    return edge_index

# 统一将模态表示转换为图级 (batch_size, feat)
def unify_to_graph_level(t: torch.Tensor, bs: int, num_nodes_local: int):
    try:
        if t.dim() == 2 and t.shape[0] == bs * num_nodes_local:  # (bs*num_nodes, feat)
            return t.view(bs, num_nodes_local, -1).mean(1)
        if t.dim() == 3 and t.shape[0] == bs:  # (bs, num_nodes, feat)
            return t.mean(1)
        if t.dim() == 2 and t.shape[0] == bs:  # already (bs, feat)
            return t
        if t.dim() == 2 and bs > 0 and t.shape[0] % bs == 0:  # fallback grouping
            k = t.shape[0] // bs
            return t.view(bs, k, -1).mean(1)
        return t
    except Exception:
        return t


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fold = options.fold

acc_list = []
f1_list = []
auc_list = []
pre_list = []
rec_list = []
sen_list = []
spe_list = []

epoch_get_acc = []
epoch_get_f1 = []
epoch_get_auc = []
epoch_get_pre = []
epoch_get_rec = []
epoch_get_sen = []
epoch_get_spe = []

for i in range(fold):
    acc_list.append(1)
    f1_list.append(1)
    auc_list.append(1)
    pre_list.append(1)
    rec_list.append(1)
    sen_list.append(1)
    spe_list.append(1)
    epoch_get_acc.append([])
    epoch_get_f1.append([])
    epoch_get_auc.append([])
    epoch_get_pre.append([])
    epoch_get_rec.append([])
    epoch_get_sen.append([])
    epoch_get_spe.append([])


def deleteDuplicatedElementFromList2(list):
    resultList = []
    for item in list:
        if not item in resultList:
            resultList.append(item)
    return resultList

for fold_i in range(fold):
    te_lst_fold = options.te_lst + f'{fold_i}.scp' if options.te_lst else None
    dev_lst_fold = None
    if hasattr(options, 'dev_lst') and options.dev_lst:
        dev_lst_fold = options.dev_lst + f'{fold_i}.scp'
    options.wlen = int(options.fs * options.cw_len / 1000.00)
    options.wshift = int(options.fs * options.cw_shift / 1000.00)

    # Build person list reference (for per-person metrics structures)
    person_name = []
    if te_lst_fold and os.path.isfile(te_lst_fold):
        wav_lst_te = ReadList(te_lst_fold)
        snt_te = len(wav_lst_te)
        print(f'test_len(list):{snt_te}')
        for audio in wav_lst_te:
            person_name.append(audio.split('_')[0])
    else:
        # fallback: derive from validation pt graph names later
        print(f'[INFO] 未找到测试列表 {te_lst_fold}, 将从验证图数据推断 person 列表')

    train_list = load_graph_list(f'data_train_{fold_i}.pt')
    train_loader_gnn = DataLoader_gnn(train_list, batch_size=options.batch_size)

    # Validation dataset (val pt)
    val_graphs = load_graph_list(f'data_val_{fold_i}.pt')
    # if person list empty, derive
    if not person_name:
        tmp_person = []
        for g in val_graphs:
            nm = g.name_my[0][0]
            tmp_person.append(nm.split('_')[0])
        person_name = deleteDuplicatedElementFromList2(tmp_person)
        print(f'[INFO] 从验证集推断 person 数量: {len(person_name)}')
    test_loader_gnn = DataLoader_gnn(val_graphs, batch_size=options.batch_size)

    # Define Early Stop variables
    patience = options.patience  # Number of epochs to wait before stopping
    best_acc = 0  # The best validation loss so far
    counter = 0  # Number of epochs since the best validation loss improved
    # Folder creation
    try:
        os.stat(options.output_folder)
    except:
        os.mkdir(options.output_folder)

    # setting seed
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)

    # loss function
    cost = nn.CrossEntropyLoss()
    l2_loss = nn.MSELoss()
    cos_loss = nn.CosineEmbeddingLoss(reduction='mean')

    GNN_model = gnn_model.GAT_my_loss_decoupled(my_model_option=options)
    print(GNN_model)
    GNN_model.cuda()
    optimizer_GNN = optim.RMSprop(GNN_model.parameters(), lr=options.lr,alpha=0.95, eps = 1e-8)
    err_tot_dev_snt_min = 1
    best_ckpt_path = None
    for epoch in range(options.N_epochs):
        test_flag = 0
        GNN_model.train()
        loss_sum = 0
        err_sum = 0
        train_bar = tqdm.tqdm(train_loader_gnn)
        N_batches = len(train_loader_gnn)
        for data in train_bar:
            data.to(device)
            lab = data.y
            # print(data.edge_index)

            batch_size = data.x.shape[0]//5
            num_nodes = 5
            edge_index_diff = create_fully_connected_edge_index_batch(batch_size, num_nodes, include_self_loops=False)
            edge_index_diff = edge_index_diff.to(device)


            pout,pout_attsinc,pout_emotion,pout_pause,pout_enegy,pout_js,re_tensor,same_diff, same_tensor,diff_tensor = GNN_model(data.x, data.edge_index, edge_index_diff, data.batch)
            pred = torch.max(pout,dim = 1)[1]

            # ---------------- Cross-Entropy (main + auxiliary modality-specific heads) ----------------
            loss = cost(pout, lab.long()) + 0.7*(
                cost(pout_attsinc, lab.long()) +
                cost(pout_emotion, lab.long()) +
                cost(pout_pause, lab.long()) +
                cost(pout_enegy, lab.long()) +
                cost(pout_js, lab.long())
            )

            x_all_re_input,x_all_re_out = re_tensor
            x_all_same,x_all_diff= same_diff
            x_attsinc_same, x_emotion_same, x_pause_same, x_enegy_same, x_js_same = same_tensor
            x_attsinc_diff, x_emotion_diff, x_pause_diff, x_enegy_diff, x_js_diff = diff_tensor

            # ---------------- Reconstruction loss ----------------
            loss_re = l2_loss(x_all_re_input, x_all_re_out)

            # ---------------- Contrastive losses (cleaned) ----------------
            # 1) Between shared (same) and diff representations (negative target = -1)
            cos_flag_neg = (-torch.ones([x_all_same.shape[0]], device=device))
            loss_same_diff = cos_loss(x_all_same, x_all_diff, cos_flag_neg)

            # 2) Positive pairwise cosine alignment across all modalities' shared embeddings
            # 统一所有模态 same 表示为图级 shape=(batch_size, feat)
            modal_raw = [x_attsinc_same, x_emotion_same, x_pause_same, x_enegy_same, x_js_same]
            modal_graph = [unify_to_graph_level(t, batch_size, num_nodes) for t in modal_raw]

            cos_flag_pos_template = torch.ones(batch_size, device=device)
            pos_pairs_loss = 0.0
            pair_count = 0
            warn_once_key = '_pairwise_shape_warned_v2'
            for mi in range(len(modal_graph)):
                for mj in range(mi+1, len(modal_graph)):
                    a = modal_graph[mi]
                    b = modal_graph[mj]
                    if a.dim() != 2 or b.dim() != 2 or a.shape[0] != batch_size or b.shape[0] != batch_size:
                        if not hasattr(torch, warn_once_key):
                            print(f'[WARN] 模态嵌入形状异常或 batch 不匹配, 已跳过 (a={a.shape}, b={b.shape}, bs={batch_size}).')
                            setattr(torch, warn_once_key, True)
                        continue
                    pos_pairs_loss += cos_loss(a, b, cos_flag_pos_template)
                    pair_count += 1
            if pair_count > 0:
                pos_pairs_loss = pos_pairs_loss / pair_count
            else:
                pos_pairs_loss = torch.tensor(0.0, device=device)

            # Total contrastive component (previously: duplicated loss_same_diff + 9~10 positive terms)
            # Weighting constants kept (0.4, 0.6) but structure simplified.
            contrastive_loss = loss_same_diff + pos_pairs_loss

            loss = loss + 0.4 * loss_re + 0.6 * contrastive_loss

            err = torch.mean((pred != lab.long()).float())
            optimizer_GNN.zero_grad()

            loss.backward()
            optimizer_GNN.step()

            loss_sum = loss_sum+loss.detach()
            err_sum = err_sum+err.detach()

        loss_tot = loss_sum/N_batches
        err_tot = err_sum/N_batches

        # scheduler.step()

        # Full Validation  new
        if epoch % options.N_eval_epoch == 0:

            GNN_model.eval()

            test_flag = 1
            loss_sum = 0
            err_sum = 0


            matrix_label = np.array([])
            matrix_pred = np.array([])

            roc_label = np.array([])
            roc_pred = np.array([])

            person_matrix = []
            person_roc = []
            person_lable = []
            for i, _ in enumerate(person_name):
                person_roc.append([])
                person_matrix.append([])
                person_lable.append([])

            with torch.no_grad():
                test_bar = tqdm.tqdm(test_loader_gnn)
                test_batches = len(test_loader_gnn)
                for data in test_bar:
                    data.to(device)
                    lab = data.y

                    batch_size = data.x.shape[0] // 5
                    num_nodes = 5
                    edge_index_diff = create_fully_connected_edge_index_batch(batch_size, num_nodes,
                                                                              include_self_loops=False)
                    edge_index_diff = edge_index_diff.to(device)


                    # Unified extraction of sample names (data.name_my often nested lists like [["xxx_1"]])
                    def extract_name(n):
                        # peel nested list/tuple layers
                        while isinstance(n, (list, tuple)) and len(n) > 0:
                            n = n[0]
                        if isinstance(n, str):
                            return n
                        # fallback: try decode tensor of shape () to string (not expected here)
                        try:
                            return str(n)
                        except Exception:
                            return ''

                    names_batch = data.name_my  # keep original list

                    pout,pout_attsinc,pout_emotion,pout_pause,pout_enegy,pout_js,_,_,_,_ = GNN_model(data.x, data.edge_index, edge_index_diff, data.batch)

                    pred = torch.max(pout,dim = 1)[1]
                    pred_roc = torch.softmax(pout, dim=1)

                    loss = cost(pout, lab.long())
                    err = torch.mean((pred!=lab.long()).float())

                    loss_sum = loss_sum+loss.detach()
                    err_sum = err_sum+err.detach()


                    matrix_label = np.append(matrix_label, lab.cpu().detach().numpy())
                    matrix_pred = np.append(matrix_pred, pred.cpu().detach().numpy())

                    # pred_roc = torch.mean(pred_roc, dim=0)
                    roc_label = np.append(roc_label, lab.cpu().detach().numpy())
                    roc_pred = np.append(roc_pred, pred_roc.cpu().detach().numpy())
                    for i_num, name_get in enumerate(names_batch):
                        base_name = extract_name(name_get)
                        person_id = base_name.split('_')[0] if '_' in base_name else base_name
                        if person_id in person_name:
                            idx = person_name.index(person_id)
                            person_roc[idx].append(pred_roc.cpu().detach().numpy()[i_num])
                            person_matrix[idx].append(pred.cpu().detach().numpy()[i_num])
                            person_lable[idx].append(lab.cpu().detach().numpy()[i_num])

                loss_tot_dev = loss_sum/test_batches
                err_tot_dev = err_sum/test_batches

                # #################
                # person_roc_means = []
                # for sublist in person_roc:
                #
                #     total = sum(sublist)
                #
                #     count = len(sublist)
                #
                #     sublist_mean = total / count
                #     person_roc_means.append(sublist_mean)
                #
                #
                # person_matrix_means = []
                # for sublist in person_matrix:
                #
                #     total = sum(sublist)
                #
                #     count = len(sublist)
                #
                #     sublist_mean = total / count
                #     if sublist_mean > 0.5:
                #         person_matrix_means.append(1)
                #     else:
                #         person_matrix_means.append(0)
                #
                #
                # person_lable_means = []
                # for sublist in person_lable:
                #
                #     total = sum(sublist)
                #
                #     count = len(sublist)
                #
                #     sublist_mean = total / count
                #     person_lable_means.append(sublist_mean)
                #
                # matrix_label = np.array(person_lable_means)
                # roc_pred = np.array(person_roc_means)
                # matrix_pred = np.array(person_matrix_means)
                # #########################################

            conf_matrix = confusion_matrix(matrix_label, matrix_pred, labels=[1, 0])
            # conf_matrix rows: true 1, true 0; cols: pred 1, pred 0 (since we passed labels=[1,0])
            # TP = conf_matrix[0,0]; FN = conf_matrix[0,1]; FP = conf_matrix[1,0]; TN = conf_matrix[1,1]
            TP = conf_matrix[0, 0] if conf_matrix.size else 0
            FN = conf_matrix[0, 1] if conf_matrix.size > 1 else 0
            FP = conf_matrix[1, 0] if conf_matrix.shape[0] > 1 else 0
            TN = conf_matrix[1, 1] if conf_matrix.shape[0] > 1 and conf_matrix.shape[1] > 1 else 0

            accuracy = metrics.accuracy_score(matrix_label, matrix_pred)
            precision = metrics.precision_score(matrix_label, matrix_pred, average='macro', zero_division=0)
            recall = metrics.recall_score(matrix_label, matrix_pred, average='macro', zero_division=0)
            f1_score = metrics.f1_score(matrix_label, matrix_pred, average='macro', zero_division=0)
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # recall for positive class
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

            print(f"Accuracy: {accuracy}")
            print(f"Precision (macro): {precision}")
            print(f"Recall (macro): {recall}")
            print(f"F1-Score (macro): {f1_score}")
            print(f"Sensitivity (TPR): {sensitivity}")
            print(f"Specificity (TNR): {specificity}\n")

            try:
                a = roc_pred.reshape(-1, options.num_classes)[:, 1]
                b = matrix_label
                auc = metrics.roc_auc_score(b, a)
            except Exception as e:
                print(f'AUC none')
                auc = 0
            print(f'AUC{auc}')

            epoch_get_acc[fold_i].append(accuracy)
            epoch_get_f1[fold_i].append(f1_score)
            epoch_get_auc[fold_i].append(auc)
            epoch_get_pre[fold_i].append(precision)
            epoch_get_rec[fold_i].append(recall)
            epoch_get_sen[fold_i].append(sensitivity)
            epoch_get_spe[fold_i].append(specificity)

            print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f fold=%f best_f1=%f" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,fold_i,best_acc))
            print('=' * 89)
            with open(options.output_folder+"/res.res", "a") as res_file:
                res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f fold=%f best_f1=%f acc=%f pre=%f rec=%f sen=%f spe=%f auc=%f\n" % (epoch, loss_tot,err_tot,loss_tot_dev,err_tot_dev,fold_i,best_acc,accuracy,precision,recall,sensitivity,specificity,auc))

            # 仍使用宏平均 F1 作为模型选择标准
            if f1_score > best_acc:
                best_acc = f1_score
                acc_list[fold_i] = accuracy
                f1_list[fold_i] = f1_score
                auc_list[fold_i] = auc
                pre_list[fold_i] = precision
                rec_list[fold_i] = recall
                sen_list[fold_i] = sensitivity
                spe_list[fold_i] = specificity
                # 保存当前 fold 最优模型
                best_ckpt_path = os.path.join(options.output_folder, f'best_fusion_fold{fold_i}.pth')
                torch.save({'model_state': GNN_model.state_dict(),
                            'epoch': epoch,
                            'f1': f1_score,
                            'accuracy': accuracy,
                            'precision_macro': precision,
                            'recall_macro': recall,
                            'sensitivity': sensitivity,
                            'specificity': specificity,
                            'auc': auc}, best_ckpt_path)
                print(f'[Fold {fold_i}] 保存新的最佳模型 -> {best_ckpt_path} (F1={f1_score:.4f} Sens={sensitivity:.4f} Spec={specificity:.4f})')
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping after {} epochs".format(epoch + 1))
                    break

    # ====== 训练循环结束，若存在单独测试集则使用最佳模型评估 ======
    test_graph_file = f'data_test_{fold_i}.pt'
    if best_ckpt_path is not None and os.path.isfile(test_graph_file):
        try:
            test_graph_list = load_graph_list(test_graph_file)
            test_loader_final = DataLoader_gnn(test_graph_list, batch_size=options.batch_size)
            # 加载最佳模型权重
            ckpt = load_ckpt(best_ckpt_path, map_location='cpu')
            GNN_model.load_state_dict(ckpt['model_state'])
            GNN_model.eval()
            print(f'[Fold {fold_i}] 使用验证集最佳模型在独立测试集上评估: {test_graph_file}')
            with torch.no_grad():
                matrix_label = np.array([])
                matrix_pred = np.array([])
                roc_label = np.array([])
                roc_pred = np.array([])
                for data in test_loader_final:
                    data.to(device)
                    lab = data.y
                    batch_size_eval = data.x.shape[0] // 5
                    edge_index_diff = create_fully_connected_edge_index_batch(batch_size_eval, 5, include_self_loops=False).to(device)
                    pout,_,_,_,_,_,_,_,_,_ = GNN_model(data.x, data.edge_index, edge_index_diff, data.batch)
                    pred = torch.max(pout, dim=1)[1]
                    pred_soft = torch.softmax(pout, dim=1)
                    matrix_label = np.append(matrix_label, lab.cpu().numpy())
                    matrix_pred = np.append(matrix_pred, pred.cpu().numpy())
                    roc_label = np.append(roc_label, lab.cpu().numpy())
                    roc_pred = np.append(roc_pred, pred_soft.cpu().numpy())
            test_acc = metrics.accuracy_score(matrix_label, matrix_pred)
            test_f1 = metrics.f1_score(matrix_label, matrix_pred, average='macro')
            try:
                auc_test = metrics.roc_auc_score(roc_label, roc_pred.reshape(-1, options.num_classes)[:,1])
            except:
                auc_test = 0
            print(f'[Fold {fold_i}] TEST -> Acc={test_acc:.4f} F1={test_f1:.4f} AUC={auc_test:.4f}')
            with open(options.output_folder+"/res.res", "a") as res_file:
                res_file.write(f"[Fold {fold_i}] TEST Acc={test_acc:.4f} F1={test_f1:.4f} AUC={auc_test:.4f} (best val F1={best_acc:.4f})\n")
        except Exception as e:
            print(f'[Fold {fold_i}] 测试集评估失败: {e}')
    else:
        if best_ckpt_path is None:
            print(f'[Fold {fold_i}] 未保存最佳模型 (可能训练未触发评估)。')
        else:
            print(f'[Fold {fold_i}] 未找到测试集图文件 {test_graph_file}，跳过测试评估。')

with open(options.output_folder+"/res.res", "a") as res_file:
    # =============== Fold-Level Best Metric Summary ===============
    mean_acc = float(np.array(acc_list).mean())
    print('[SUMMARY] best_fold_acc_list=', acc_list, 'mean=', mean_acc)
    res_file.write(f"best_acc_each_fold={acc_list}---mean={mean_acc}\n")

    mean_f1 = float(np.array(f1_list).mean())
    print('[SUMMARY] best_fold_f1_list=', f1_list, 'mean=', mean_f1)
    res_file.write(f"best_f1_each_fold={f1_list}---mean={mean_f1}\n")

    mean_auc = float(np.array(auc_list).mean())
    print('[SUMMARY] best_fold_auc_list=', auc_list, 'mean=', mean_auc)
    res_file.write(f"best_auc_each_fold={auc_list}---mean={mean_auc}\n")

    mean_precision = float(np.array(pre_list).mean())
    print('[SUMMARY] best_fold_precision_list=', pre_list, 'mean=', mean_precision)
    # precision 不强制写入避免日志膨胀，如需可解除注释
    # res_file.write(f"best_precision_each_fold={pre_list}---mean={mean_precision}\n")

    mean_recall = float(np.array(rec_list).mean())
    print('[SUMMARY] best_fold_recall_list=', rec_list, 'mean=', mean_recall)
    # res_file.write(f"best_recall_each_fold={rec_list}---mean={mean_recall}\n")

    mean_sen = float(np.array(sen_list).mean())
    print('[SUMMARY] best_fold_sensitivity_list=', sen_list, 'mean=', mean_sen)
    res_file.write(f"best_sensitivity_each_fold={sen_list}---mean={mean_sen}\n")

    mean_spe = float(np.array(spe_list).mean())
    print('[SUMMARY] best_fold_specificity_list=', spe_list, 'mean=', mean_spe)
    res_file.write(f"best_specificity_each_fold={spe_list}---mean={mean_spe}\n")

    # =============== Epoch Curve Aggregation Across Folds ===============
    def aggregate_epoch_curves(curve_lists, tag_name):
        min_len = min(len(lst) for lst in curve_lists if len(lst) > 0) if any(len(lst) for lst in curve_lists) else 0
        if min_len == 0:
            print(f'[WARN] No curve data for {tag_name}')
            return []
        mean_curve = []
        for ep in range(min_len):
            mean_curve.append(float(sum(lst[ep] for lst in curve_lists) / fold))
        print(f'[EPOCH-CURVE] {tag_name}: len={len(mean_curve)} last={mean_curve[-1] if mean_curve else None}')
        res_file.write(f"epoch_curve_{tag_name}={mean_curve}\n")
        return mean_curve

    curve_acc = aggregate_epoch_curves(epoch_get_acc, 'acc')
    curve_f1 = aggregate_epoch_curves(epoch_get_f1, 'f1')
    curve_auc = aggregate_epoch_curves(epoch_get_auc, 'auc')
    curve_precision = aggregate_epoch_curves(epoch_get_pre, 'precision')
    curve_recall = aggregate_epoch_curves(epoch_get_rec, 'recall')
    curve_sen = aggregate_epoch_curves(epoch_get_sen, 'sensitivity')
    curve_spe = aggregate_epoch_curves(epoch_get_spe, 'specificity')

    # 末尾再给一个总览行方便 grep
    res_file.write('# SUMMARY_DONE\n')