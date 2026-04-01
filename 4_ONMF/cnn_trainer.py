"""
CNN 训练与评估模块
==================
Torre-Cruz et al. 2023 Section 5

训练流程：
  实现论文 Section 4.2 Training Protocol:
    - 10-fold 交叉验证 × 5次重复 = 50 个评估点（第5.5.1节）
    - 每 Fold: 75%训练+验证, 25%测试；训练中 25% 作验证
    - epochs=30, batch=16, lr=0.001, Adam
    - 早停 patience=10，监控验证集损失（val_loss）
    - 平均混淤矩阵汇总所有 fold 结果并输出各指标

评估指标（Section 4.3）：
  Acc  = (TP+TN) / (TP+TN+FP+FN)
  Se   = TP / (TP+FN)  → 异常心音检出率
  Sp   = TN / (TN+FP)  → 正常心音正确率
  Pre  = TP / (TP+FP)  → 精确率 (论文 Fig. 7(D))
  Sco  = (Se + Sp) / 2 → 主要排名指标 (论文 Fig. 7(E))
  F1   = 2*Pre*Se / (Pre+Se)
"""

import time
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# 评估指标计算
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算二分类指标：Acc, Se, Sp, Pre, Sco=(Se+Sp)/2, F1。
    对应论文 Section 4.3 的 5 个指标: Acc / Se / Sp / Pre / Sco。

    Parameters
    ----------
    y_true : 真实标签数组 (0=正常, 1=异常)
    y_pred : 预测标签数组 (0=正常, 1=异常)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    acc = (tp + tn) / max(len(y_true), 1)
    se  = tp / max(tp + fn, 1)          # Sensitivity
    sp  = tn / max(tn + fp, 1)          # Specificity
    pre = tp / max(tp + fp, 1)          # Precision
    sco = (se + sp) / 2                 # Score=(Se+Sp)/2, Fig.7(E)
    f1  = 2 * pre * se / max(pre + se, 1e-9)

    return {
        'accuracy':    round(acc, 4),
        'sensitivity': round(se,  4),
        'specificity': round(sp,  4),
        'precision':   round(pre, 4),
        'score':       round(sco, 4),
        'f1':          round(f1,  4),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


# ---------------------------------------------------------------------------
# 批次评估
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: Optional[nn.Module] = None) -> Tuple[Dict[str, float], float]:
    """
    在给定 DataLoader 上评估模型。

    Returns
    -------
    metrics : dict（accuracy, sensitivity, specificity, score）
    avg_loss: float（若 criterion 为 None 则返回 0.0）
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            total_loss += criterion(logits, y).item()
            n_batches += 1
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    avg_loss = total_loss / max(n_batches, 1)
    return metrics, avg_loss


# ---------------------------------------------------------------------------
# 早停器
# ---------------------------------------------------------------------------

class EarlyStopping:
    """监控指标（越小越好 或 越大越好），patience 轮无提升则停止。"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 mode: str = 'min'):
        """
        Parameters
        ----------
        patience  : 连续多少轮无提升后触发早停
        min_delta : 最小提升阈值
        mode      : 'min' = 监控越小越好 (val_loss)
                    'max' = 监控越大越好 (score/acc)
        """
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best      = float('inf') if mode == 'min' else -float('inf')
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Returns True 若应停止训练。"""
        if self.mode == 'min':
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta

        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
    model_name: str = 'model',
    class_weights: Optional[torch.Tensor] = None,
    scheduler_patience: int = 5,
    early_stop_patience: int = 15,
    verbose: bool = True,
    progress_callback=None,
) -> Dict:
    """
    训练 CNN 模型。

    Parameters
    ----------
    model              : 待训练的 nn.Module
    train_loader       : 训练集 DataLoader
    val_loader         : 验证集 DataLoader
    num_epochs         : 最大训练轮数
    lr                 : Adam 初始学习率
    weight_decay       : L2 正则化系数
    device             : torch.device（None 则自动选择 cuda/mps/cpu）
    save_dir           : 模型保存目录（None 则不保存）
    model_name         : 保存文件的前缀名
    class_weights      : 类别权重张量 [w0, w1]（处理类别不均衡）
    scheduler_patience : LR 衰减 patience 轮数
    early_stop_patience: 早停 patience 轮数
    verbose            : 是否打印训练日志
    progress_callback  : callable(epoch, total, metrics_dict) 用于 GUI 回调

    Returns
    -------
    history : dict，包含训练历史和最终测试结果
      'train_loss', 'val_loss', 'val_acc', 'val_score',
      'best_epoch', 'best_val_score', 'training_time_s'
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    model = model.to(device)

    # 损失函数（含类别权重）
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=scheduler_patience, min_lr=1e-7
    )
    early_stopper = EarlyStopping(patience=early_stop_patience, mode='min')

    # 保存目录
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        best_ckpt = os.path.join(save_dir, f'{model_name}_best.pt')
    else:
        best_ckpt = None

    history = {
        'train_loss': [], 'val_loss': [],
        'val_acc': [], 'val_score': [],
        'val_se': [], 'val_sp': [],
        'lr': [],
    }
    best_score = float('inf')   # 赶山 val_loss
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        # ── 训练一个 epoch ─────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = train_loss / max(n_train_batches, 1)

        # ── 验证 ────────────────────────────────────────────────────────────
        val_metrics, avg_val_loss = evaluate(model, val_loader, device, criterion)
        val_score = val_metrics['score']
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_score'].append(val_score)
        history['val_se'].append(val_metrics['sensitivity'])
        history['val_sp'].append(val_metrics['specificity'])
        history['lr'].append(current_lr)

        # 学习率调度
        scheduler.step(avg_val_loss)

        # 保存最优模型（按 val_loss 最低）
        if avg_val_loss < best_score:
            best_score = avg_val_loss
            best_epoch = epoch
            if best_ckpt:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_score': val_score,
                    'val_metrics': val_metrics,
                }, best_ckpt)

        # 日志
        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == num_epochs):
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{num_epochs}  "
                f"loss={avg_train_loss:.4f}  "
                f"val_loss={avg_val_loss:.4f}  "
                f"acc={val_metrics['accuracy']:.4f}  "
                f"Se={val_metrics['sensitivity']:.4f}  "
                f"Sp={val_metrics['specificity']:.4f}  "
                f"score={val_score:.4f}  "
                f"lr={current_lr:.2e}  "
                f"[{elapsed:.0f}s]"
            )

        # GUI 进度回调
        if progress_callback is not None:
            progress_callback(epoch, num_epochs, {
                'train_loss': avg_train_loss,
                'val_loss':   avg_val_loss,
                **val_metrics,
            })

        # 早停（监控 val_loss）
        if early_stopper.step(avg_val_loss):
            if verbose:
                print(f"早停触发：第 {epoch} 轮，最优轮次 {best_epoch} (val_loss={best_score:.4f})")
            break

    elapsed_total = time.time() - t0
    history['best_epoch'] = best_epoch
    history['best_val_score'] = best_score   # 实为最优 val_loss
    history['training_time_s'] = elapsed_total

    if verbose:
        print(f"\n训练完成：共 {epoch} 轮，耗时 {elapsed_total:.1f}s")
        print(f"最优轮次: {best_epoch}，验证 Score={(best_score):.4f}")

    return history


# ---------------------------------------------------------------------------
# 测试集终测
# ---------------------------------------------------------------------------

def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    在测试集上最终评估。可选从 checkpoint 加载最优权重。

    Returns
    -------
    dict: accuracy, sensitivity, specificity, score, tp, tn, fp, fn
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"已加载最优权重: {checkpoint_path}  (val_score={ckpt.get('val_score', '?'):.4f})")

    metrics, _ = evaluate(model, test_loader, device)
    print("\n===== 测试集结果 =====")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Sensitivity : {metrics['sensitivity']:.4f}  (异常检出率)")
    print(f"  Specificity : {metrics['specificity']:.4f}  (正常识别率)")
    print(f"  Score (Se+Sp)/2: {metrics['score']:.4f}")
    print(f"  混淆矩阵: TP={metrics['tp']} TN={metrics['tn']} "
          f"FP={metrics['fp']} FN={metrics['fn']}")
    return metrics


# ---------------------------------------------------------------------------
# 跨架构批量训练（消融实验）
# ---------------------------------------------------------------------------

def benchmark_architectures(
    archs: List[str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    input_size: tuple = (128, 256),
    num_epochs: int = 50,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
    pretrained: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    批量训练并比较多种 CNN 架构（复现论文 Table 3 对比实验）。

    Parameters
    ----------
    archs : 架构列表，如 ['lenet5', 'alexnet', 'resnet50']

    Returns
    -------
    results : {arch_name: {history, test_metrics}}
    """
    from cnn_models import build_model

    results = {}
    for arch in archs:
        print(f"\n{'='*60}")
        print(f"  训练架构: {arch.upper()}")
        print('='*60)

        try:
            model = build_model(arch=arch, input_size=input_size, pretrained=pretrained)
            save_subdir = os.path.join(save_dir, arch) if save_dir else None

            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                lr=lr,
                device=device,
                save_dir=save_subdir,
                model_name=arch,
                verbose=verbose,
            )

            # 测试集评估（加载最优权重）
            ckpt = None
            if save_subdir:
                ckpt = os.path.join(save_subdir, f'{arch}_best.pt')
            test_metrics = test_model(model, test_loader, device, ckpt)

            results[arch] = {
                'history': history,
                'test_metrics': test_metrics,
            }

        except Exception as e:
            print(f"  [错误] {arch}: {e}")
            results[arch] = {'error': str(e)}

    # 打印汇总表
    print(f"\n{'='*60}")
    print("  架构对比汇总（测试集）")
    print(f"{'='*60}")
    print(f"{'架构':>12}  {'Acc':>6}  {'Se':>6}  {'Sp':>6}  {'Pre':>6}  {'Score':>6}")
    print('-' * 52)
    for arch, r in results.items():
        if 'test_metrics' in r:
            m = r['test_metrics']
            print(f"{arch:>12}  {m['accuracy']:>6.4f}  {m['sensitivity']:>6.4f}  "
                  f"{m['specificity']:>6.4f}  {m['precision']:>6.4f}  {m['score']:>6.4f}")
        else:
            print(f"{arch:>12}  [错误: {r.get('error', '?')}]")

    return results


# ---------------------------------------------------------------------------
# 10-fold × 5次重复 交叉验证（论文 Section 4.2 完整复现）
# ---------------------------------------------------------------------------

def train_kfold(
    dataset,
    arch: str = 'ujanet',
    input_size: tuple = (128, 256),
    n_splits: int = 10,
    n_repeats: int = 5,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 0.001,
    early_stop_patience: int = 10,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
    verbose: bool = True,
    progress_callback=None,
) -> Dict:
    """
    论文 Section 4.2 完整 K-fold 训练协议：
      - 10-fold 交叉验证 × 5 次重复 = 50 个评估点（box plot 数据）
      - 每 fold: 75% 训练+验证 / 25% 测试（StratifiedShuffleSplit）
      - 训练中 25% 用作验证，早停 patience=10 监控 val_loss
      - 所有 fold 混淆矩阵累加 → 计算平均指标
      - 返回 per-fold 数据 + 平均混淆矩阵指标，供 box plot 可视化（Fig. 7）

    Parameters
    ----------
    dataset              : ONMFDataset 全量数据集
    arch                 : CNN 架构名（默认 'ujanet'，论文推荐）
    input_size           : (K, T) = (128, 256) 默认
    n_splits             : 每次重复的 fold 数（论文=10）
    n_repeats            : 重复次数（论文=5），保证 box plot 共 n_splits×n_repeats=50 点
    epochs               : 最大训练轮数（论文=30）
    batch_size           : batch 大小（论文=16）
    lr                   : Adam 初始学习率（论文=0.001）
    early_stop_patience  : 早停 patience（论文=10，监控 val_loss）
    pretrained           : 是否使用 ImageNet 预训练权重
    device               : torch.device（None 则自动选择）
    save_dir             : 最优 fold 模型保存目录（None 不保存）
    verbose              : 是否打印详细日志
    progress_callback    : callable(fold_done, total_folds, fold_metrics)

    Returns
    -------
    dict 包含：
      'fold_metrics'  : List[dict]，共 n_splits×n_repeats 个 fold 的测试结果
      'avg_metrics'   : dict，从累加混淆矩阵计算的整体平均指标
      'sum_cm'        : np.ndarray (2,2)，累加混淆矩阵 [[TP,FN],[FP,TN]]
      'n_splits'      : 实际 fold 数
      'n_repeats'     : 重复次数
    """
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
    except ImportError:
        raise ImportError(
            "需要 scikit-learn 支持K-fold CV。\n"
            "安装: pip install scikit-learn  或  conda install scikit-learn"
        )

    from torch.utils.data import Subset
    from cnn_models import build_model

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    labels = np.array(dataset.get_labels())
    total_folds = n_splits * n_repeats
    fold_idx = 0
    fold_metrics: List[Dict] = []
    sum_cm = np.zeros((2, 2), dtype=np.int64)   # [[TP, FN], [FP, TN]]

    best_fold_score = -1.0
    best_ckpt_path  = None

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for repeat in range(n_repeats):
        # 外层 split: 75% train+val, 25% test
        outer = StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=0.25,
            random_state=repeat * 1000 + 42
        )
        for train_val_idx, test_idx in outer.split(np.zeros(len(labels)), labels):
            fold_idx += 1
            if verbose:
                print(f"\n[Fold {fold_idx}/{total_folds}]  "
                      f"repeat={repeat+1}/{n_repeats}  "
                      f"train+val={len(train_val_idx)}, test={len(test_idx)}")

            # 内层 split: 从 train_val_idx 中再取 25% 作验证
            inner = StratifiedShuffleSplit(
                n_splits=1, test_size=0.25, random_state=fold_idx
            )
            sub_labels = labels[train_val_idx]
            train_rel, val_rel = next(
                inner.split(np.zeros(len(train_val_idx)), sub_labels)
            )
            train_idx = train_val_idx[train_rel]
            val_idx   = train_val_idx[val_rel]

            train_loader = DataLoader(
                Subset(dataset, train_idx),
                batch_size=batch_size, shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                Subset(dataset, val_idx),
                batch_size=batch_size, shuffle=False, num_workers=0
            )
            test_loader = DataLoader(
                Subset(dataset, test_idx),
                batch_size=batch_size, shuffle=False, num_workers=0
            )

            # 每 fold 重建模型
            model = build_model(
                arch=arch, input_size=input_size, pretrained=pretrained
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            stopper   = EarlyStopping(patience=early_stop_patience, mode='min')
            best_val_loss = float('inf')
            best_state    = None

            for epoch in range(1, epochs + 1):
                # 训练
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()

                # 验证
                model.eval()
                val_loss_sum, n_vb = 0.0, 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        val_loss_sum += criterion(model(xb), yb).item()
                        n_vb += 1
                avg_val_loss = val_loss_sum / max(n_vb, 1)

                # 保存最优 epoch（val_loss 最低）
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state    = {k: v.cpu().clone() for k, v in
                                     model.state_dict().items()}

                if stopper.step(avg_val_loss):
                    break

            # 加载该 fold 最优权重
            if best_state is not None:
                model.load_state_dict(
                    {k: v.to(device) for k, v in best_state.items()}
                )

            # 测试集评估
            test_metrics, _ = evaluate(model, test_loader, device)
            fold_metrics.append(test_metrics)

            # 累加混淆矩阵
            tp, tn = test_metrics['tp'], test_metrics['tn']
            fp, fn = test_metrics['fp'], test_metrics['fn']
            sum_cm[0, 0] += tp   # TP
            sum_cm[0, 1] += fn   # FN
            sum_cm[1, 0] += fp   # FP
            sum_cm[1, 1] += tn   # TN

            if verbose:
                m = test_metrics
                print(f"  -> Acc={m['accuracy']:.4f}  Se={m['sensitivity']:.4f}  "
                      f"Sp={m['specificity']:.4f}  Pre={m['precision']:.4f}  "
                      f"Sco={m['score']:.4f}")

            # 可选：保存最优 fold 模型
            if save_dir and test_metrics['score'] > best_fold_score:
                best_fold_score = test_metrics['score']
                best_ckpt_path  = os.path.join(
                    save_dir, f'{arch}_kfold_best.pt'
                )
                torch.save({
                    'fold': fold_idx, 'arch': arch,
                    'model_state_dict': {k: v.cpu() for k, v in
                                         model.state_dict().items()},
                    'test_metrics': test_metrics,
                }, best_ckpt_path)

            if progress_callback is not None:
                progress_callback(fold_idx, total_folds, test_metrics)

    # 从累加混淆矩阵计算整体平均指标
    tp_s = int(sum_cm[0, 0]); fn_s = int(sum_cm[0, 1])
    fp_s = int(sum_cm[1, 0]); tn_s = int(sum_cm[1, 1])
    total_n = tp_s + tn_s + fp_s + fn_s
    avg_acc  = (tp_s + tn_s) / max(total_n, 1)
    avg_se   = tp_s / max(tp_s + fn_s, 1)
    avg_sp   = tn_s / max(tn_s + fp_s, 1)
    avg_pre  = tp_s / max(tp_s + fp_s, 1)
    avg_sco  = (avg_se + avg_sp) / 2
    avg_f1   = 2 * avg_pre * avg_se / max(avg_pre + avg_se, 1e-9)

    avg_metrics = {
        'accuracy':    round(avg_acc, 4),
        'sensitivity': round(avg_se,  4),
        'specificity': round(avg_sp,  4),
        'precision':   round(avg_pre, 4),
        'score':       round(avg_sco, 4),
        'f1':          round(avg_f1,  4),
        'tp': tp_s, 'tn': tn_s, 'fp': fp_s, 'fn': fn_s,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {n_splits}-fold × {n_repeats}次重复  交叉验证汇总")
        print(f"{'='*60}")
        print(f"  Acc  = {avg_metrics['accuracy']:.4f}")
        print(f"  Se   = {avg_metrics['sensitivity']:.4f}  (异常检出率)")
        print(f"  Sp   = {avg_metrics['specificity']:.4f}  (正常识别率)")
        print(f"  Pre  = {avg_metrics['precision']:.4f}  (精确率)")
        print(f"  Sco  = {avg_metrics['score']:.4f}  (Se+Sp)/2")
        print(f"  F1   = {avg_metrics['f1']:.4f}")
        if best_ckpt_path:
            print(f"  最优Fold模型: {best_ckpt_path}")

    return {
        'fold_metrics':  fold_metrics,
        'avg_metrics':   avg_metrics,
        'sum_cm':        sum_cm,
        'n_splits':      n_splits,
        'n_repeats':     n_repeats,
        'arch':          arch,
        'best_ckpt':     best_ckpt_path,
    }
