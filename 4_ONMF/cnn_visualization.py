"""
CNN 训练结果可视化模块
======================
Torre-Cruz et al. 2023 Section 5

提供训练曲线、混淆矩阵、多架构对比等可视化函数。
与现有 visualization.py 保持独立，不产生命名冲突。
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Optional, List


# 中文字体配置（与主模块保持一致）
def _setup_font():
    import matplotlib.font_manager as fm
    import sys
    for font in ['Microsoft YaHei', 'SimHei', 'SimSun']:
        if font in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return
_setup_font()


# ---------------------------------------------------------------------------
# 训练历史曲线
# ---------------------------------------------------------------------------

def plot_training_history(
    history: Dict,
    arch_name: str = '',
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    绘制训练过程曲线：
      上左: 训练/验证 Loss
      上右: 验证 Accuracy
      下左: 验证 Sensitivity / Specificity
      下右: 验证 Score = (Se+Sp)/2 + 学习率
    """
    epochs = list(range(1, len(history['train_loss']) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    title = f'训练历史  |  {arch_name}' if arch_name else '训练历史'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='训练Loss', color='#2196F3')
    if history.get('val_loss'):
        ax.plot(epochs, history['val_loss'], label='验证Loss', color='#E53935', linestyle='--')
    ax.set_title('损失曲线')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history['val_acc'], label='验证Acc', color='#4CAF50')
    ax.axhline(max(history['val_acc']), color='#4CAF50', linestyle=':', alpha=0.5,
               label=f"最优={max(history['val_acc']):.4f}")
    ax.set_title('验证准确率')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Se / Sp
    ax = axes[1, 0]
    if history.get('val_se'):
        ax.plot(epochs, history['val_se'], label='Sensitivity', color='#FF9800')
    if history.get('val_sp'):
        ax.plot(epochs, history['val_sp'], label='Specificity', color='#9C27B0')
    ax.set_title('敏感性 / 特异性')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Score + LR
    ax = axes[1, 1]
    ax.plot(epochs, history['val_score'], label='Score=(Se+Sp)/2', color='#F44336', linewidth=2)
    best_e = history.get('best_epoch', int(np.argmax(history['val_score'])) + 1)
    best_s = history.get('best_val_score', max(history['val_score']))
    ax.axvline(best_e, color='#F44336', linestyle=':', alpha=0.7,
               label=f"最优 Epoch={best_e} ({best_s:.4f})")
    ax2 = ax.twinx()
    if history.get('lr'):
        ax2.semilogy(epochs, history['lr'], color='gray', alpha=0.4, linewidth=0.8)
        ax2.set_ylabel('LR', color='gray')
    ax.set_title('验证 Score = (Se+Sp)/2')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# 混淆矩阵
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    metrics: Dict,
    arch_name: str = '',
    show: bool = True,
    save_path: Optional[str] = None,
):
    """绘制混淆矩阵热力图 + 指标列表。"""
    tp = metrics['tp']
    tn = metrics['tn']
    fp = metrics['fp']
    fn = metrics['fn']
    cm = np.array([[tn, fp], [fn, tp]])
    labels = ['正常 (0)', '异常 (1)']

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    title = f'混淆矩阵  |  {arch_name}' if arch_name else '混淆矩阵'
    fig.suptitle(title, fontsize=12, fontweight='bold')

    # 热力图
    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues', vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f'预测: {l}' for l in labels])
    ax.set_yticklabels([f'真实: {l}' for l in labels])
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = 'white' if val > cm.max() / 2 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=16, color=color, fontweight='bold')
    plt.colorbar(im, ax=ax, label='样本数')
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    ax.set_title('混淆矩阵')

    # 指标表格
    ax2 = axes[1]
    ax2.axis('off')
    rows = [
        ['指标', '值'],
        ['Accuracy',          f"{metrics.get('accuracy', 0):.4f}"],
        ['Sensitivity (Se)',   f"{metrics.get('sensitivity', 0):.4f}"],
        ['Specificity (Sp)',   f"{metrics.get('specificity', 0):.4f}"],
        ['Precision (Pre)',    f"{metrics.get('precision', 0):.4f}"],
        ['Score (Se+Sp)/2',   f"{metrics.get('score', 0):.4f}"],
        ['F1-Score',          f"{metrics.get('f1', 0):.4f}"],
        ['TP', str(tp)],
        ['TN', str(tn)],
        ['FP', str(fp)],
        ['FN', str(fn)],
    ]
    table = ax2.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 1.8)
    ax2.set_title('评估指标')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# 多架构对比柱状图（复现论文 Table 3 风格）
# ---------------------------------------------------------------------------

def plot_arch_comparison(
    results: Dict[str, Dict],
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    绘制多架构测试集指标对比柱状图。

    Parameters
    ----------
    results: {arch_name: {'test_metrics': {...}}}
    """
    archs, accs, ses, sps, pres, scores = [], [], [], [], [], []
    for arch, r in results.items():
        if 'test_metrics' not in r:
            continue
        m = r['test_metrics']
        archs.append(arch.upper())
        accs.append(m.get('accuracy', 0))
        ses.append(m.get('sensitivity', 0))
        sps.append(m.get('specificity', 0))
        pres.append(m.get('precision', 0))
        scores.append(m.get('score', 0))

    if not archs:
        print("无有效结果可绘制")
        return None

    x = np.arange(len(archs))
    width = 0.15

    fig, ax = plt.subplots(figsize=(max(8, len(archs) * 2.5), 6))
    ax.bar(x - 2.0 * width, accs,  width, label='Accuracy',    color='#2196F3', alpha=0.85)
    ax.bar(x - 1.0 * width, ses,   width, label='Sensitivity', color='#FF9800', alpha=0.85)
    ax.bar(x + 0.0 * width, sps,   width, label='Specificity', color='#9C27B0', alpha=0.85)
    ax.bar(x + 1.0 * width, pres,  width, label='Precision',   color='#4CAF50', alpha=0.85)
    ax.bar(x + 2.0 * width, scores, width, label='Score=(Se+Sp)/2', color='#F44336', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('性能指标')
    ax.set_title('CNN 架构性能对比（测试集）\n Torre-Cruz et al. 2023 Table 3 风格', fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(0.85, color='gray', linestyle='--', alpha=0.5, label='85% 参考线')

    # 在最高 score 上标注数值
    for i, (arch, sc) in enumerate(zip(archs, scores)):
        ax.text(i + 1.5 * width, sc + 0.01, f'{sc:.3f}',
                ha='center', va='bottom', fontsize=9, color='#F44336')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# K-fold 交叉验证结果 Box Plot（复现论文 Fig. 7）
# ---------------------------------------------------------------------------

def plot_kfold_boxplots(
    kfold_result: Dict,
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    绘制 K-fold 交叉验证结果的 box plot，复现论文 Fig. 7。
    5 个子图对应: Acc / Se / Sp / Pre / Score(Sco)

    Parameters
    ----------
    kfold_result : train_kfold() 的返回值
    """
    fold_metrics = kfold_result['fold_metrics']
    avg_metrics  = kfold_result['avg_metrics']
    arch_name    = kfold_result.get('arch', '')
    n_splits     = kfold_result.get('n_splits', 10)
    n_repeats    = kfold_result.get('n_repeats', 5)

    metric_keys   = ['accuracy', 'sensitivity', 'specificity', 'precision', 'score']
    metric_labels = ['Acc', 'Se', 'Sp', 'Pre', 'Sco']
    colors        = ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50', '#F44336']

    data = [[m[k] for m in fold_metrics] for k in metric_keys]

    fig, axes = plt.subplots(1, 5, figsize=(16, 5), sharey=False)
    title = (f'K-fold 交叉验证结果  |  {arch_name.upper()}\n'
             f'{n_splits}-fold × {n_repeats}次重复 = {len(fold_metrics)} 个评估点'
             f'（复现论文 Fig. 7）')
    fig.suptitle(title, fontsize=12, fontweight='bold')

    for ax, vals, label, color in zip(axes, data, metric_labels, colors):
        bp = ax.boxplot(
            vals,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='black',
                           markeredgecolor='black', markersize=6),
            medianprops=dict(color='black', linewidth=2),
            boxprops=dict(facecolor=color, alpha=0.5),
            whiskerprops=dict(color=color),
            capprops=dict(color=color),
            flierprops=dict(marker='o', color=color, alpha=0.5),
        )
        avg_val = avg_metrics[metric_keys[metric_labels.index(label)]]
        ax.set_title(f'{label}\n均值={avg_val:.4f}', fontsize=10)
        ax.set_ylabel('指标值')
        ax.set_ylim(max(0, min(vals) - 0.1), min(1.05, max(vals) + 0.1))
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticks([])
        # 标注数据点
        jitter = np.random.uniform(-0.15, 0.15, len(vals))
        ax.scatter(1 + jitter, vals, color=color, alpha=0.4, s=15, zorder=3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# 样本预测可视化（分析错误样本）
# ---------------------------------------------------------------------------

def plot_prediction_samples(
    model,
    dataset,
    device,
    n_samples: int = 8,
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    可视化若干测试样本的 H 矩阵及预测结果（正确=绿色边框，错误=红色边框）。
    """
    import torch
    model.eval()
    class_names = ['正常', '异常']
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    cols = min(4, n_samples)
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).flatten()
    fig.suptitle('预测样本可视化  （绿色=正确，红色=错误）', fontsize=12)

    with torch.no_grad():
        for plot_i, idx in enumerate(indices):
            x, true_label = dataset[idx]
            logit = model(x.unsqueeze(0).to(device))
            pred = logit.argmax(dim=1).item()
            proba = torch.softmax(logit, dim=1)[0][pred].item()

            ax = axes[plot_i]
            h_img = x.squeeze().cpu().numpy()   # (K, T) or (T,)
            if h_img.ndim == 1:
                h_img = h_img.reshape(1, -1)
            ax.imshow(h_img, aspect='auto', origin='lower',
                      cmap='hot', vmin=0, vmax=1)
            ax.set_title(
                f'真实: {class_names[true_label]}\n'
                f'预测: {class_names[pred]} ({proba:.2f})',
                fontsize=9
            )
            ax.axis('off')
            # 边框颜色
            color = '#4CAF50' if pred == true_label else '#F44336'
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

    # 隐藏多余子图
    for i in range(len(indices), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig
