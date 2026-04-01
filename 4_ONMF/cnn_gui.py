"""
CNN 训练 GUI — Torre-Cruz et al. 2023 Section 5
================================================
独立训练界面：加载已生成的 ONMF 特征 (.npy) → 配置 CNN → 训练 → 评估 → 可视化

使用前需先运行主程序 main_gui.py 保存特征矩阵 (*_H.npy / *_W.npy)
"""

import os
import sys
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm


# 中文字体
def _setup_chinese_font():
    candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'STSong']
    available = {f.name for f in _fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return
    fp = r'C:\Windows\Fonts\msyh.ttc'
    if os.path.exists(fp):
        prop = _fm.FontProperties(fname=fp)
        plt.rcParams['font.sans-serif'] = [prop.get_name(), 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

_setup_chinese_font()


# ---------------------------------------------------------------------------
# 全局训练状态
# ---------------------------------------------------------------------------
_train_result   = {}    # 保存训练历史和测试结果
_kfold_result   = {}    # 保存 k-fold 结果
_trained_model  = None  # 保存训练完成的模型对象
_test_dataset   = None  # 保存测试集 dataset


def _log(widget, msg: str):
    widget.config(state=tk.NORMAL)
    widget.insert(tk.END, msg + '\n')
    widget.see(tk.END)
    widget.config(state=tk.DISABLED)


# ---------------------------------------------------------------------------
# 训练线程
# ---------------------------------------------------------------------------

def _train_thread(params, log_widget, status_var, progress_var,
                  run_btn, root, result_frame_updater):
    global _train_result, _kfold_result, _trained_model, _test_dataset

    try:
        import torch
        from cnn_dataset import make_dataloaders, ONMFDataset
        from cnn_models import build_model, model_summary
        from cnn_trainer import train_model, test_model, train_kfold

        # ── 设备 ──────────────────────────────────────────────────────────
        if torch.cuda.is_available() and params['use_gpu']:
            device = torch.device('cuda')
        elif (hasattr(torch.backends, 'mps') and
              torch.backends.mps.is_available() and params['use_gpu']):
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

        _log(log_widget, f'[设备] {device}')

        mode = params.get('mode', 'simple')  # 'simple' or 'kfold'

        if mode == 'kfold':
            # ===== K-fold 交叉验证模式（论文 Section 4.2）=====
            status_var.set('加载全量数据集...')
            label_src = params['label_source'] if params['label_source'] else 'filename'
            full_ds = ONMFDataset(
                feature_dir=params['feature_dir'],
                label_source=label_src,
                feature_type=params['feature_type'],
                target_size=(params['input_k'], params['input_t']),
            )
            _test_dataset = full_ds
            from collections import Counter
            dist = Counter(full_ds.get_labels())
            _log(log_widget, f'[数据集] 共 {len(full_ds)} 样本 '
                 f'(正常={dist[0]}, 异常={dist[1]})')

            model_tmp = build_model(
                arch=params['arch'],
                input_size=(params['input_k'], params['input_t'])
            )
            n_params = sum(p.numel() for p in model_tmp.parameters() if p.requires_grad)
            _log(log_widget, f'[模型] {params["arch"].upper()}  参数量: {n_params:,}')
            del model_tmp

            n_splits  = params.get('n_splits', 10)
            n_repeats = params.get('n_repeats', 5)
            total_folds = n_splits * n_repeats
            _log(log_widget, f'[K-fold] {n_splits}-fold × {n_repeats}次重复 = '
                 f'{total_folds} 个评估点（论文 Fig. 7）')

            def _kfold_progress(fold_done, total, fold_m):
                progress_var.set(int(fold_done / total * 100))
                status_var.set(
                    f'Fold {fold_done}/{total}  '
                    f'Sco={fold_m.get("score", 0):.4f}'
                )
                root.update_idletasks()

            result = train_kfold(
                dataset=full_ds,
                arch=params['arch'],
                input_size=(params['input_k'], params['input_t']),
                n_splits=n_splits,
                n_repeats=n_repeats,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                lr=params['lr'],
                early_stop_patience=params['early_stop'],
                pretrained=params['pretrained'],
                device=device,
                save_dir=params.get('save_dir') or None,
                verbose=True,
                progress_callback=_kfold_progress,
            )

            _kfold_result.update(result)
            _train_result.update({'kfold': result, 'arch': params['arch']})

            avg = result['avg_metrics']
            _log(log_widget, '====== K-fold 训练完成 ======')
            _log(log_widget,
                 f'平均  Acc={avg["accuracy"]:.4f}  '
                 f'Se={avg["sensitivity"]:.4f}  '
                 f'Sp={avg["specificity"]:.4f}  '
                 f'Pre={avg["precision"]:.4f}  '
                 f'Sco={avg["score"]:.4f}  '
                 f'F1={avg["f1"]:.4f}')
            status_var.set(f'完成！Sco={avg["score"]:.4f}')
            progress_var.set(100)

        else:
            # ===== 简单 split 模式 =====
            status_var.set('准备数据集...')
            label_src = params['label_source'] if params['label_source'] else 'filename'
            train_loader, val_loader, test_loader, full_ds = make_dataloaders(
                feature_dir=params['feature_dir'],
                label_source=label_src,
                feature_type=params['feature_type'],
                target_size=(params['input_k'], params['input_t']),
                train_ratio=params['train_ratio'],
                val_ratio=params['val_ratio'],
                batch_size=params['batch_size'],
                num_workers=0,
                balance_train=params['balance'],
            )
            _test_dataset = full_ds

            _log(log_widget, f'[数据集] 训练/验证/测试: '
                 f'{len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}')

            status_var.set(f'构建模型 {params["arch"]}...')
            input_size = (params['input_k'], params['input_t'])
            model = build_model(
                arch=params['arch'],
                input_size=input_size,
                pretrained=params['pretrained'],
                dropout=params['dropout'],
            )
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            _log(log_widget, f'[模型] {params["arch"].upper()}  参数量: {n_params:,}')

            class_weights = None
            if params['balance']:
                labels = full_ds.get_labels()
                n_total = len(labels)
                n0 = labels.count(0); n1 = labels.count(1)
                w0 = n_total / (2.0 * n0) if n0 > 0 else 1.0
                w1 = n_total / (2.0 * n1) if n1 > 0 else 1.0
                class_weights = torch.tensor([w0, w1], dtype=torch.float32)
                _log(log_widget, f'[类别权重] 正常={w0:.3f}, 异常={w1:.3f}')

            def _progress_cb(epoch, total, metrics):
                progress_var.set(int(epoch / total * 100))
                status_var.set(
                    f'训练 Epoch {epoch}/{total}  '
                    f'Sco={metrics.get("score", 0):.4f}'
                )
                root.update_idletasks()

            status_var.set('训练中...')
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=params['epochs'],
                lr=params['lr'],
                weight_decay=params['weight_decay'],
                device=device,
                save_dir=params.get('save_dir') or None,
                model_name=params['arch'],
                class_weights=class_weights,
                scheduler_patience=5,
                early_stop_patience=params['early_stop'],
                verbose=True,
                progress_callback=_progress_cb,
            )

            status_var.set('测试集评估...')
            ckpt = None
            if params.get('save_dir'):
                ckpt_path = os.path.join(
                    params['save_dir'], params['arch'],
                    f'{params["arch"]}_best.pt'
                )
                if os.path.isfile(ckpt_path):
                    ckpt = ckpt_path

            test_metrics = test_model(model, test_loader, device, ckpt)

            _trained_model = model
            _train_result.update({
                'history': history,
                'test_metrics': test_metrics,
                'arch': params['arch'],
                'device': str(device),
            })

            _log(log_widget, '====== 训练完成 ======')
            _log(log_widget,
                 f'最优轮次: {history["best_epoch"]}, '
                 f'最优 val_loss={history["best_val_score"]:.4f}')
            _log(log_widget,
                 f'测试  Acc={test_metrics["accuracy"]:.4f}  '
                 f'Se={test_metrics["sensitivity"]:.4f}  '
                 f'Sp={test_metrics["specificity"]:.4f}  '
                 f'Pre={test_metrics["precision"]:.4f}  '
                 f'Sco={test_metrics["score"]:.4f}  '
                 f'F1={test_metrics["f1"]:.4f}')
            status_var.set(f'完成！Sco={test_metrics["score"]:.4f}')
            progress_var.set(100)

        root.after(0, result_frame_updater)

    except ImportError as e:
        _log(log_widget, f'\n[依赖缺失] {e}')
        _log(log_widget, '请安装: pip install torch torchvision scikit-learn')
        status_var.set('错误：依赖缺失')
    except Exception as e:
        _log(log_widget, f'\n[错误] {e}\n{traceback.format_exc()}')
        status_var.set('训练出错')
    finally:
        root.after(0, lambda: run_btn.config(state=tk.NORMAL))
        root.after(0, lambda: progress_var.set(0))


# ---------------------------------------------------------------------------
# 主 GUI 构建
# ---------------------------------------------------------------------------

def build_cnn_gui():
    root = tk.Tk()
    root.title('CNN 分类训练  —  Torre-Cruz et al. 2023 Section 5')
    root.geometry('1100x800')
    root.resizable(True, True)

    # ── 标题 ──────────────────────────────────────────────────────────────
    header = tk.Label(
        root,
        text='CNN PCG 心音分类训练',
        font=('Microsoft YaHei', 14, 'bold'),
        fg='#1a5276', pady=8
    )
    header.pack(fill=tk.X)

    sub = tk.Label(
        root,
        text='输入：ONMF H矩阵特征 (.npy)  →  CNN（LeNet5/AlexNet/VGG16/ResNet50/GoogLeNet）  →  正常/异常分类',
        font=('Microsoft YaHei', 9), fg='#555'
    )
    sub.pack()
    ttk.Separator(root, orient='horizontal').pack(fill=tk.X, pady=4)

    # ── 主布局 ────────────────────────────────────────────────────────────
    main = ttk.Frame(root)
    main.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

    left = ttk.Frame(main, width=360)
    left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
    left.pack_propagate(False)

    right = ttk.Frame(main)
    right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # ================================================================
    # 左侧：配置面板
    # ================================================================

    # ── 路径 ──────────────────────────────────────────────────────────────
    path_lf = ttk.LabelFrame(left, text='数据路径', padding=8)
    path_lf.pack(fill=tk.X, pady=(0, 6))

    feat_dir_var    = tk.StringVar()
    label_csv_var   = tk.StringVar()
    save_dir_var    = tk.StringVar()

    def _browse(var, title='选择文件夹'):
        d = filedialog.askdirectory(title=title)
        if d: var.set(d)

    def _browse_csv(var):
        f = filedialog.askopenfilename(
            title='选择标签CSV文件',
            filetypes=[('CSV文件', '*.csv'), ('所有文件', '*.*')]
        )
        if f: var.set(f)

    def _make_path_row(parent, label, var, row, browse_fn):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w')
        e = ttk.Entry(parent, textvariable=var, width=26)
        e.grid(row=row, column=1, sticky='ew', padx=(4, 0))
        ttk.Button(parent, text='浏览', width=5,
                   command=browse_fn).grid(row=row, column=2, padx=(4, 0))

    _make_path_row(path_lf, '特征目录:', feat_dir_var, 0,
                   lambda: _browse(feat_dir_var, '选择 *_H.npy 所在目录'))
    ttk.Label(path_lf, text='(保存了 *_H.npy 的文件夹)', foreground='gray',
              font=('', 8)).grid(row=1, column=1, sticky='w')
    _make_path_row(path_lf, '标签CSV:', label_csv_var, 2,
                   lambda: _browse_csv(label_csv_var))
    ttk.Label(path_lf, text='(可选: 留空则从文件名推断标签)', foreground='gray',
              font=('', 8)).grid(row=3, column=1, sticky='w')
    _make_path_row(path_lf, '保存目录:', save_dir_var, 4,
                   lambda: _browse(save_dir_var, '选择模型/结果保存目录'))
    path_lf.columnconfigure(1, weight=1)

    # ── 模型参数 ───────────────────────────────────────────────────────────
    model_lf = ttk.LabelFrame(left, text='模型参数', padding=8)
    model_lf.pack(fill=tk.X, pady=(0, 6))

    def _row(parent, label, row, default, width=8, hint=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=2)
        var = tk.StringVar(value=str(default))
        ttk.Entry(parent, textvariable=var, width=width).grid(
            row=row, column=1, sticky='w', padx=4)
        if hint:
            ttk.Label(parent, text=hint, foreground='gray',
                      font=('', 8)).grid(row=row, column=2, sticky='w')
        return var

    # 架构下拉
    ttk.Label(model_lf, text='CNN架构:').grid(row=0, column=0, sticky='w', pady=2)
    arch_var = tk.StringVar(value='ujanet')
    arch_cb  = ttk.Combobox(
        model_lf, textvariable=arch_var, width=14,
        values=['ujanet', 'lenet5', 'alexnet', 'vgg16', 'resnet50', 'googlenet'],
        state='readonly'
    )
    arch_cb.grid(row=0, column=1, sticky='w', padx=4)
    ttk.Label(model_lf, text='论文推荐 ujanet', foreground='#c0392b',
              font=('', 8, 'bold')).grid(row=0, column=2, sticky='w')

    # 特征类型
    ttk.Label(model_lf, text='特征类型:').grid(row=1, column=0, sticky='w', pady=2)
    feat_type_var = tk.StringVar(value='H')
    frame_ft = ttk.Frame(model_lf)
    frame_ft.grid(row=1, column=1, columnspan=2, sticky='w')
    ttk.Radiobutton(frame_ft, text='H（时域）', variable=feat_type_var, value='H').pack(side=tk.LEFT)
    ttk.Radiobutton(frame_ft, text='W（频域）', variable=feat_type_var, value='W').pack(side=tk.LEFT)

    input_k_var    = _row(model_lf, '输入K维:', 2, 128, hint='ONMF秩K')
    input_t_var    = _row(model_lf, '输入T帧:', 3, 256, hint='时间帧数(调整)')
    pretrained_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(model_lf, text='使用预训练权重（ImageNet）',
                    variable=pretrained_var).grid(row=4, columnspan=3, sticky='w')
    dropout_var    = _row(model_lf, 'Dropout:', 5, 0.3, hint='0=不使用')

    # ── 训练模式 + 训练参数 ─────────────────────────────────────────────
    train_lf = ttk.LabelFrame(left, text='训练参数', padding=8)
    train_lf.pack(fill=tk.X, pady=(0, 6))

    # 训练模式选择
    ttk.Label(train_lf, text='训练模式:').grid(row=0, column=0, sticky='w', pady=2)
    mode_var = tk.StringVar(value='kfold')
    mode_frame = ttk.Frame(train_lf)
    mode_frame.grid(row=0, column=1, columnspan=2, sticky='w')
    ttk.Radiobutton(mode_frame, text='10-fold CV（论文）',
                    variable=mode_var, value='kfold').pack(side=tk.LEFT)
    ttk.Radiobutton(mode_frame, text='简单分割',
                    variable=mode_var, value='simple').pack(side=tk.LEFT)

    epochs_var = _row(train_lf, '最大Epoch:', 1, 30,   hint='论文=30')
    lr_var     = _row(train_lf, '学习率:',   2, '0.001', hint='Adam论文=0.001')
    wd_var     = _row(train_lf, 'Weight Decay:', 3, '0')
    bs_var     = _row(train_lf, 'Batch Size:',  4, 16,    hint='论文=16')
    early_var  = _row(train_lf, '早停patience:', 5, 10,   hint='论文=10,监控val_loss')

    # K-fold 参数（仅 kfold 模式显示）
    kfold_frame = ttk.Frame(train_lf)
    kfold_frame.grid(row=6, column=0, columnspan=3, sticky='w', pady=(2, 0))
    n_splits_var  = tk.StringVar(value='10')
    n_repeats_var = tk.StringVar(value='5')
    ttk.Label(kfold_frame, text='K折数:').pack(side=tk.LEFT)
    ttk.Entry(kfold_frame, textvariable=n_splits_var, width=4).pack(side=tk.LEFT, padx=2)
    ttk.Label(kfold_frame, text='重复次数:').pack(side=tk.LEFT, padx=(8, 0))
    ttk.Entry(kfold_frame, textvariable=n_repeats_var, width=4).pack(side=tk.LEFT, padx=2)
    ttk.Label(kfold_frame, text='→共50个评估点', foreground='gray',
              font=('', 8)).pack(side=tk.LEFT, padx=4)

    def _on_mode_change(*_):
        if mode_var.get() == 'kfold':
            kfold_frame.grid()
            simple_frame.grid_remove()
        else:
            kfold_frame.grid_remove()
            simple_frame.grid()
    mode_var.trace_add('write', _on_mode_change)

    # 简单分割参数配置
    simple_frame = ttk.Frame(train_lf)
    simple_frame.grid(row=7, column=0, columnspan=3, sticky='w')
    train_r_var = tk.StringVar(value='0.70')
    val_r_var   = tk.StringVar(value='0.15')
    ttk.Label(simple_frame, text='训练比:').pack(side=tk.LEFT)
    ttk.Entry(simple_frame, textvariable=train_r_var, width=5).pack(side=tk.LEFT, padx=2)
    ttk.Label(simple_frame, text='验证比:').pack(side=tk.LEFT, padx=(6, 0))
    ttk.Entry(simple_frame, textvariable=val_r_var, width=5).pack(side=tk.LEFT, padx=2)
    simple_frame.grid_remove()  # 默认隐藏

    balance_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(train_lf, text='平衡类别（仅简单模式）',
                    variable=balance_var).grid(row=8, columnspan=3, sticky='w')
    gpu_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(train_lf, text='使用 GPU（CUDA/MPS，如可用）',
                    variable=gpu_var).grid(row=9, columnspan=3, sticky='w')
    dropout_var = _row(model_lf, 'Dropout:', 5, 0.5, hint='论文=0.5')

    # ── 运行按钮 ───────────────────────────────────────────────────────────
    status_var   = tk.StringVar(value='就绪')
    progress_var = tk.IntVar(value=0)

    run_btn = ttk.Button(left, text='▶  开始训练')
    run_btn.pack(fill=tk.X, pady=4)

    ttk.Progressbar(left, variable=progress_var, maximum=100).pack(fill=tk.X, pady=2)
    ttk.Label(left, textvariable=status_var, foreground='#1a5276', font=('', 9)).pack()

    # ================================================================
    # 右侧：可视化按钮 + 日志
    # ================================================================

    vis_lf = ttk.LabelFrame(right, text='结果可视化', padding=6)
    vis_lf.pack(fill=tk.X, pady=(0, 4))

    vis_btn_frame = ttk.Frame(vis_lf)
    vis_btn_frame.pack(fill=tk.X)

    def _check_result(action_name):
        if not _train_result:
            messagebox.showinfo('提示', f'请先完成训练后再{action_name}')
            return False
        return True

    def _show_fig(fig, title='可视化结果'):
        """将 matplotlib figure 嵌入 Toplevel 子窗口（避免多 tk.Tk() 冲突）"""
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        top = tk.Toplevel(root)
        top.title(title)
        top.resizable(True, True)
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, top)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        top.lift()
        top.focus_force()

    def _show_history():
        if not _check_result('查看训练曲线'): return
        if 'history' not in _train_result:
            messagebox.showinfo('提示', 'K-fold 模式下无单次训练曲线，\n请使用 Box Plot 查看各 fold 汇总结果。')
            return
        from cnn_visualization import plot_training_history
        plt.close('all')
        fig = plot_training_history(_train_result['history'],
                                    arch_name=_train_result.get('arch', ''),
                                    show=False)
        _show_fig(fig, f'训练曲线 — {_train_result.get("arch", "")}')

    def _show_confusion():
        if not _check_result('查看混淆矩阵'): return
        if 'test_metrics' not in _train_result:
            if _kfold_result and 'avg_metrics' in _kfold_result:
                from cnn_visualization import plot_confusion_matrix
                plt.close('all')
                fig = plot_confusion_matrix(_kfold_result['avg_metrics'],
                                            arch_name=_train_result.get('arch', ''),
                                            show=False)
                _show_fig(fig, f'混淆矩阵（K-fold均值）— {_train_result.get("arch", "")}')
            else:
                messagebox.showinfo('提示', 'K-fold 模式下无单次混淆矩阵，\n请使用 Box Plot 查看各 fold 汇总结果。')
            return
        from cnn_visualization import plot_confusion_matrix
        plt.close('all')
        fig = plot_confusion_matrix(_train_result['test_metrics'],
                                    arch_name=_train_result.get('arch', ''),
                                    show=False)
        _show_fig(fig, f'混淆矩阵 — {_train_result.get("arch", "")}')

    def _show_samples():
        if not _check_result('查看样本预测'): return
        if _trained_model is None or _test_dataset is None:
            messagebox.showinfo('提示', '模型或测试集尚未就绪')
            return
        from cnn_visualization import plot_prediction_samples
        import torch
        device = torch.device('cpu')
        plt.close('all')
        fig = plot_prediction_samples(_trained_model, _test_dataset,
                                      device=device, n_samples=8, show=False)
        _show_fig(fig, '样本预测')

    def _save_model():
        if not _check_result('保存模型'): return
        if _trained_model is None:
            messagebox.showinfo('提示', '无已训练模型')
            return
        import torch
        path = filedialog.asksaveasfilename(
            title='保存模型',
            defaultextension='.pt',
            filetypes=[('PyTorch模型', '*.pt'), ('所有文件', '*.*')]
        )
        if path:
            torch.save({
                'model_state_dict': _trained_model.state_dict(),
                'arch': _train_result.get('arch'),
                'test_metrics': _train_result.get('test_metrics'),
            }, path)
            _log(log_widget, f'模型已保存: {path}')
            messagebox.showinfo('保存成功', f'模型已保存到:\n{path}')

    def _show_boxplot():
        if not _kfold_result:
            messagebox.showinfo('提示', '请先完成 K-fold 训练后再查看 Box Plot')
            return
        from cnn_visualization import plot_kfold_boxplots
        plt.close('all')
        fig = plot_kfold_boxplots(_kfold_result, show=False)
        _show_fig(fig, f'K-fold Box Plot — {_kfold_result.get("arch", "")}')

    vis_buttons = [
        ('Box Plot',   _show_boxplot),
        ('训练曲线',   _show_history),
        ('混淤矩阵',   _show_confusion),
        ('保存模型',   _save_model),
    ]
    for text, cmd in vis_buttons:
        ttk.Button(vis_btn_frame, text=text, command=cmd, width=12).pack(
            side=tk.LEFT, padx=4, pady=2)

    # 结果摘要标签（训练完成后刷新）
    result_var = tk.StringVar(value='（训练完成后显示结果）')
    result_lbl = ttk.Label(vis_lf, textvariable=result_var,
                           font=('Consolas', 10), foreground='#1a5276')
    result_lbl.pack(anchor='w', padx=4, pady=(4, 0))

    def _update_result_label():
        if _kfold_result and 'avg_metrics' in _kfold_result:
            m = _kfold_result['avg_metrics']
            result_var.set(
                f'K-fold均值 → Acc={m["accuracy"]:.4f}  '
                f'Se={m["sensitivity"]:.4f}  '
                f'Sp={m["specificity"]:.4f}  '
                f'Pre={m["precision"]:.4f}  '
                f'Sco={m["score"]:.4f}  '
                f'F1={m["f1"]:.4f}'
            )
        elif _train_result and 'test_metrics' in _train_result:
            m = _train_result['test_metrics']
            result_var.set(
                f'测试集 → Acc={m["accuracy"]:.4f}  '
                f'Se={m["sensitivity"]:.4f}  '
                f'Sp={m["specificity"]:.4f}  '
                f'Pre={m["precision"]:.4f}  '
                f'Sco={m["score"]:.4f}  '
                f'F1={m["f1"]:.4f}'
            )

    # 日志
    log_lf = ttk.LabelFrame(right, text='训练日志', padding=4)
    log_lf.pack(fill=tk.BOTH, expand=True)

    log_widget = scrolledtext.ScrolledText(
        log_lf, state=tk.DISABLED,
        font=('Consolas', 8),
        bg='#1e1e1e', fg='#d4d4d4',
        insertbackground='white', height=22
    )
    log_widget.pack(fill=tk.BOTH, expand=True)

    # ── 运行按钮回调 ──────────────────────────────────────────────────────
    def _on_run():
        feat_dir = feat_dir_var.get().strip()
        if not feat_dir or not os.path.isdir(feat_dir):
            messagebox.showerror('错误', '请选择有效的特征目录（含 *_H.npy 文件）')
            return

        try:
            params = {
                'feature_dir':   feat_dir,
                'label_source':  label_csv_var.get().strip(),
                'save_dir':      save_dir_var.get().strip(),
                'arch':          arch_var.get(),
                'feature_type':  feat_type_var.get(),
                'input_k':       int(input_k_var.get()),
                'input_t':       int(input_t_var.get()),
                'pretrained':    pretrained_var.get(),
                'dropout':       float(dropout_var.get()),
                'epochs':        int(epochs_var.get()),
                'lr':            float(lr_var.get()),
                'weight_decay':  float(wd_var.get()),
                'batch_size':    int(bs_var.get()),
                'early_stop':    int(early_var.get()),
                'mode':          mode_var.get(),
                'n_splits':      int(n_splits_var.get()),
                'n_repeats':     int(n_repeats_var.get()),
                'train_ratio':   float(train_r_var.get()),
                'val_ratio':     float(val_r_var.get()),
                'balance':       balance_var.get(),
                'use_gpu':       gpu_var.get(),
            }
        except ValueError as e:
            messagebox.showerror('参数错误', str(e))
            return

        run_btn.config(state=tk.DISABLED)
        _log(log_widget, f'开始训练: 架构={params["arch"]}, 模式={params["mode"]}, 特征={params["feature_type"]}')
        _log(log_widget, f'特征目录: {feat_dir}')

        t = threading.Thread(
            target=_train_thread,
            args=(params, log_widget, status_var, progress_var,
                  run_btn, root, _update_result_label),
            daemon=True
        )
        t.start()

    run_btn.config(command=_on_run)

    # ── 底部 ──────────────────────────────────────────────────────────────
    ttk.Separator(root, orient='horizontal').pack(fill=tk.X, padx=10)
    ttk.Label(
        root,
        text='Torre-Cruz et al. 2023 | 评估指标: Acc / Se / Sp / Pre / Sco=(Se+Sp)/2 / F1  |  训练协议: 10-fold×5重复=50个评估点',
        font=('', 8), foreground='gray'
    ).pack(pady=3)

    return root


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    root = build_cnn_gui()
    root.mainloop()


if __name__ == '__main__':
    main()
