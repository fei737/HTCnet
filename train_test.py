import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_processor import get_dataloader
from tci_net import TCI_Net, TCI_Loss
from metrics import calculate_metrics  # 下文实现

# 训练配置
config = {
    'data_root': '/path/to/NEU_RSDDS_AUG',  # 替换为你的数据集路径
    'batch_size': 16,
    'epochs': 600,
    'lr': 1e-4,
    'weight_decay': 0.0005,
    'device': torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'save_path': './tci_net_best.pth'
}


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        rgb, depth, rgbd = batch['rgb'].to(device), batch['depth'].to(device), batch['rgbd'].to(device)
        gt_edge, gt = batch['edge'].to(device), batch['gt'].to(device)

        optimizer.zero_grad()
        edge_pred, final_pred = model(rgb, depth, rgbd)
        loss = criterion(edge_pred, final_pred, gt_edge, gt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * rgb.size(0)
    return total_loss / len(dataloader.dataset)


def test_epoch(model, dataloader, device):
    model.eval()
    metrics = {'Sm': [], 'Em': [], 'Fm': [], 'MAE': [], 'IoU': []}
    with torch.no_grad():
        for batch in dataloader:
            rgb, depth, rgbd = batch['rgb'].to(device), batch['depth'].to(device), batch['rgbd'].to(device)
            gt = batch['gt'].to(device)
            _, final_pred = model(rgb, depth, rgbd)

            # 计算指标（batch级）
            batch_metrics = calculate_metrics(final_pred, gt)
            for k, v in batch_metrics.items():
                metrics[k].append(v)

    # 平均指标
    for k in metrics:
        metrics[k] = sum(metrics[k]) / len(metrics[k])
    return metrics


def main():
    # 数据加载
    train_loader = get_dataloader(config['data_root'], 'train', config['batch_size'])
    test_loader = get_dataloader(config['data_root'], 'test', config['batch_size'])

    # 模型、损失、优化器
    model = TCI_Net().to(config['device'])
    criterion = TCI_Loss().to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # 训练循环
    best_fm = 0.0
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config['device'])
        test_metrics = test_epoch(model, test_loader, config['device'])
        scheduler.step()

        # 保存最优模型
        if test_metrics['Fm'] > best_fm:
            best_fm = test_metrics['Fm']
            torch.save(model.state_dict(), config['save_path'])
            print(f'Epoch {epoch + 1}: Best Fm={best_fm:.4f}, saved model')

        # 打印日志
        print(
            f'Epoch {epoch + 1}: Loss={train_loss:.4f}, Sm={test_metrics["Sm"]:.4f}, Fm={test_metrics["Fm"]:.4f}, MAE={test_metrics["MAE"]:.4f}')


if __name__ == '__main__':
    main()