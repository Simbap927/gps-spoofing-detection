import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class GPSSpoofingDataset(Dataset):
    """GPS Spoofing Detection Dataset"""

    def __init__(self, X, y):
        """
        Args:
            X: numpy array, shape [n_samples, n_timesteps, n_features]
            y: numpy array, shape [n_samples]
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_processed_data(data_dir='data/processed'):
    """전처리된 npy 파일 로드"""
    data_path = Path(data_dir)

    X_train = np.load(data_path / 'X_train.npy')
    y_train = np.load(data_path / 'y_train.npy')
    X_val = np.load(data_path / 'X_val.npy')
    y_val = np.load(data_path / 'y_val.npy')
    X_test = np.load(data_path / 'X_test.npy')
    y_test = np.load(data_path / 'y_test.npy')

    print(f"Loaded from {data_dir}/")
    print(f"  - Train: {X_train.shape[0]:,} samples")
    print(f"  - Val:   {X_val.shape[0]:,} samples")
    print(f"  - Test:  {X_test.shape[0]:,} samples")

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                      batch_size=32, num_workers=4, shuffle_train=True):
    """PyTorch DataLoader 생성"""

    # Dataset 생성
    train_dataset = GPSSpoofingDataset(X_train, y_train)
    val_dataset = GPSSpoofingDataset(X_val, y_val)
    test_dataset = GPSSpoofingDataset(X_test, y_test)

    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nDataLoaders created:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches:   {len(val_loader)}")
    print(f"  - Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader


def get_dataloaders(data_dir='data/processed', batch_size=32, num_workers=4):
    """전처리된 데이터 로드 및 DataLoader 반환 (간편 함수)"""
    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_data(data_dir)
    return create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test,
                            batch_size, num_workers)


if __name__ == '__main__':
    # 테스트
    print("=" * 60)
    print("Testing DataLoader")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    # 첫 배치 확인
    X_batch, y_batch = next(iter(train_loader))
    print(f"\nFirst batch:")
    print(f"  - X shape: {X_batch.shape}")  # [batch_size, n_timesteps, n_features]
    print(f"  - y shape: {y_batch.shape}")  # [batch_size]
    print(f"  - X dtype: {X_batch.dtype}")
    print(f"  - y dtype: {y_batch.dtype}")
    print(f"  - y values: {y_batch[:10].tolist()}")

    print("\n" + "=" * 60)
    print("DataLoader test completed!")
    print("=" * 60)
