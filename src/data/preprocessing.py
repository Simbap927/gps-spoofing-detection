import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import json
import warnings
warnings.filterwarnings('ignore')


# ===== 1. 데이터 로드 =====
def load_datasets(base_path='data/raw'):
    """3개 AV-GPS-Dataset 병합"""
    dfs = []
    for i in [1, 2, 3]:
        df = pd.read_csv(f"{base_path}/AV-GPS-Dataset-{i}.csv")
        print(f"  Loaded Dataset-{i}: {len(df):,} samples")
        dfs.append(df)

    df_merged = pd.concat(dfs, ignore_index=True)
    print(f"Total merged: {len(df_merged):,} samples\n")
    return df_merged


# ===== 2. 결측치 처리 =====
def handle_missing_values(df):
    """선형 보간 후 남은 결측치가 있는 행 제거"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing_before = df[numeric_cols].isnull().sum().sum()
    before = len(df)

    # 선형 보간
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')

    # 남은 결측치가 있는 행 제거
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    removed = before - len(df)
    missing_after = df[numeric_cols].isnull().sum().sum()
    print(f"  Missing values: {missing_before:,} → {missing_after:,}")
    print(f"  Removed {removed:,} rows with remaining NaN\n")
    return df


# ===== 3. GPS 거리 특징 엔지니어링 =====
def haversine_distance(lat1, lon1, lat2, lon2):
    """GPS 좌표 간 거리 계산 (미터)"""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def engineer_features(df):
    """GPS 좌표 간 거리 특징 생성 (벡터화)"""
    # 벡터화된 GPS 거리 계산
    lat1 = df['GPS Latitude'].shift(1).values
    lon1 = df['GPS Longitude'].shift(1).values
    lat2 = df['GPS Latitude'].values
    lon2 = df['GPS Longitude'].values

    # 첫 번째 행은 NaN이 되므로 0으로 설정
    lat1[0], lon1[0] = lat2[0], lon2[0]

    df['GPS_distance'] = haversine_distance(lat1, lon1, lat2, lon2)
    df.loc[0, 'GPS_distance'] = 0.0  # 첫 샘플은 0

    print(f"  Engineered GPS_distance feature (vectorized)\n")
    return df


# ===== 4. 특징 선택 =====
def select_features(df):
    """모델 입력 특징 선택 (GPS 위경도 제외)"""
    # 제외할 컬럼
    exclude = [
        # GPS 좌표: GPS_distance로 변환해서 사용 (절대 위치보다 변화량이 spoofing 탐지에 유용)
        'GPS Latitude', 'GPS Longitude', 'GPS MGRS',
        # 시간 정보: 시계열 순서는 윈도우 구조로 표현됨
        'Run Time', 'Clock Time', 'Clock Date', 'Hobbs',
        'Data Type'  # 타겟 변수
    ]

    # 수치형 특징만 선택
    # 포함되는 특징: 자세(Roll, Pitch, Yaw, Heading), 운동(Velocity, Steering Angle),
    #               GPS(Course, HDOP, VDOP, Satellite Count), 센서(Vibration, Temperature),
    #               파생 특징(GPS_distance)
    features = [col for col in df.columns
                if col not in exclude and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]

    print(f"  Selected {len(features)} features")
    print(f"  Features: {features[:5]}... (showing first 5)\n")
    return features


# ===== 5. 윈도우 생성 (10초, non-overlapping) =====
def create_windows(df, features, window_size=10, stride=10):
    """10초 윈도우 생성 (레이블 혼재 제외)"""
    windows, labels = [], []

    for i in range(0, len(df) - window_size + 1, stride):
        window = df.iloc[i:i+window_size]
        window_labels = window['Data Type'].values

        # 레이블이 단일한 경우만 포함
        if len(np.unique(window_labels)) == 1:
            windows.append(window[features].values)
            labels.append(window_labels[0])

    X = np.array(windows, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    print(f"  Created {len(X):,} windows (window_size={window_size}, stride={stride})")
    print(f"  Normal: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
    print(f"  Spoofed: {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)\n")

    return X, y


# ===== 6. Train/Val/Test 분할 (7:2:1, Stratified Shuffle) =====
def split_dataset(X, y):
    """Stratified shuffle split (7:2:1)"""
    # Train 70%, Temp 30%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Val 20%, Test 10% (from temp 30%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.333, stratify=y_temp, random_state=42
    )

    print(f"  Train: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%) - Normal: {(y_train==0).sum():,}, Spoofed: {(y_train==1).sum():,}")
    print(f"  Val:   {len(X_val):,} ({len(X_val)/len(X)*100:.1f}%) - Normal: {(y_val==0).sum():,}, Spoofed: {(y_val==1).sum():,}")
    print(f"  Test:  {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%) - Normal: {(y_test==0).sum():,}, Spoofed: {(y_test==1).sum():,}\n")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ===== 7. 정규화 =====
def normalize_features(X_train, X_val, X_test):
    """StandardScaler로 정규화"""
    n_samples, n_timesteps, n_features = X_train.shape

    # Reshape to [n_samples * n_timesteps, n_features]
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)

    # Fit on train, transform all
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_flat).reshape(n_samples, n_timesteps, n_features)
    X_val_norm = scaler.transform(X_val_flat).reshape(len(X_val), n_timesteps, n_features)
    X_test_norm = scaler.transform(X_test_flat).reshape(len(X_test), n_timesteps, n_features)

    print(f"  Normalized with StandardScaler\n")
    return X_train_norm, X_val_norm, X_test_norm, scaler


# ===== 8. 저장 =====
def save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, scaler, features, output_dir='data/processed'):
    """npy 파일로 저장"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(output_path / 'X_train.npy', X_train)
    np.save(output_path / 'y_train.npy', y_train)
    np.save(output_path / 'X_val.npy', X_val)
    np.save(output_path / 'y_val.npy', y_val)
    np.save(output_path / 'X_test.npy', X_test)
    np.save(output_path / 'y_test.npy', y_test)

    with open(output_path / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # 메타데이터
    metadata = {
        'features': features,
        'n_features': len(features),
        'window_size': 10,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'label_mapping': {0: 'Normal', 1: 'Spoofed'}
    }
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved to: {output_dir}/")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_val:   {X_val.shape}")
    print(f"  - X_test:  {X_test.shape}")
    print(f"  - scaler.pkl, metadata.json\n")


# ===== 메인 파이프라인 =====
def main():
    print("=" * 70)
    print("GPS Spoofing Detection - Preprocessing Pipeline")
    print("=" * 70)

    print("\n[1/8] Loading datasets...")
    df = load_datasets()

    print("[2/8] Handling missing values...")
    df = handle_missing_values(df)

    print("[3/8] Engineering GPS distance feature...")
    df = engineer_features(df)

    print("[4/8] Selecting features...")
    features = select_features(df)

    print("[5/8] Creating windows...")
    X, y = create_windows(df, features, window_size=10, stride=10)

    print("[6/8] Splitting dataset (stratified shuffle)...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)

    print("[7/8] Normalizing features...")
    X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)

    print("[8/8] Saving processed data...")
    save_processed_data(X_train, y_train, X_val, y_val, X_test, y_test, scaler, features)

    print("=" * 70)
    print("✅ Preprocessing completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
