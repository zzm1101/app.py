import numpy as np
import pandas as pd
import os
import glob
import pickle
import logging
import time as time_module
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pywt
from scipy.stats import kurtosis, spearmanr
from scipy.optimize import curve_fit
from config import Config
from filelock import FileLock

logger = logging.getLogger(__name__)

class LSTMVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, num_layers=2, dropout=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True, dropout=dropout)

    def encode(self, x):
        _, (h_n, _) = self.encoder_lstm(x)
        h = h_n[-1]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        h = self.decoder_fc(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder_lstm(h)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        return recon, mu, logvar

def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = nn.MSELoss()(recon, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss

def load_device_data(device_id, cfg):
    logger.info(f"[1/7] 加载设备 {device_id} 的数据...")
    device_dir = os.path.join(cfg.SEAL_DATA_DIR, device_id)
    csv_files = glob.glob(os.path.join(device_dir, "*.csv"))
    if not csv_files:
        logger.warning(f"设备 {device_id} 下没有 CSV 文件")
        return None
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, parse_dates=['timestamp'], index_col='timestamp')
            dfs.append(df)
        except Exception as e:
            logger.error(f"读取文件失败 {file}: {e}")
    if not dfs:
        return None
    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    for col in cfg.SEAL_SENSOR_COLUMNS['numeric']:
        if col not in combined.columns:
            combined[col] = np.nan
    logger.info(f"数据加载完成，共 {len(combined)} 行原始记录")
    return combined

def add_working_condition_label(df, cfg):
    df = df.copy()
    df['working_condition'] = 0
    cip_col = cfg.SEAL_CONDITION_COLUMNS.get('cip_temp', 'temperature')
    empty_col = cfg.SEAL_CONDITION_COLUMNS.get('empty_current', 'current')
    if cip_col in df.columns:
        cip_mask = df[cip_col] > cfg.SEAL_CIP_TEMPERATURE_THRESHOLD
        df.loc[cip_mask, 'working_condition'] = 2
    if empty_col in df.columns:
        empty_mask = df[empty_col] < cfg.SEAL_EMPTY_CURRENT_THRESHOLD
        df.loc[empty_mask, 'working_condition'] = 1
    cip_mask = df['working_condition'] == 2
    cip_end_times = df[cip_mask].index[df[cip_mask].index.to_series().diff() > pd.Timedelta(minutes=10)]
    for end_time in cip_end_times:
        transition_end = end_time + pd.Timedelta(minutes=cfg.SEAL_TRANSITION_DURATION_MIN)
        transition_mask = (df.index > end_time) & (df.index <= transition_end)
        df.loc[transition_mask, 'working_condition'] = 3
    return df

def preprocess_data(df, cfg):
    logger.info("[2/7] 预处理数据（重采样、插值、工况标记）...")
    df = df.resample('1min').mean()
    df = df.interpolate(method='linear', limit=5).dropna()
    df = add_working_condition_label(df, cfg)
    cip_data = df[df['working_condition'] == 2]
    cip_stats = {
        'total_cip_count': len(cip_data) // 120 if len(cip_data) > 0 else 0,
        'avg_cip_temp': cip_data['temperature'].mean() if len(cip_data) > 0 else 0,
        'total_cip_duration_min': len(cip_data),
        'last_cip_time': cip_data.index[-1] if len(cip_data) > 0 else df.index[0]
    }
    df_valid = df[df['working_condition'] == 0].copy()
    logger.info(f"有效数据（正常工况）共 {len(df_valid)} 行")
    return df_valid, df, cip_stats

def wavelet_entropy_func(signal):
    if len(signal) < 10:
        return 0
    wavelet = pywt.Wavelet('db4')
    max_level = pywt.dwt_max_level(len(signal), wavelet.dec_len)
    if max_level < 1:
        return 0
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)
    energy = [np.sum(np.square(c)) for c in coeffs]
    total = sum(energy)
    if total == 0:
        return 0
    prob = [e / total for e in energy]
    return -sum(p * np.log2(p + 1e-10) for p in prob)

def extract_features(df, cfg, cip_stats=None):
    logger.info("[3/7] 提取特征...")
    feature_recipes = cfg.SEAL_FEATURE_RECIPES
    features = pd.DataFrame(index=df.index)
    for col, recipes in feature_recipes.items():
        if col not in df.columns:
            continue
        series = df[col]
        for recipe in recipes:
            if recipe == 'mean':
                features[f'{col}_mean'] = series.rolling(cfg.SEAL_WINDOW_SIZE).mean()
            elif recipe == 'std':
                features[f'{col}_std'] = series.rolling(cfg.SEAL_WINDOW_SIZE).std()
            elif recipe == 'kurtosis':
                features[f'{col}_kurtosis'] = series.rolling(cfg.SEAL_WINDOW_SIZE).apply(kurtosis, raw=False)
            elif recipe == 'gradient':
                features[f'{col}_grad'] = series.rolling(cfg.SEAL_WINDOW_SIZE).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            elif recipe == 'diff_abs':
                features[f'{col}_diff_abs'] = (series - series.rolling(cfg.SEAL_WINDOW_SIZE).mean()).abs()
            elif recipe == 'wavelet_entropy':
                features[f'{col}_wavelet_entropy'] = series.rolling(cfg.SEAL_WINDOW_SIZE).apply(wavelet_entropy_func)
    if cip_stats is not None:
        features['cip_count'] = cip_stats['total_cip_count']
        features['cip_avg_temp'] = cip_stats['avg_cip_temp']
        features['cip_total_duration'] = cip_stats['total_cip_duration_min']
        features['hours_since_last_cip'] = (df.index - cip_stats['last_cip_time']).total_seconds() / 3600
    features = features.dropna()
    logger.info(f"特征提取完成，共 {features.shape[1]} 维特征，{len(features)} 个时间点")
    return features

def train_or_load_vae(features, device_id, cfg, epoch_callback=None):
    model_dir = os.path.join(cfg.SEAL_MODEL_DIR, device_id)
    os.makedirs(model_dir, exist_ok=True)
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    model_path = os.path.join(model_dir, 'vae.pth')
    stats_path = os.path.join(model_dir, 'stats.pkl')
    meta_path = os.path.join(model_dir, 'metadata.pkl')
    lock_path = os.path.join(model_dir, '.lock')

    with FileLock(lock_path, timeout=30):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features.values)
        seq_len = cfg.SEAL_VAE_SEQ_LEN
        if len(scaled) < seq_len + 10:
            raise ValueError(f"数据量不足，需要至少 {seq_len+10} 个点，当前 {len(scaled)}")
        X = np.array([scaled[i:i+seq_len] for i in range(len(scaled)-seq_len+1)])
        input_dim = features.shape[1]

        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            if meta['input_dim'] != input_dim:
                logger.warning(f"特征维度从 {meta['input_dim']} 变为 {input_dim}，清理旧模型文件")
                for f in glob.glob(os.path.join(model_dir, '*')):
                    if os.path.basename(f) != '.lock':
                        os.remove(f)
                return train_or_load_vae(features, device_id, cfg, epoch_callback)

        if os.path.exists(model_path) and os.path.exists(stats_path):
            model_mtime = os.path.getmtime(model_path)
            if time_module.time() - model_mtime > 7 * 24 * 3600:
                logger.info("模型已过期（>7天），将重新训练...")
            else:
                with open(stats_path, 'rb') as f:
                    saved = pickle.load(f)
                if len(X) <= saved['train_size'] * 1.2:
                    logger.info("[4/7] 发现已有模型，直接加载...")
                    if epoch_callback:
                        epoch_callback(1, 1)
                    scaler = pickle.load(open(scaler_path, 'rb'))
                    model = LSTMVAE(input_dim, cfg.SEAL_VAE_HIDDEN_DIM, cfg.SEAL_VAE_LATENT_DIM)
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    model.eval()
                    logger.info("模型加载完成")
                    return model, scaler, saved['train_mean'], saved['train_std']

        logger.info(f"[4/7] 未发现可用模型，开始训练 LSTM-VAE（共 {cfg.SEAL_VAE_EPOCHS} 轮）...")
        train_len = int(0.8 * len(X))
        X_train = X[:train_len]
        dataset = TensorDataset(torch.FloatTensor(X_train))
        loader = DataLoader(dataset, batch_size=cfg.SEAL_VAE_BATCH_SIZE, shuffle=True)
        model = LSTMVAE(input_dim, cfg.SEAL_VAE_HIDDEN_DIM, cfg.SEAL_VAE_LATENT_DIM)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for epoch in range(cfg.SEAL_VAE_EPOCHS):
            total_loss = 0
            for batch in loader:
                x = batch[0]
                optimizer.zero_grad()
                recon, mu, logvar = model(x)
                loss = vae_loss(recon, x, mu, logvar)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"Epoch {epoch+1}/{cfg.SEAL_VAE_EPOCHS}, Loss: {total_loss/len(loader):.4f}")
            if epoch_callback:
                epoch_callback(epoch + 1, cfg.SEAL_VAE_EPOCHS)

        model.eval()
        train_errors = []
        with torch.no_grad():
            for i in range(0, train_len, cfg.SEAL_VAE_BATCH_SIZE):
                x = torch.FloatTensor(X_train[i:i+cfg.SEAL_VAE_BATCH_SIZE])
                recon, _, _ = model(x)
                mse = torch.mean((recon - x)**2, dim=(1,2)).numpy()
                train_errors.extend(mse)
        train_mean = np.mean(train_errors)
        train_std = np.std(train_errors)
        logger.info(f"训练完成，重构误差均值: {train_mean:.4f}, 标准差: {train_std:.4f}")
        pickle.dump(scaler, open(scaler_path, 'wb'))
        torch.save(model.state_dict(), model_path)
        with open(stats_path, 'wb') as f:
            pickle.dump({'train_size': train_len, 'train_mean': train_mean, 'train_std': train_std}, f)
        with open(meta_path, 'wb') as f:
            pickle.dump({'input_dim': input_dim}, f)
        logger.info("模型已保存到磁盘")
        return model, scaler, train_mean, train_std

def compute_health_scores(model, scaler, features, train_mean, train_std, cfg):
    logger.info("[5/7] 计算健康分数及特征贡献...")
    seq_len = cfg.SEAL_VAE_SEQ_LEN
    scaled = scaler.transform(features.values)
    X = np.array([scaled[i:i+seq_len] for i in range(len(scaled)-seq_len+1)])
    model.eval()
    errors = []
    feature_contributions = []
    with torch.no_grad():
        for i in range(0, len(X), cfg.SEAL_VAE_BATCH_SIZE):
            x = torch.FloatTensor(X[i:i+cfg.SEAL_VAE_BATCH_SIZE])
            recon, _, _ = model(x)
            mse = torch.mean((recon - x)**2, dim=(1,2)).numpy()
            errors.extend(mse)
            per_feature_mse = torch.mean((recon - x)**2, dim=1).numpy()
            feature_contributions.extend(per_feature_mse)
    full_errors = np.full(len(features), np.mean(errors))
    full_errors[seq_len-1:] = errors
    full_feature_contrib = np.full((len(features), features.shape[1]), np.nan)
    full_feature_contrib[seq_len-1:] = np.array(feature_contributions)
    z = (full_errors - train_mean) / (train_std + 1e-8)
    health = 1 / (1 + np.exp(2 * (z - 2.0)))
    health = np.clip(health, 0, 1)
    logger.info(f"健康分数范围: {health.min():.2f} ~ {health.max():.2f}")
    return full_errors, health, full_feature_contrib

def predict_rul(health_seq):
    logger.info("[6/7] 预测剩余使用寿命 (RUL)...")
    x = np.arange(len(health_seq))
    y = health_seq
    if len(y) < 50:
        logger.warning("测试数据不足 50 个点，无法预测 RUL")
        return None, False
    rho, _ = spearmanr(x, y)
    if rho > -0.3:
        logger.warning(f"健康序列无明显下降趋势 (Spearman ρ={rho:.2f})，无法可靠预测 RUL")
        return None, False
    try:
        def exp_decay(t, a, b): return a * np.exp(b * t)
        popt, _ = curve_fit(exp_decay, x, y, p0=(1, -0.01), maxfev=1000)
        a, b = popt
        if b >= 0:
            raise ValueError
        x_fail = (np.log(Config.SEAL_HEALTH_ALARM) - np.log(a)) / b
        rul_points = x_fail - len(x)
        rul_days = max(0, rul_points) / 1440
        logger.info(f"RUL 预测 (指数拟合): {rul_days:.1f} 天")
        return rul_days, True
    except:
        coef = np.polyfit(x, y, 1)
        if coef[0] >= 0:
            return None, False
        x_fail = (Config.SEAL_HEALTH_ALARM - coef[1]) / coef[0]
        rul_days = max(0, x_fail - len(x)) / 1440
        logger.info(f"RUL 预测 (线性拟合): {rul_days:.1f} 天")
        return rul_days, True

def train_and_predict(device_id, config_overrides=None, progress_callback=None):
    try:
        cfg = Config.get_with_overrides(config_overrides or {})
        logger.info(f"========== 开始分析设备 {device_id} ==========")

        def report(step, total=7, message=""):
            if progress_callback:
                progress_callback(step, total, message)

        report(1, 7, "加载数据")
        df = load_device_data(device_id, cfg)
        if df is None or len(df) < 200:
            logger.error("数据不足（少于200行），分析终止")
            return {"success": False, "message": "Insufficient data"}

        report(2, 7, "预处理数据")
        df_valid, df_full, cip_stats = preprocess_data(df, cfg)
        if len(df_valid) < 200:
            logger.error("有效数据不足（少于200行），分析终止")
            return {"success": False, "message": "Insufficient valid data"}

        report(3, 7, "提取特征")
        features = extract_features(df_valid, cfg, cip_stats)

        def training_progress(epoch, total_epochs):
            report(4, 7, f"训练模型 (epoch {epoch}/{total_epochs})")

        model, scaler, train_mean, train_std = train_or_load_vae(
            features, device_id, cfg, epoch_callback=training_progress
        )

        report(5, 7, "计算健康分数")
        errors, health, feature_contrib = compute_health_scores(model, scaler, features, train_mean, train_std, cfg)
        train_len = int(0.8 * len(health))
        test_health = health[train_len:]
        test_errors = errors[train_len:]
        test_dates = features.index[train_len:].strftime('%Y-%m-%d %H:%M:%S').tolist()
        test_feature_contrib = feature_contrib[train_len:]

        report(6, 7, "预测剩余寿命")
        rul_days, rul_confidence = predict_rul(test_health)

        report(7, 7, "生成报告")
        if not rul_confidence:
            rul_days = "Uncertain"

        anomalies = []
        for i, (date, h, e) in enumerate(zip(test_dates, test_health, test_errors)):
            if h < cfg.SEAL_HEALTH_WARNING:
                anomalies.append({
                    "date": date,
                    "health": round(float(h), 2),
                    "score": round(float(e), 4)
                })

        feature_names = features.columns.tolist()
        contrib_summary = []
        for i in range(len(test_dates)):
            contrib_dict = {feature_names[j]: float(test_feature_contrib[i][j]) for j in range(len(feature_names))}
            sorted_contrib = sorted(contrib_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            contrib_summary.append({k: v for k, v in sorted_contrib})

        logger.info(f"[7/7] 分析完成！平均健康度: {np.mean(test_health):.2f}, 异常点: {len(anomalies)} 个")
        logger.info("=" * 50)
        return {
            "success": True,
            "device_id": device_id,
            "dates": test_dates,
            "health_scores": [round(float(h), 2) for h in test_health],
            "vae_scores": [round(float(e), 4) for e in test_errors],
            "threshold": cfg.SEAL_HEALTH_WARNING,
            "avg_health": round(float(np.mean(test_health)), 2),
            "rul": round(rul_days, 1) if isinstance(rul_days, float) else rul_days,
            "rul_confidence": rul_confidence,
            "anomalies": anomalies[:10],
            "anomaly_count": len(anomalies),
            "feature_contributions": contrib_summary,
            "feature_names": feature_names
        }
    except Exception as e:
        logger.exception(f"分析失败: {e}")
        return {"success": False, "message": str(e)}