import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from sediment_hotspot.config import MODEL, SEDIMENT
from sediment_hotspot.evaluate import evaluate_outputs
from sediment_hotspot.models.cnn import SpatialCNN
from sediment_hotspot.models.fusion import SedimentHotspotFusionNet
from sediment_hotspot.models.lstm import FlowLSTM
from sediment_hotspot.losses.counterfactual_loss import CounterfactualConsistencyLoss
from sediment_hotspot.training.dataset import SedimentHotspotDataset, split_dataset_manifest
from sediment_hotspot.utils.io import ensure_dir


class CNNOnlyBaseline(nn.Module):
    def __init__(self, image_size=(128, 128)):
        super().__init__()
        h = MODEL.hidden_dim
        self.satellite_cnn = SpatialCNN(MODEL.image_channels, h)
        self.terrain_cnn = SpatialCNN(MODEL.terrain_channels, h)
        self.landuse_cnn = SpatialCNN(MODEL.landuse_channels, h)
        self.head = nn.Sequential(nn.Linear(h * 3, 256), nn.ReLU(), nn.Linear(256, image_size[0] * image_size[1]), nn.Sigmoid())
        self.zone_head = nn.Sequential(nn.Linear(h * 3, 128), nn.ReLU(), nn.Linear(128, MODEL.zones), nn.Sigmoid())
        self.image_size = image_size

    def forward(self, batch):
        s = self.satellite_cnn(batch["satellite"])
        t = self.terrain_cnn(batch["terrain"])
        l = self.landuse_cnn(batch["landuse"])
        fused = torch.cat([s, t, l], dim=-1)
        return {"heatmap": self.head(fused).view(-1, 1, *self.image_size), "zone_risk": self.zone_head(fused)}


class BiLSTMOnlyBaseline(nn.Module):
    def __init__(self, image_size=(128, 128)):
        super().__init__()
        h = MODEL.hidden_dim
        self.flow_lstm = FlowLSTM(MODEL.flow_features, h)
        self.rain_encoder = nn.Sequential(nn.Linear(MODEL.rain_features, h), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(h * 2, 256), nn.ReLU(), nn.Linear(256, image_size[0] * image_size[1]), nn.Sigmoid())
        self.zone_head = nn.Sequential(nn.Linear(h * 2, 128), nn.ReLU(), nn.Linear(128, MODEL.zones), nn.Sigmoid())
        self.image_size = image_size

    def forward(self, batch):
        fl = self.flow_lstm(batch["flow_sequence"])
        rn = self.rain_encoder(batch["rain_features"])
        fused = torch.cat([fl, rn], dim=-1)
        return {"heatmap": self.head(fused).view(-1, 1, *self.image_size), "zone_risk": self.zone_head(fused)}


class CNNBiLSTMBaseline(nn.Module):
    def __init__(self, image_size=(128, 128)):
        super().__init__()
        h = MODEL.hidden_dim
        self.satellite_cnn = SpatialCNN(MODEL.image_channels, h)
        self.terrain_cnn = SpatialCNN(MODEL.terrain_channels, h)
        self.landuse_cnn = SpatialCNN(MODEL.landuse_channels, h)
        self.flow_lstm = FlowLSTM(MODEL.flow_features, h)
        self.rain_encoder = nn.Sequential(nn.Linear(MODEL.rain_features, h), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(h * 5, 256), nn.ReLU(), nn.Linear(256, image_size[0] * image_size[1]), nn.Sigmoid())
        self.zone_head = nn.Sequential(nn.Linear(h * 5, 128), nn.ReLU(), nn.Linear(128, MODEL.zones), nn.Sigmoid())
        self.image_size = image_size

    def forward(self, batch):
        s = self.satellite_cnn(batch["satellite"])
        t = self.terrain_cnn(batch["terrain"])
        l = self.landuse_cnn(batch["landuse"])
        fl = self.flow_lstm(batch["flow_sequence"])
        rn = self.rain_encoder(batch["rain_features"])
        fused = torch.cat([s, t, l, fl, rn], dim=-1)
        return {"heatmap": self.head(fused).view(-1, 1, *self.image_size), "zone_risk": self.zone_head(fused)}


class StandardAttentionBaseline(nn.Module):
    def __init__(self, image_size=(128, 128)):
        super().__init__()
        h = MODEL.hidden_dim
        self.satellite_cnn = SpatialCNN(MODEL.image_channels, h)
        self.terrain_cnn = SpatialCNN(MODEL.terrain_channels, h)
        self.landuse_cnn = SpatialCNN(MODEL.landuse_channels, h)
        self.flow_lstm = FlowLSTM(MODEL.flow_features, h)
        self.rain_encoder = nn.Sequential(nn.Linear(MODEL.rain_features, h), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(h, h // 2), nn.Tanh(), nn.Linear(h // 2, 1))
        self.head = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Linear(256, image_size[0] * image_size[1]), nn.Sigmoid())
        self.zone_head = nn.Sequential(nn.Linear(h, 128), nn.ReLU(), nn.Linear(128, MODEL.zones), nn.Sigmoid())
        self.image_size = image_size

    def forward(self, batch):
        reps = [
            self.satellite_cnn(batch["satellite"]),
            self.terrain_cnn(batch["terrain"]),
            self.landuse_cnn(batch["landuse"]),
            self.flow_lstm(batch["flow_sequence"]),
            self.rain_encoder(batch["rain_features"]),
        ]
        states = torch.stack(reps, dim=1)
        attn = torch.softmax(self.score(states).squeeze(-1), dim=-1)
        fused = (states * attn.unsqueeze(-1)).sum(dim=1)
        return {"heatmap": self.head(fused).view(-1, 1, *self.image_size), "zone_risk": self.zone_head(fused)}


def extract_flat_tabular(df):
    X, y_heat, y_zone = [], [], []
    for _, row in df.iterrows():
        s = np.load(row.satellite_stack).mean(axis=(1, 2))
        t = np.load(row.terrain_stack).mean(axis=(1, 2))
        l = np.load(row.landuse_stack).mean(axis=(1, 2))
        fl = np.load(row.flow_sequence).mean(axis=0)
        rn = np.load(row.rain_features)
        feat = np.concatenate([s, t, l, fl, rn])
        X.append(feat)
        hm = np.load(row.target_heatmap_adaptive if "target_heatmap_adaptive" in row and Path(str(row.target_heatmap_adaptive)).exists() else row.target_heatmap)
        y_heat.append(hm.mean())
        zr = np.load(row.zone_risk)
        y_zone.append(zr)
    return np.array(X), np.array(y_heat), np.array(y_zone)


def evaluate_torch_model(model, loader, device="cpu"):
    model.eval()
    all_pred_heat, all_true_heat = [], []
    all_pred_zone, all_true_zone = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)
            all_pred_heat.append(out["heatmap"].cpu().numpy())
            all_true_heat.append(batch["target_heatmap"].cpu().numpy())
            all_pred_zone.append(out["zone_risk"].cpu().numpy())
            all_true_zone.append(batch["zone_risk"].cpu().numpy())

    pred_h = np.concatenate(all_pred_heat, axis=0)
    true_h = np.concatenate(all_true_heat, axis=0)
    pred_z = np.concatenate(all_pred_zone, axis=0)
    true_z = np.concatenate(all_true_zone, axis=0)
    return evaluate_outputs(true_h, pred_h, zone_true=true_z, zone_pred=pred_z)


def run_all_baselines(manifest_csv: str, output_csv: str = "results/baseline_results.csv", epochs: int = 10, batch_size: int = 4):
    ensure_dir(Path(output_csv).parent)
    df = pd.read_csv(manifest_csv)
    train_df, val_df, test_df = split_dataset_manifest(df)

    test_ds = SedimentHotspotDataset(test_df)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    train_ds = SedimentHotspotDataset(train_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    results = []

    # 1. Random Forest
    print("Training 1/8: Random Forest...")
    X_train, y_h_train, y_z_train = extract_flat_tabular(train_df)
    X_test, y_h_test, y_z_test = extract_flat_tabular(test_df)
    rf_heat = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train, y_h_train)
    rf_zone = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train, y_z_train)
    rf_p_h = rf_heat.predict(X_test)
    rf_p_z = rf_zone.predict(X_test)
    rf_metrics = evaluate_outputs(y_h_test, rf_p_h, zone_true=y_z_test, zone_pred=rf_p_z)
    rf_metrics["model"] = "Random Forest"
    results.append(rf_metrics)

    # 2. XGBoost
    print("Training 2/8: XGBoost...")
    from sklearn.multioutput import MultiOutputRegressor
    xgb_heat = xgb.XGBRegressor(n_estimators=50, random_state=42).fit(X_train, y_h_train)
    xgb_zone = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=50, random_state=42)).fit(X_train, y_z_train)
    xgb_p_h = xgb_heat.predict(X_test)
    xgb_p_z = xgb_zone.predict(X_test)
    xgb_metrics = evaluate_outputs(y_h_test, xgb_p_h, zone_true=y_z_test, zone_pred=xgb_p_z)
    xgb_metrics["model"] = "XGBoost"
    results.append(xgb_metrics)

    # Deep learning baselines
    torch_baselines = [
        ("CNN-only", CNNOnlyBaseline(), False),
        ("BiLSTM-only", BiLSTMOnlyBaseline(), False),
        ("CNN + BiLSTM", CNNBiLSTMBaseline(), False),
        ("Standard Attention Fusion", StandardAttentionBaseline(), False),
        ("Proposed Multimodal Model", SedimentHotspotFusionNet(config=replace(SEDIMENT, use_counterfactual_loss=False)), False),
        ("Full Model + CSDN", SedimentHotspotFusionNet(config=SEDIMENT), True),
    ]

    cf_loss_fn = CounterfactualConsistencyLoss(tolerance=SEDIMENT.counterfactual_tolerance)

    for idx, (name, model, use_cf) in enumerate(torch_baselines, start=3):
        print(f"Training {idx}/8: {name}...")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        bce = nn.BCELoss()
        mse = nn.MSELoss()
        for ep in range(epochs):
            model.train()
            for b in train_loader:
                out = model(b)
                loss = bce(out["heatmap"], b["target_heatmap"]) + mse(out["zone_risk"], b["zone_risk"])
                if "contrastive_loss" in out and out["contrastive_loss"] is not None:
                    loss += 0.05 * out["contrastive_loss"]
                if use_cf:
                    l_cf, _ = cf_loss_fn(model, b, out)
                    loss += 0.10 * l_cf
                opt.zero_grad()
                loss.backward()
                opt.step()
        m_metrics = evaluate_torch_model(model, test_loader)
        m_metrics["model"] = name
        results.append(m_metrics)

    res_df = pd.DataFrame(results)
    res_df.to_csv(output_csv, index=False)
    print(f"Saved baseline results to {output_csv}")
    return res_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="results/baseline_results.csv")
    p.add_argument("--epochs", type=int, default=10)
    a = p.parse_args()
    run_all_baselines(a.manifest, a.output, a.epochs)
