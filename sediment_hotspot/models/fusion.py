import torch
from torch import nn

from sediment_hotspot.config import MODEL, SEDIMENT
from sediment_hotspot.models.causal_attention import CausalAttentionFusion
from sediment_hotspot.models.cnn import SpatialCNN
from sediment_hotspot.models.contrastive import ModalContrastiveAlignment
from sediment_hotspot.models.dsdi import DynamicSedimentDepositionIndex, DSDIFeatureEncoder
from sediment_hotspot.models.gnn import WatershedReservoirGNN
from sediment_hotspot.models.lstm import FlowLSTM


class SpatialDecoder(nn.Module):
    """Deep spatial decoder with progressive bilinear upsampling, convolutions,
    and multi-modal spatial feature fusion.
    Preserves 2D spatial coherence and continuity for sediment hotspot risk prediction.
    """

    def __init__(self, in_features: int = 128, out_channels: int = 1, image_size: tuple[int, int] = (128, 128)):
        super().__init__()
        self.image_size = image_size
        self.proj = nn.Sequential(
            nn.Linear(in_features, 128 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # Progressive upsampling: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        self.up1 = block(128, 64)  # 8x8
        self.up2 = block(64, 48)   # 16x16
        self.up3 = block(48, 32)   # 32x32
        self.up4 = block(32, 24)   # 64x64
        self.up5 = block(24, 16)   # 128x128

        # Spatial skip-fusion from 32x32 CNN features with FiLM modulation
        self.spatial_fuse = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.gamma_proj = nn.Linear(in_features, 32)
        self.beta_proj = nn.Linear(in_features, 32)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

        self.out_conv = nn.Sequential(
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        spatial_context: torch.Tensor | None = None,
        dsdi_prior_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = x.shape[0]
        if spatial_context is not None:
            # Multi-modal spatial fusion: start from preserved 32x32 spatial map
            ctx32 = self.spatial_fuse(spatial_context)
            gamma = self.gamma_proj(x).view(b, 32, 1, 1)
            beta = self.beta_proj(x).view(b, 32, 1, 1)
            h = ctx32 * (1.0 + gamma) + beta
            h = self.up4(h)
            h = self.up5(h)
        else:
            # Fallback path if spatial context is not supplied
            h = self.proj(x).view(b, 128, 4, 4)
            h = self.up1(h)
            h = self.up2(h)
            h = self.up3(h)
            h = self.up4(h)
            h = self.up5(h)

        raw_map = self.out_conv(h)
        if raw_map.shape[-2:] != self.image_size:
            raw_map = nn.functional.interpolate(raw_map, size=self.image_size, mode="bilinear", align_corners=False)

        if dsdi_prior_map is not None:
            # Blend deep learned spatial residual with environmental DSDI prior
            if dsdi_prior_map.shape[-2:] != self.image_size:
                dsdi_prior_map = nn.functional.interpolate(dsdi_prior_map, size=self.image_size, mode="bilinear", align_corners=False)
            out = torch.clamp(0.6 * raw_map + 0.4 * dsdi_prior_map, 0.0, 1.0)
        else:
            out = raw_map
        return out


class SedimentHotspotFusionNet(nn.Module):
    """End-to-End Sediment Hotspot Prediction Network.
    Integrates DSDI + WR-GNN + Causal Attention Fusion + Multi-Modal Contrastive Alignment + Spatial Decoder.
    """

    def __init__(self, image_size: tuple[int, int] = (128, 128), config=SEDIMENT):
        super().__init__()
        self.config = config
        self.image_size = image_size
        h = MODEL.hidden_dim

        self.satellite_cnn = SpatialCNN(MODEL.image_channels, h)
        self.terrain_cnn = SpatialCNN(MODEL.terrain_channels, h)
        self.landuse_cnn = SpatialCNN(MODEL.landuse_channels, h)
        self.flow_lstm = FlowLSTM(MODEL.flow_features, h)
        self.rain_encoder = nn.Sequential(
            nn.Linear(MODEL.rain_features, h),
            nn.ReLU(),
        )

        self.dsdi = DynamicSedimentDepositionIndex(MODEL.dsdi_features)
        self.dsdi_encoder = DSDIFeatureEncoder(MODEL.dsdi_features, h)
        self.gnn = WatershedReservoirGNN(MODEL.graph_features, h, layers=config.graph_layers)

        self.fusion = CausalAttentionFusion(h, modality_names=MODEL.modality_names, temperature=config.causal_temperature)
        self.contrastive = ModalContrastiveAlignment(h, projection=MODEL.projection_dim)
        self.dropout = nn.Dropout(MODEL.dropout)

        self.heatmap_head = SpatialDecoder(in_features=h, out_channels=1, image_size=image_size)
        self.zone_head = nn.Sequential(
            nn.Linear(h, 128),
            nn.ReLU(),
            nn.Linear(128, MODEL.zones),
            nn.Sigmoid(),
        )
        self.dsdi_head = nn.Sequential(
            nn.Linear(h, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = batch["satellite"].device
        batch_size = batch["satellite"].shape[0]

        satellite_rep = self.satellite_cnn(batch["satellite"])
        terrain_rep = self.terrain_cnn(batch["terrain"])
        landuse_rep = self.landuse_cnn(batch["landuse"])
        flow_rep = self.flow_lstm(batch["flow_sequence"])
        rain_rep = self.rain_encoder(batch["rain_features"])

        z = batch.get("dsdi_features", torch.zeros((batch_size, MODEL.dsdi_features), device=device))
        dsdi_map_input = batch.get("dsdi_feature_map", None)
        dsdi_out = self.dsdi(z, feature_map=dsdi_map_input)
        dsdi_rep = self.dsdi_encoder(z, dsdi_out)

        graph_nodes = batch.get("graph_nodes", torch.zeros((batch_size, 5, MODEL.graph_features), device=device))
        graph_adj = batch.get("graph_adjacency", None)
        graph_rep, _ = self.gnn(graph_nodes, graph_adj)

        representations = {
            "satellite": satellite_rep,
            "flow": flow_rep,
            "terrain": terrain_rep,
            "landuse": landuse_rep,
            "rain": rain_rep,
            "graph": graph_rep,
            "dsdi": dsdi_rep,
        }

        fused, weights = self.fusion(representations, dsdi_prior=dsdi_out)
        contrastive_res = self.contrastive(representations)

        fused_drop = self.dropout(fused)

        # Multi-modal spatial context from CNN feature extractors
        sat_feat = self.satellite_cnn.extract_features(batch["satellite"])
        terr_feat = self.terrain_cnn.extract_features(batch["terrain"])
        land_feat = self.landuse_cnn.extract_features(batch["landuse"])

        # Modulate spatial feature maps by learned causal attention weights
        w_sat = weights["final"][:, 0:1, None, None]
        w_terr = weights["final"][:, 2:3, None, None]
        w_land = weights["final"][:, 3:4, None, None]
        spatial_context = w_sat * sat_feat + w_terr * terr_feat + w_land * land_feat

        heatmap = self.heatmap_head(
            fused_drop,
            spatial_context=spatial_context,
            dsdi_prior_map=dsdi_out.get("map", None),
        ).view(-1, 1, *self.image_size)
        zone_risk = self.zone_head(fused_drop)
        dsdi_pred = self.dsdi_head(fused_drop)

        return {
            "heatmap": heatmap,
            "zone_risk": zone_risk,
            "dsdi": dsdi_out["index"],
            "dsdi_map": dsdi_out["map"],
            "dsdi_prediction": dsdi_pred,
            "dsdi_feature_weights": dsdi_out["feature_weights"],
            "attention": weights["final"],
            "learned_attention": weights["learned"],
            "causal_attention": weights["causal"],
            "prior_attention": weights["prior"],
            "fusion_weight": weights["final"],
            "contrastive_loss": contrastive_res["loss"],
            "contrastive_embeddings": contrastive_res.get("embeddings", {}),
            "representations": representations,
        }
