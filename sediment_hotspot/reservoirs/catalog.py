"""Reservoir Catalog and Hydrological Zone Specifications.

Maps monitoring stations in Krishna, Narmada, and Kaveri basins to actual
reservoirs, defining geographic boundaries, dam coordinates, upstream/downstream
connectivity, and hydrologically meaningful zone partitions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ZoneSpec:
    zone_id: str
    name: str
    zone_type: str  # e.g., "River-Entry", "Riverine", "Transition", "Central", "Lacustrine", "Dam-Proximal"
    description: str
    relative_flow_position: float  # 0.0 (inlet) to 1.0 (dam)
    nominal_depth_category: str  # "Shallow", "Intermediate", "Deep"


@dataclass(frozen=True)
class ReservoirSpec:
    id: str
    name: str
    basin: str
    river: str
    dam_name: str
    monitoring_station_id: str  # Associated ID from 14-point network
    dam_coordinates: Tuple[float, float]  # (lat, lon)
    inflow_coordinates: Tuple[float, float]  # (lat, lon)
    bbox: Dict[str, float]  # lat_min, lat_max, lon_min, lon_max
    upstream_stations: List[str]
    downstream_stations: List[str]
    full_reservoir_level_m: float
    gross_storage_bcm: float
    catchment_area_sqkm: float
    is_primary: bool = True
    zones: List[ZoneSpec] = field(default_factory=list)


RESERVOIR_CATALOG: Dict[str, ReservoirSpec] = {
    # -------------------------------------------------------------------------
    # 1. KRISHNA BASIN: Nagarjuna Sagar Reservoir (Prioritized)
    # -------------------------------------------------------------------------
    "krishna_nagarjuna_sagar": ReservoirSpec(
        id="krishna_nagarjuna_sagar",
        name="Nagarjuna Sagar Reservoir",
        basin="Krishna",
        river="Krishna River",
        dam_name="Nagarjuna Sagar Dam",
        monitoring_station_id="K4_Nagarjuna_Sagar",
        dam_coordinates=(16.5750, 79.3167),
        inflow_coordinates=(16.4600, 79.0800),
        bbox={
            "lat_min": 16.42,
            "lat_max": 16.64,
            "lon_min": 79.05,
            "lon_max": 79.35,
        },
        upstream_stations=["K1_Mahabaleshwar", "K2_Sangli_Almatti", "K3_Raichur_Kurnool"],
        downstream_stations=["K5_Vijayawada_Delta"],
        full_reservoir_level_m=179.8,
        gross_storage_bcm=11.472,
        catchment_area_sqkm=215000.0,
        is_primary=True,
        zones=[
            ZoneSpec(
                zone_id="NS_Z1",
                name="Upstream River-Entry Zone",
                zone_type="River-Entry",
                description="Krishna-Dindi confluence canyon inlet with high hydraulic energy; initial coarse gravel/sand drop.",
                relative_flow_position=0.08,
                nominal_depth_category="Shallow",
            ),
            ZoneSpec(
                zone_id="NS_Z2",
                name="Upper Riverine Inflow Zone",
                zone_type="Riverine",
                description="Confined fluvial transport reach along the Nallamala range gorge; active bedload transport.",
                relative_flow_position=0.25,
                nominal_depth_category="Shallow-Intermediate",
            ),
            ZoneSpec(
                zone_id="NS_Z3",
                name="Transition Inundation Zone",
                zone_type="Transition",
                description="Hydraulic backwater deceleration zone where flow cross-section widens and medium silt settles.",
                relative_flow_position=0.45,
                nominal_depth_category="Intermediate",
            ),
            ZoneSpec(
                zone_id="NS_Z4",
                name="Central Reservoir Broad Reach",
                zone_type="Central",
                description="Expansive deep open-water impoundment basin with high water residence time and clay/silt fallout.",
                relative_flow_position=0.65,
                nominal_depth_category="Deep",
            ),
            ZoneSpec(
                zone_id="NS_Z5",
                name="Deep Lacustrine Storage Zone",
                zone_type="Lacustrine",
                description="Thermally stratified lacustrine dead-storage zone subject to density turbidity currents.",
                relative_flow_position=0.83,
                nominal_depth_category="Deep",
            ),
            ZoneSpec(
                zone_id="NS_Z6",
                name="Dam-Proximal Forebay Zone",
                zone_type="Dam-Proximal",
                description="Forebay intake pool directly upstream of Nagarjuna Sagar masonry dam spillway and penstocks.",
                relative_flow_position=0.96,
                nominal_depth_category="Deep",
            ),
        ],
    ),

    # -------------------------------------------------------------------------
    # 2. NARMADA BASIN: Omkareshwar Reservoir (Prioritized)
    # -------------------------------------------------------------------------
    "narmada_omkareshwar": ReservoirSpec(
        id="narmada_omkareshwar",
        name="Omkareshwar Reservoir",
        basin="Narmada",
        river="Narmada River",
        dam_name="Omkareshwar Dam",
        monitoring_station_id="N3_Omkareshwar",
        dam_coordinates=(22.2452, 76.1510),
        inflow_coordinates=(22.2580, 76.3200),
        bbox={
            "lat_min": 22.19,
            "lat_max": 22.29,
            "lon_min": 76.12,
            "lon_max": 76.35,
        },
        upstream_stations=["N1_Amarkantak", "N2_Jabalpur"],
        downstream_stations=["N4_Bharuch"],
        full_reservoir_level_m=196.6,
        gross_storage_bcm=0.987,
        catchment_area_sqkm=64880.0,
        is_primary=True,
        zones=[
            ZoneSpec(
                zone_id="OM_Z1",
                name="Indira Sagar Tailrace Entry Zone",
                zone_type="River-Entry",
                description="Turbulent discharge inflow entry downstream of Indira Sagar tailrace; high energy scouring zone.",
                relative_flow_position=0.10,
                nominal_depth_category="Intermediate",
            ),
            ZoneSpec(
                zone_id="OM_Z2",
                name="Riverine Canyon Zone",
                zone_type="Riverine",
                description="Deep basalt canyon reaches of the Narmada valley with confined velocity and lateral erosion.",
                relative_flow_position=0.28,
                nominal_depth_category="Intermediate",
            ),
            ZoneSpec(
                zone_id="OM_Z3",
                name="Kaveri-Narmada Confluence Transition Zone",
                zone_type="Transition",
                description="Hydrological confluence with Kaveri tributary branch forming local sediment wedge.",
                relative_flow_position=0.48,
                nominal_depth_category="Intermediate",
            ),
            ZoneSpec(
                zone_id="OM_Z4",
                name="Mid-Reservoir Mainstream Zone",
                zone_type="Central",
                description="Widened mainstream storage pool with reduced flow velocity and suspended solids deposition.",
                relative_flow_position=0.68,
                nominal_depth_category="Intermediate-Deep",
            ),
            ZoneSpec(
                zone_id="OM_Z5",
                name="Mandhata Island Lacustrine Reach",
                zone_type="Lacustrine",
                description="Braided deep channel surrounding Mandhata Island with significant silt deposition pockets.",
                relative_flow_position=0.85,
                nominal_depth_category="Deep",
            ),
            ZoneSpec(
                zone_id="OM_Z6",
                name="Omkareshwar Dam Forebay Zone",
                zone_type="Dam-Proximal",
                description="Deep pool immediately upstream of Omkareshwar gravity dam and powerhouse intake.",
                relative_flow_position=0.97,
                nominal_depth_category="Deep",
            ),
        ],
    ),

    # -------------------------------------------------------------------------
    # 3. KAVERI BASIN: Mettur Dam (Stanley Reservoir) (Prioritized)
    # -------------------------------------------------------------------------
    "kaveri_mettur": ReservoirSpec(
        id="kaveri_mettur",
        name="Mettur Dam (Stanley Reservoir)",
        basin="Kaveri",
        river="Kaveri River",
        dam_name="Mettur Dam",
        monitoring_station_id="C4_Mettur_Dam",
        dam_coordinates=(11.7870, 77.8008),
        inflow_coordinates=(12.0100, 77.7700),
        bbox={
            "lat_min": 11.75,
            "lat_max": 12.04,
            "lon_min": 77.70,
            "lon_max": 77.88,
        },
        upstream_stations=["C1_Talakaveri", "C2_Kodagu", "C3_Mysuru_KRS"],
        downstream_stations=["C5_Thanjavur_Delta"],
        full_reservoir_level_m=241.0,
        gross_storage_bcm=2.640,
        catchment_area_sqkm=42217.0,
        is_primary=True,
        zones=[
            ZoneSpec(
                zone_id="MT_Z1",
                name="Hogenakkal Falls River-Entry Zone",
                zone_type="River-Entry",
                description="Kaveri gorge entry below Hogenakkal falls; high stream power, coarse sediment input.",
                relative_flow_position=0.08,
                nominal_depth_category="Shallow",
            ),
            ZoneSpec(
                zone_id="MT_Z2",
                name="Upper Riverine Delta Inflow Zone",
                zone_type="Riverine",
                description="Upper deltaic formation where river transitions into reservoir headwaters with sand bars.",
                relative_flow_position=0.25,
                nominal_depth_category="Shallow-Intermediate",
            ),
            ZoneSpec(
                zone_id="MT_Z3",
                name="Chinnar Tributary Transition Zone",
                zone_type="Transition",
                description="Confluence with Chinnar river creating dynamic tributary sediment mixing and delta advance.",
                relative_flow_position=0.45,
                nominal_depth_category="Intermediate",
            ),
            ZoneSpec(
                zone_id="MT_Z4",
                name="Central Stanley Reservoir Basin",
                zone_type="Central",
                description="Broadest section of Stanley Reservoir; extensive fine sediment deposition over decades.",
                relative_flow_position=0.65,
                nominal_depth_category="Deep",
            ),
            ZoneSpec(
                zone_id="MT_Z5",
                name="Lacustrine Storage Reach",
                zone_type="Lacustrine",
                description="Deep stratified lacustrine section with low disturbance and sustained clay flocculation.",
                relative_flow_position=0.84,
                nominal_depth_category="Deep",
            ),
            ZoneSpec(
                zone_id="MT_Z6",
                name="Mettur Dam Forebay Zone",
                zone_type="Dam-Proximal",
                description="Deep dead storage forebay adjacent to Mettur Dam upstream face, Ellis saddle, and sluices.",
                relative_flow_position=0.96,
                nominal_depth_category="Deep",
            ),
        ],
    ),

    # -------------------------------------------------------------------------
    # 4. KAVERI BASIN: Krishna Raja Sagara (KRS) (Secondary)
    # -------------------------------------------------------------------------
    "kaveri_krs": ReservoirSpec(
        id="kaveri_krs",
        name="Krishna Raja Sagara (KRS)",
        basin="Kaveri",
        river="Kaveri River",
        dam_name="KRS Dam",
        monitoring_station_id="C3_Mysuru_KRS",
        dam_coordinates=(12.2958, 76.5744),
        inflow_coordinates=(12.3500, 76.4500),
        bbox={
            "lat_min": 12.25,
            "lat_max": 12.38,
            "lon_min": 76.42,
            "lon_max": 76.62,
        },
        upstream_stations=["C1_Talakaveri", "C2_Kodagu"],
        downstream_stations=["C4_Mettur_Dam", "C5_Thanjavur_Delta"],
        full_reservoir_level_m=228.0,
        gross_storage_bcm=1.400,
        catchment_area_sqkm=10619.0,
        is_primary=False,
        zones=[
            ZoneSpec("KRS_Z1", "Hemavati-Kaveri Inlet", "River-Entry", "Upper dual-tributary inlet", 0.12, "Shallow"),
            ZoneSpec("KRS_Z2", "Riverine Inundation Reach", "Riverine", "Shallow fluvial delta", 0.32, "Shallow"),
            ZoneSpec("KRS_Z3", "Central Storage Basin", "Central", "Main body silt deposition", 0.58, "Intermediate"),
            ZoneSpec("KRS_Z4", "Lacustrine Pool", "Lacustrine", "Deep calm pool", 0.80, "Deep"),
            ZoneSpec("KRS_Z5", "KRS Dam Forebay", "Dam-Proximal", "Forebay near Brindavan gardens sluices", 0.96, "Deep"),
        ],
    ),

    # -------------------------------------------------------------------------
    # 5. KRISHNA BASIN: Almatti Reservoir (Secondary)
    # -------------------------------------------------------------------------
    "krishna_almatti": ReservoirSpec(
        id="krishna_almatti",
        name="Almatti Reservoir (Lal Bahadur Shastri)",
        basin="Krishna",
        river="Krishna River",
        dam_name="Almatti Dam",
        monitoring_station_id="K2_Sangli_Almatti",
        dam_coordinates=(16.3315, 75.8885),
        inflow_coordinates=(16.4200, 75.7000),
        bbox={
            "lat_min": 16.25,
            "lat_max": 16.46,
            "lon_min": 75.65,
            "lon_max": 75.95,
        },
        upstream_stations=["K1_Mahabaleshwar"],
        downstream_stations=["K3_Raichur_Kurnool", "K4_Nagarjuna_Sagar", "K5_Vijayawada_Delta"],
        full_reservoir_level_m=519.6,
        gross_storage_bcm=3.440,
        catchment_area_sqkm=35925.0,
        is_primary=False,
        zones=[
            ZoneSpec("ALM_Z1", "Upper Krishna Entry", "River-Entry", "Upper inlet reach", 0.15, "Shallow"),
            ZoneSpec("ALM_Z2", "Transition Reach", "Transition", "Backwater silt deposit", 0.40, "Intermediate"),
            ZoneSpec("ALM_Z3", "Central Almatti Basin", "Central", "Broad storage lake", 0.68, "Intermediate"),
            ZoneSpec("ALM_Z4", "Almatti Forebay", "Dam-Proximal", "Dam-proximal forebay and spillway", 0.95, "Deep"),
        ],
    ),
}


def get_reservoir_spec(reservoir_id: str) -> ReservoirSpec:
    if reservoir_id not in RESERVOIR_CATALOG:
        raise KeyError(
            f"Unknown reservoir id: {reservoir_id}. Available: {list(RESERVOIR_CATALOG.keys())}"
        )
    return RESERVOIR_CATALOG[reservoir_id]


def list_primary_reservoirs() -> List[ReservoirSpec]:
    return [r for r in RESERVOIR_CATALOG.values() if r.is_primary]


def list_all_reservoirs() -> List[ReservoirSpec]:
    return list(RESERVOIR_CATALOG.values())
