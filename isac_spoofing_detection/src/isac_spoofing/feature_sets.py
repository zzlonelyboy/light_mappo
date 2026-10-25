from __future__ import annotations

from .features import (
    COORDINATED_NEIGHBOR_FEATURE_NAMES,
    FEATURE_NAMES,
    FIXED_FORMATION_SIGNED_ANOMALY_CONTRAST_FEATURE_NAME,
    FORMATION_DEFORMATION_NEIGHBOR_FEATURE_NAMES,
    INTER_UAV_FEATURE_NAMES,
    RESIDUAL_FEATURE_NAMES,
    SELF_FEATURE_NAMES,
)


LEGACY_TEMPORAL_FEATURE_NAMES: tuple[str, ...] = (
    "cusum_range",
    "cusum_radial_velocity",
    "cusum_direction",
)

LEGACY_FULL_F14: tuple[str, ...] = (
    FEATURE_NAMES[:6]
    + LEGACY_TEMPORAL_FEATURE_NAMES
    + FEATURE_NAMES[6:]
)

LEGACY_FULL_F15: tuple[str, ...] = tuple(
    f"{stat}_{modality}"
    for stat in ("mean", "max", "cusum")
    for modality in ("range", "radial_velocity", "direction")
) + (
    "gps_isac_position_error_norm",
    "gps_isac_position_error_heading_projection",
    "neighbor_relative_bias_norm",
    "neighbor_geometry_error_mean",
    "neighbor_geometry_error_max",
    "neighbor_geometry_error_drift",
)

_GROUPS: dict[str, tuple[str, ...]] = {
    "bs_residual": FEATURE_NAMES[:6],
    "gps_isac_position": FEATURE_NAMES[6:8],
    "neighbor_consistency": FEATURE_NAMES[8:11],
}

SELF_CONSENSUS_DEVIATION_FEATURE_NAME = "self_consensus_deviation"
SIGNED_ANOMALY_CONTRAST_FEATURE_NAME = "signed_self_neighbor_anomaly_contrast"
LONGITUDINAL_FORMATION_STRAIN_FEATURE_NAME = (
    "median_longitudinal_formation_strain"
)
TRANSVERSE_FORMATION_STRAIN_FEATURE_NAME = "median_transverse_formation_strain"
SELF_CONSENSUS_ONLY: tuple[str, ...] = (
    RESIDUAL_FEATURE_NAMES
    + SELF_FEATURE_NAMES
    + (SELF_CONSENSUS_DEVIATION_FEATURE_NAME,)
)
SIGNED_ANOMALY_CONTRAST_ONLY: tuple[str, ...] = (
    RESIDUAL_FEATURE_NAMES
    + SELF_FEATURE_NAMES
    + (SIGNED_ANOMALY_CONTRAST_FEATURE_NAME,)
)
FIXED_SIGNED_ANOMALY_CONTRAST_ONLY: tuple[str, ...] = (
    RESIDUAL_FEATURE_NAMES
    + SELF_FEATURE_NAMES
    + (FIXED_FORMATION_SIGNED_ANOMALY_CONTRAST_FEATURE_NAME,)
)
SIGNED_PLUS_TRANSVERSE_DEFORMATION: tuple[str, ...] = (
    RESIDUAL_FEATURE_NAMES
    + SELF_FEATURE_NAMES
    + (
        SIGNED_ANOMALY_CONTRAST_FEATURE_NAME,
        TRANSVERSE_FORMATION_STRAIN_FEATURE_NAME,
    )
)
FORMATION_DEFORMATION_FULL: tuple[str, ...] = (
    RESIDUAL_FEATURE_NAMES
    + SELF_FEATURE_NAMES
    + FORMATION_DEFORMATION_NEIGHBOR_FEATURE_NAMES
)
COORDINATED_NEIGHBOR_ONLY: tuple[str, ...] = (
    RESIDUAL_FEATURE_NAMES
    + SELF_FEATURE_NAMES
    + COORDINATED_NEIGHBOR_FEATURE_NAMES
)
INTER_UAV_CONSISTENCY_ONLY: tuple[str, ...] = (
    RESIDUAL_FEATURE_NAMES
    + SELF_FEATURE_NAMES
    + INTER_UAV_FEATURE_NAMES
)
POSITION_ONLY: tuple[str, ...] = SELF_FEATURE_NAMES
NO_BS_RESIDUAL_SIGNED_ONLY: tuple[str, ...] = (
    SELF_FEATURE_NAMES
    + (SIGNED_ANOMALY_CONTRAST_FEATURE_NAME,)
)
NO_BS_RESIDUAL_INTER_UAV_ONLY: tuple[str, ...] = (
    SELF_FEATURE_NAMES
    + INTER_UAV_FEATURE_NAMES
)
NO_GPS_ISAC_POSITION_SIGNED_ONLY: tuple[str, ...] = (
    RESIDUAL_FEATURE_NAMES
    + (SIGNED_ANOMALY_CONTRAST_FEATURE_NAME,)
)


def _without(*groups: str) -> tuple[str, ...]:
    removed = {name for group in groups for name in _GROUPS[group]}
    return tuple(name for name in FEATURE_NAMES if name not in removed)


FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "full_f11": FEATURE_NAMES,
    # Retained only so existing datasets/checkpoints remain explicitly
    # addressable. New data generation never emits these legacy schemas.
    "full_f14": LEGACY_FULL_F14,
    "full_f15": LEGACY_FULL_F15,
    "no_bs_residual": _without("bs_residual"),
    "no_gps_isac_position": _without("gps_isac_position"),
    "no_neighbor": _without("neighbor_consistency"),
    "position_only": POSITION_ONLY,
    "no_bs_residual_signed_only": NO_BS_RESIDUAL_SIGNED_ONLY,
    "no_bs_residual_inter_uav_only": NO_BS_RESIDUAL_INTER_UAV_ONLY,
    "no_gps_isac_position_signed_only": NO_GPS_ISAC_POSITION_SIGNED_ONLY,
    "self_consensus_only": SELF_CONSENSUS_ONLY,
    "signed_anomaly_contrast_only": SIGNED_ANOMALY_CONTRAST_ONLY,
    "dynamic_signed_anomaly_contrast_only": SIGNED_ANOMALY_CONTRAST_ONLY,
    "fixed_signed_anomaly_contrast_only": FIXED_SIGNED_ANOMALY_CONTRAST_ONLY,
    "signed_plus_transverse_deformation": SIGNED_PLUS_TRANSVERSE_DEFORMATION,
    "formation_deformation_full": FORMATION_DEFORMATION_FULL,
    "coordinated_neighbor_only": COORDINATED_NEIGHBOR_ONLY,
    "inter_uav_consistency": INTER_UAV_CONSISTENCY_ONLY,
    # Compatibility aliases for earlier commands. The formal schema is now
    # inherently temporal-free, so no additional feature is removed here.
    "no_handcrafted_temporal": FEATURE_NAMES,
    "no_temporal_no_bs_residual": _without("bs_residual"),
    "no_temporal_no_gps_isac_position": _without("gps_isac_position"),
    "no_temporal_no_neighbor": _without("neighbor_consistency"),
}


FEATURE_SET_DESCRIPTIONS: dict[str, str] = {
    "full_f11": (
        "Complete 11-dimensional instantaneous detector input: six BS-link "
        "residual statistics, two GPS-ISAC discrepancies, and three "
        "target-centered neighbor features (reference)."
    ),
    "full_f14": "Legacy 14-dimensional input containing three CUSUM features.",
    "full_f15": "Legacy 15-dimensional detector input.",
    "no_bs_residual": "Remove all six instantaneous BS-link residual statistics.",
    "no_gps_isac_position": "Remove the two GPS-ISAC position discrepancy features.",
    "no_bs_residual_signed_only": (
        "Keep only the two GPS-ISAC position features and the signed anomaly "
        "contrast — removes all BS residual statistics and uses the lightweight "
        "single-neighbor-feature design."
    ),
    "no_bs_residual_inter_uav_only": (
        "Keep only the two GPS-ISAC position features and the inter-UAV "
        "ranging consistency — removes all BS residual statistics."
    ),
    "no_gps_isac_position_signed_only": (
        "Keep only the six BS-link residual statistics and the signed anomaly "
        "contrast — removes GPS-ISAC position discrepancies and uses the "
        "lightweight single-neighbor-feature design."
    ),
    "no_neighbor": (
        "Keep the six BS residual statistics and two GPS-ISAC position "
        "features; remove every inter-UAV neighbor feature."
    ),
    "position_only": (
        "Keep only the two GPS-ISAC position-discrepancy features."
    ),
    "self_consensus_only": (
        "Keep the eight non-neighbor features and only the strongest robust "
        "neighbor feature, self_consensus_deviation."
    ),
    "signed_anomaly_contrast_only": (
        "Keep the eight non-neighbor features and only the signed target-versus-"
        "neighbor anomaly contrast."
    ),
    "dynamic_signed_anomaly_contrast_only": (
        "Keep the eight non-neighbor features and the signed anomaly contrast "
        "computed from per-step dynamic ISAC nearest neighbors."
    ),
    "fixed_signed_anomaly_contrast_only": (
        "Keep the eight non-neighbor features and the signed anomaly contrast "
        "computed from one attack-independent fixed formation topology."
    ),
    "signed_plus_transverse_deformation": (
        "Keep the eight non-neighbor features, the dynamic signed anomaly "
        "contrast, and the median transverse GPS-versus-ISAC formation strain."
    ),
    "formation_deformation_full": (
        "Keep the eight non-neighbor features and all three physically "
        "interpretable formation-deformation features: signed anomaly contrast, "
        "median longitudinal strain, and median transverse strain."
    ),
    "coordinated_neighbor_only": (
        "Keep the eight non-neighbor features and the three coordinated "
        "neighbourhood features: deviation median distance, mean geometry "
        "error, and max geometry error."
    ),
    "inter_uav_consistency": (
        "Eight core features plus inter-UAV ranging consistency, which "
        "compares simulated direct UAV-to-UAV ISAC ranges against "
        "GPS-derived inter-UAV distances using the minimum and maximum "
        "disparity over the selected K neighbors (2-D)."
    ),
    "no_handcrafted_temporal": (
        "Compatibility alias of full_f11; the formal schema contains no "
        "hand-crafted temporal features."
    ),
    "no_temporal_no_bs_residual": (
        "Alias of no_bs_residual under the temporal-free formal schema."
    ),
    "no_temporal_no_gps_isac_position": (
        "Alias of no_gps_isac_position under the temporal-free formal schema."
    ),
    "no_temporal_no_neighbor": (
        "Alias of no_neighbor under the temporal-free formal schema."
    ),
}


DEFAULT_ABLATION_FEATURE_SETS: tuple[str, ...] = (
    "full_f11",
    "no_bs_residual",
    "no_gps_isac_position",
    "no_neighbor",
)


def resolve_feature_set(
    name: str,
    available_feature_names: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    requested = str(name)
    if available_feature_names is not None and requested not in {
        "full_f14",
        "full_f15",
    }:
        available = tuple(str(value) for value in available_feature_names)
        if len(available) < 8 or len(set(available)) != len(available):
            raise ValueError(
                "dynamic compact feature schema must contain at least the eight "
                "unique canonical non-neighbor feature names"
            )
        if available[:6] != RESIDUAL_FEATURE_NAMES or available[6:8] != SELF_FEATURE_NAMES:
            raise ValueError(
                "dynamic compact feature schema must contain the canonical "
                "six residual and two self features before its neighbor features"
            )
        dynamic_groups = {
            "bs_residual": available[:6],
            "gps_isac_position": available[6:8],
            "neighbor_consistency": available[8:],
        }
        aliases = {
            "full_f11": (),
            "no_handcrafted_temporal": (),
            "no_bs_residual": ("bs_residual",),
            "no_temporal_no_bs_residual": ("bs_residual",),
            "no_gps_isac_position": ("gps_isac_position",),
            "no_temporal_no_gps_isac_position": ("gps_isac_position",),
            "no_neighbor": ("neighbor_consistency",),
            "no_temporal_no_neighbor": ("neighbor_consistency",),
        }
        if requested == "self_consensus_only":
            if SELF_CONSENSUS_DEVIATION_FEATURE_NAME not in available[8:11]:
                raise ValueError(
                    "feature set 'self_consensus_only' requires a dataset "
                    "containing the uncertainty-robust neighbor feature "
                    f"{SELF_CONSENSUS_DEVIATION_FEATURE_NAME!r}"
                )
            return available[:8] + (SELF_CONSENSUS_DEVIATION_FEATURE_NAME,)
        if requested in {
            "signed_anomaly_contrast_only",
            "dynamic_signed_anomaly_contrast_only",
        }:
            if SIGNED_ANOMALY_CONTRAST_FEATURE_NAME not in available[8:11]:
                raise ValueError(
                    f"feature set {requested!r} requires a "
                    "dataset containing the target-centered neighbor feature "
                    f"{SIGNED_ANOMALY_CONTRAST_FEATURE_NAME!r}"
                )
            return available[:8] + (SIGNED_ANOMALY_CONTRAST_FEATURE_NAME,)
        if requested == "fixed_signed_anomaly_contrast_only":
            if (
                FIXED_FORMATION_SIGNED_ANOMALY_CONTRAST_FEATURE_NAME
                not in available[8:]
            ):
                raise ValueError(
                    "feature set 'fixed_signed_anomaly_contrast_only' requires "
                    "a dataset containing the fixed-formation neighbor feature "
                    f"{FIXED_FORMATION_SIGNED_ANOMALY_CONTRAST_FEATURE_NAME!r}"
                )
            return available[:8] + (
                FIXED_FORMATION_SIGNED_ANOMALY_CONTRAST_FEATURE_NAME,
            )
        if requested in {
            "signed_plus_transverse_deformation",
            "formation_deformation_full",
        }:
            required = (
                (
                    SIGNED_ANOMALY_CONTRAST_FEATURE_NAME,
                    TRANSVERSE_FORMATION_STRAIN_FEATURE_NAME,
                )
                if requested == "signed_plus_transverse_deformation"
                else FORMATION_DEFORMATION_NEIGHBOR_FEATURE_NAMES
            )
            missing = [
                feature for feature in required if feature not in available[8:]
            ]
            if missing:
                raise ValueError(
                    f"feature set {requested!r} requires a formation-deformation "
                    f"dataset containing: {missing}"
                )
            return available[:8] + tuple(required)
        if requested in aliases:
            removed = {
                feature
                for group in aliases[requested]
                for feature in dynamic_groups[group]
            }
            return tuple(feature for feature in available if feature not in removed)
    try:
        return FEATURE_SETS[requested]
    except KeyError as exc:
        available = ", ".join(FEATURE_SETS)
        raise ValueError(
            f"unknown feature set {name!r}; choose from: {available}"
        ) from exc
