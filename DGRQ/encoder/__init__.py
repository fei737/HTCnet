from .common import (
    ConvBNAct,
    ConvGNAct,
    FixedBoundaryExtractor,
    LearnableBlend,
    LearnableGate,
    set_encoder_drop_path,
)
from .fusion import (
    AngularGuidedGlobalAttention,
    CrossScaleFeatureAligner,
    CrossScaleDualGatedAlignmentModule,
    DynamicGatedLocalSemanticRoutingModule,
    GeometryAwareContextAttention,
    LocalSemanticFusionBlock,
    SP_ASPP,
    StripPyramidASPP,
)
from .geometry import (
    AngularBoundaryDescriptor,
    AngularGeometryFieldDescriptor,
    AngularGuidancePyramid,
    BimodalStructuralDifferentialEnhancer,
    CrossModalReliabilityEstimator,
    DirectionalEdgePromptRefiner,
    DirectionalGeometryEdgePromptRefiner,
    FrequencyAwareGeometryPrompt,
    FrequencyDisentangledGeometricPromptField,
    GeometryAppearanceConsistencyRouting,
    GeometryPromptRecovery,
    GeometryPromptRecoveryBlock,
    GeometryReliabilityEstimator,
    GuidanceDrivenStructuralTextureFilter,
    ModalUncertaintyEstimationBlock,
    MultiScaleAngularGuidancePyramid,
    ReliabilityAwareAutocorrelationPromptMixer,
    ReliabilityAwareAutocorrelationPromptMixerBlock,
    RGBGuidedGeometryRecovery,
    RGBGuidedGeometryPromptRecoveryBlock,
    StructuralContrastEnhancer,
    StructuralPromptFilter,
)


def __getattr__(name):
    if name in {"UncertaintyGuidedFusionEncoder", "DGRQEncoder"}:
        from .backbone import DGRQEncoder, UncertaintyGuidedFusionEncoder
        globals()["UncertaintyGuidedFusionEncoder"] = UncertaintyGuidedFusionEncoder
        globals()["DGRQEncoder"] = DGRQEncoder
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "UncertaintyGuidedFusionEncoder",
    "DGRQEncoder",
    "ConvBNAct",
    "ConvGNAct",
    "FixedBoundaryExtractor",
    "LearnableBlend",
    "LearnableGate",
    "set_encoder_drop_path",
    "AngularBoundaryDescriptor",
    "AngularGuidancePyramid",
    "CrossModalReliabilityEstimator",
    "DirectionalEdgePromptRefiner",
    "FrequencyAwareGeometryPrompt",
    "GeometryPromptRecovery",
    "GeometryReliabilityEstimator",
    "RGBGuidedGeometryRecovery",
    "ReliabilityAwareAutocorrelationPromptMixer",
    "StructuralContrastEnhancer",
    "StructuralPromptFilter",
    "AngularGuidedGlobalAttention",
    "CrossScaleFeatureAligner",
    "GeometryAwareContextAttention",
    "LocalSemanticFusionBlock",
    "StripPyramidASPP",
    "AngularGeometryFieldDescriptor",
    "BimodalStructuralDifferentialEnhancer",
    "DirectionalGeometryEdgePromptRefiner",
    "FrequencyDisentangledGeometricPromptField",
    "GeometryAppearanceConsistencyRouting",
    "GeometryPromptRecoveryBlock",
    "GuidanceDrivenStructuralTextureFilter",
    "ModalUncertaintyEstimationBlock",
    "MultiScaleAngularGuidancePyramid",
    "RGBGuidedGeometryPromptRecoveryBlock",
    "ReliabilityAwareAutocorrelationPromptMixerBlock",
    "CrossScaleDualGatedAlignmentModule",
    "DynamicGatedLocalSemanticRoutingModule",
    "SP_ASPP",
]
