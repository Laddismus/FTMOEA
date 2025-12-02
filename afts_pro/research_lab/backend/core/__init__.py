"""Core services for the Research Lab backend."""

from research_lab.backend.core.config_loader import ResearchConfigLoader
from research_lab.backend.core.job_runner import InMemoryJobRunner, JobRunnerBase
from research_lab.backend.core.model_registry import FileSystemModelRegistry, ModelRegistryBase
from research_lab.backend.core.analytics import (
    FeatureExplorer,
    RollingKpiEngine,
    DriftDetector,
    RegimeClusteringService,
    SeriesStats,
    RollingKpiWindow,
    DriftResult,
    RegimeClusteringResult,
)
from research_lab.backend.core.strategy_builder import (
    NodeParamDefinition,
    NodeSpec,
    StrategyNode,
    StrategyEdge,
    StrategyGraph,
    ValidationIssue,
    NodeCatalog,
    GraphEngine,
    StrategyDslSerializer,
    StrategyConfigTranslator,
)
from research_lab.backend.core.python_strategies import (
    PythonStrategyInterface,
    BasePythonStrategy,
    PythonStrategyMetadata,
    PythonStrategyRegistrationRequest,
    PythonStrategyValidationResult,
    import_strategy_class,
    extract_metadata,
    PythonStrategyRegistry,
)

__all__ = [
    "ResearchConfigLoader",
    "JobRunnerBase",
    "InMemoryJobRunner",
    "ModelRegistryBase",
    "FileSystemModelRegistry",
    "FeatureExplorer",
    "RollingKpiEngine",
    "DriftDetector",
    "RegimeClusteringService",
    "SeriesStats",
    "RollingKpiWindow",
    "DriftResult",
    "RegimeClusteringResult",
    "NodeParamDefinition",
    "NodeSpec",
    "StrategyNode",
    "StrategyEdge",
    "StrategyGraph",
    "ValidationIssue",
    "NodeCatalog",
    "GraphEngine",
    "StrategyDslSerializer",
    "StrategyConfigTranslator",
    "PythonStrategyInterface",
    "BasePythonStrategy",
    "PythonStrategyMetadata",
    "PythonStrategyRegistrationRequest",
    "PythonStrategyValidationResult",
    "import_strategy_class",
    "extract_metadata",
    "PythonStrategyRegistry",
]
