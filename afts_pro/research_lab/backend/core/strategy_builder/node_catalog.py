"""Catalog of available strategy builder nodes."""

from __future__ import annotations

from typing import List, Optional

from research_lab.backend.core.strategy_builder.models import NodeParamDefinition, NodeSpec

ALLOWED_DTYPES = {"float", "int", "bool", "string"}


class NodeCatalog:
    """Holds the curated Core v1 node specifications for the strategy builder."""

    def __init__(self) -> None:
        self._nodes = self._load_core_v1_nodes()

    def list_nodes(self) -> List[NodeSpec]:
        """Return all available node specifications."""

        return list(self._nodes.values())

    def get_node(self, type: str) -> Optional[NodeSpec]:
        """Return a node specification by type."""

        return self._nodes.get(type)

    def _load_core_v1_nodes(self) -> dict[str, NodeSpec]:
        nodes: dict[str, NodeSpec] = {}

        def add(spec: NodeSpec) -> None:
            nodes[spec.type] = spec

        # Source Nodes
        add(
            NodeSpec(
                type="price_source",
                category="source",
                stage="entry",
                description="Base OHLCV price source for a given symbol and timeframe.",
                inputs=[],
                outputs=["open", "high", "low", "close", "volume"],
                params=[
                    NodeParamDefinition(name="symbol", dtype="string", required=True),
                    NodeParamDefinition(name="timeframe", dtype="string", required=True),
                ],
                tags=["price", "ohlcv"],
                version="1.0.0",
            )
        )
        add(
            NodeSpec(
                type="htf_price_source",
                category="source",
                stage="entry",
                description="Higher timeframe aggregated OHLCV source.",
                inputs=[],
                outputs=["open", "high", "low", "close", "volume"],
                params=[
                    NodeParamDefinition(name="symbol", dtype="string", required=True),
                    NodeParamDefinition(name="timeframe", dtype="string", required=True),
                    NodeParamDefinition(name="aggregation", dtype="string", required=False, default="ohlc"),
                ],
                tags=["price", "ohlcv", "htf"],
                version="1.0.0",
            )
        )

        # Indicator Nodes
        add(
            NodeSpec(
                type="indicator_sma",
                category="indicator",
                stage="entry",
                description="Simple moving average over a numeric input series.",
                inputs=["source"],
                outputs=["sma"],
                params=[
                    NodeParamDefinition(name="length", dtype="int", required=True),
                    NodeParamDefinition(name="field", dtype="string", required=True, default="close"),
                ],
                tags=["trend", "moving_average"],
                version="1.0.0",
            )
        )
        add(
            NodeSpec(
                type="indicator_ema",
                category="indicator",
                stage="entry",
                description="Exponential moving average over a numeric input series.",
                inputs=["source"],
                outputs=["ema"],
                params=[
                    NodeParamDefinition(name="length", dtype="int", required=True),
                    NodeParamDefinition(name="field", dtype="string", required=True, default="close"),
                ],
                tags=["trend", "moving_average"],
                version="1.0.0",
            )
        )
        add(
            NodeSpec(
                type="indicator_rsi",
                category="indicator",
                stage="entry",
                description="Relative Strength Index over a price/returns input.",
                inputs=["source"],
                outputs=["rsi"],
                params=[
                    NodeParamDefinition(name="length", dtype="int", required=True),
                    NodeParamDefinition(name="field", dtype="string", required=True, default="close"),
                ],
                tags=["momentum", "oscillator"],
                version="1.0.0",
            )
        )
        add(
            NodeSpec(
                type="indicator_atr",
                category="indicator",
                stage="entry",
                description="Average True Range volatility indicator.",
                inputs=["source"],
                outputs=["atr"],
                params=[NodeParamDefinition(name="length", dtype="int", required=True)],
                tags=["volatility"],
                version="1.0.0",
            )
        )

        # Condition Nodes
        add(
            NodeSpec(
                type="condition_greater_than",
                category="condition",
                stage="entry",
                description="Checks if left input is greater than right input.",
                inputs=["left", "right"],
                outputs=["condition"],
                params=[NodeParamDefinition(name="strict", dtype="bool", required=False, default=True)],
                tags=["comparison"],
                version="1.0.0",
            )
        )
        add(
            NodeSpec(
                type="condition_less_than",
                category="condition",
                stage="entry",
                description="Checks if left input is less than right input.",
                inputs=["left", "right"],
                outputs=["condition"],
                params=[NodeParamDefinition(name="strict", dtype="bool", required=False, default=True)],
                tags=["comparison"],
                version="1.0.0",
            )
        )
        add(
            NodeSpec(
                type="condition_cross_over",
                category="condition",
                stage="entry",
                description="Detects cross-over event from below to above.",
                inputs=["fast", "slow"],
                outputs=["condition"],
                params=[],
                tags=["crossover", "trend"],
                version="1.0.0",
            )
        )
        add(
            NodeSpec(
                type="condition_cross_under",
                category="condition",
                stage="entry",
                description="Detects cross-under event from above to below.",
                inputs=["fast", "slow"],
                outputs=["condition"],
                params=[],
                tags=["crossover", "trend"],
                version="1.0.0",
            )
        )

        # Logic Nodes
        add(
            NodeSpec(
                type="logic_and",
                category="logic",
                stage="entry",
                description="Logical AND over two boolean conditions.",
                inputs=["a", "b"],
                outputs=["condition"],
                params=[],
                tags=["logic"],
                version="1.0.0",
            )
        )
        add(
            NodeSpec(
                type="logic_or",
                category="logic",
                stage="entry",
                description="Logical OR over two boolean conditions.",
                inputs=["a", "b"],
                outputs=["condition"],
                params=[],
                tags=["logic"],
                version="1.0.0",
            )
        )
        add(
            NodeSpec(
                type="logic_not",
                category="logic",
                stage="entry",
                description="Logical NOT of a boolean condition.",
                inputs=["value"],
                outputs=["condition"],
                params=[],
                tags=["logic"],
                version="1.0.0",
            )
        )

        # Signal Nodes
        add(
            NodeSpec(
                type="signal_long",
                category="signal",
                stage="entry",
                description="Produces a long entry signal when its input condition is true.",
                inputs=["condition"],
                outputs=["signal_long"],
                params=[],
                tags=["signal", "entry"],
                version="1.0.0",
            )
        )
        add(
            NodeSpec(
                type="signal_short",
                category="signal",
                stage="entry",
                description="Produces a short entry signal when its input condition is true.",
                inputs=["condition"],
                outputs=["signal_short"],
                params=[],
                tags=["signal", "entry"],
                version="1.0.0",
            )
        )
        add(
            NodeSpec(
                type="signal_flat",
                category="signal",
                stage="exit",
                description="Produces a flat/exit signal when its input condition is true.",
                inputs=["condition"],
                outputs=["signal_flat"],
                params=[],
                tags=["signal", "exit"],
                version="1.0.0",
            )
        )

        # Risk Nodes
        add(
            NodeSpec(
                type="risk_fixed_sl_tp",
                category="risk",
                stage="risk",
                description="Defines fixed SL/TP in percentage terms.",
                inputs=["signal"],
                outputs=["risk_profile"],
                params=[
                    NodeParamDefinition(name="sl_pct", dtype="float", required=True),
                    NodeParamDefinition(name="tp_pct", dtype="float", required=True),
                ],
                tags=["risk", "sl", "tp"],
                version="1.0.0",
            )
        )
        add(
            NodeSpec(
                type="risk_atr_sl_tp",
                category="risk",
                stage="risk",
                description="Defines SL/TP based on ATR multiples.",
                inputs=["signal", "atr"],
                outputs=["risk_profile"],
                params=[
                    NodeParamDefinition(name="sl_atr_mult", dtype="float", required=True),
                    NodeParamDefinition(name="tp_atr_mult", dtype="float", required=True),
                ],
                tags=["risk", "atr", "sl", "tp"],
                version="1.0.0",
            )
        )

        # Filter Nodes
        add(
            NodeSpec(
                type="filter_session",
                category="filter",
                stage="filter",
                description="Filters signals by trading session/time-of-day.",
                inputs=["signal"],
                outputs=["signal"],
                params=[NodeParamDefinition(name="session", dtype="string", required=True)],
                tags=["session", "time_filter"],
                version="1.0.0",
            )
        )
        add(
            NodeSpec(
                type="filter_volatility",
                category="filter",
                stage="filter",
                description="Allows signals only if volatility (e.g. ATR) is above/below threshold.",
                inputs=["signal", "atr"],
                outputs=["signal"],
                params=[
                    NodeParamDefinition(name="mode", dtype="string", required=True, default="min"),
                    NodeParamDefinition(name="threshold", dtype="float", required=True),
                ],
                tags=["volatility", "filter"],
                version="1.0.0",
            )
        )

        return nodes


__all__ = ["NodeCatalog", "ALLOWED_DTYPES"]
