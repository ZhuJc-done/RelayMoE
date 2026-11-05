from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Mapping

from .model import HierarchicalTopology
from .scheduler import Schedule, ScheduledAction


@dataclass(frozen=True)
class SimulationConfig:
    stage_launch_overhead_us: float = 1.5
    relay_kernel_overhead_us: float = 0.35
    reduction_launch_overhead_us: float = 0.2
    reduction_bandwidth_GBps: float = 200.0

    def __post_init__(self) -> None:
        if self.stage_launch_overhead_us < 0:
            raise ValueError("stage launch overhead must be non-negative")
        if self.relay_kernel_overhead_us < 0:
            raise ValueError("relay overhead must be non-negative")
        if self.reduction_launch_overhead_us < 0:
            raise ValueError("reduction launch overhead must be non-negative")
        if self.reduction_bandwidth_GBps <= 0:
            raise ValueError("reduction bandwidth must be positive")


@dataclass(frozen=True)
class ActionTiming:
    action_name: str
    start_us: float
    end_us: float
    duration_us: float
    serialization_us: float
    path_latency_us: float
    processing_overhead_us: float
    max_edge_concurrency: int
    physical_edges: tuple[str, ...]


@dataclass(frozen=True)
class StageTiming:
    index: int
    start_us: float
    end_us: float
    duration_us: float
    actions: tuple[ActionTiming, ...]


@dataclass(frozen=True)
class PhaseSimulationResult:
    phase: str
    stages: tuple[StageTiming, ...]
    total_time_us: float
    total_physical_bytes: int
    edge_bytes: Mapping[str, int]
    edge_busy_us: Mapping[str, float]
    edge_peak_concurrency: Mapping[str, int]

    @property
    def max_edge_busy_us(self) -> float:
        return max(self.edge_busy_us.values(), default=0.0)

    @property
    def action_count(self) -> int:
        return sum(len(stage.actions) for stage in self.stages)

    @property
    def max_edge_concurrency(self) -> int:
        return max(self.edge_peak_concurrency.values(), default=0)


@dataclass(frozen=True)
class SimulationResult:
    dispatch: PhaseSimulationResult
    combine: PhaseSimulationResult

    @property
    def total_time_us(self) -> float:
        return self.dispatch.total_time_us + self.combine.total_time_us

    @property
    def total_physical_bytes(self) -> int:
        return (
            self.dispatch.total_physical_bytes
            + self.combine.total_physical_bytes
        )

    def to_dict(self) -> dict:
        result = asdict(self)
        result["total_time_us"] = self.total_time_us
        result["total_physical_bytes"] = self.total_physical_bytes
        return result


class StageSimulator:
    """Coarse stage-synchronous simulator.

    A path is modeled as pipelined: serialization is determined by its slowest
    edge while per-edge propagation latencies add. If N actions in one stage
    share a directed edge, each receives 1/N of that edge's bandwidth for the
    stage. Edge-exclusive schedules are the special case N=1.
    """

    def __init__(
        self,
        topology: HierarchicalTopology,
        config: SimulationConfig | None = None,
    ) -> None:
        self.topology = topology
        self.config = config or SimulationConfig()

    def simulate(
        self,
        dispatch: Schedule,
        combine: Schedule,
    ) -> SimulationResult:
        dispatch.validate()
        combine.validate()
        if dispatch.phase != "dispatch" or combine.phase != "combine":
            raise ValueError("simulate expects dispatch and combine schedules")
        return SimulationResult(
            dispatch=self.simulate_phase(dispatch),
            combine=self.simulate_phase(combine),
        )

    def simulate_phase(self, schedule: Schedule) -> PhaseSimulationResult:
        schedule.validate()
        stage_timings: list[StageTiming] = []
        edge_bytes: dict[str, int] = defaultdict(int)
        edge_busy_us: dict[str, float] = defaultdict(float)
        edge_peak_concurrency: dict[str, int] = defaultdict(int)
        current_time_us = 0.0

        for stage in schedule.stages:
            edge_concurrency = Counter(
                edge_name
                for action in stage.actions
                for edge_name in action.physical_edges
            )
            for edge_name, concurrency in edge_concurrency.items():
                edge_peak_concurrency[edge_name] = max(
                    edge_peak_concurrency[edge_name],
                    concurrency,
                )
            action_durations = [
                self._action_duration(action, edge_concurrency)
                for action in stage.actions
            ]
            max_action_duration = max(
                (duration[0] for duration in action_durations), default=0.0
            )
            stage_duration = (
                self.config.stage_launch_overhead_us + max_action_duration
            )
            action_start = current_time_us + self.config.stage_launch_overhead_us
            action_timings: list[ActionTiming] = []

            for action, (
                duration_us,
                serialization_us,
                path_latency_us,
                processing_overhead_us,
                max_edge_concurrency,
            ) in zip(stage.actions, action_durations):
                action_timings.append(
                    ActionTiming(
                        action_name=action.name,
                        start_us=action_start,
                        end_us=action_start + duration_us,
                        duration_us=duration_us,
                        serialization_us=serialization_us,
                        path_latency_us=path_latency_us,
                        processing_overhead_us=processing_overhead_us,
                        max_edge_concurrency=max_edge_concurrency,
                        physical_edges=action.physical_edges,
                    )
                )
                for edge_name in action.physical_edges:
                    edge = self.topology.edges[edge_name]
                    edge_bytes[edge_name] += action.size_bytes
                    edge_busy_us[edge_name] += edge.serialization_time_us(
                        action.size_bytes
                    )

            stage_timings.append(
                StageTiming(
                    index=stage.index,
                    start_us=current_time_us,
                    end_us=current_time_us + stage_duration,
                    duration_us=stage_duration,
                    actions=tuple(action_timings),
                )
            )
            current_time_us += stage_duration

        return PhaseSimulationResult(
            phase=schedule.phase,
            stages=tuple(stage_timings),
            total_time_us=current_time_us,
            total_physical_bytes=sum(edge_bytes.values()),
            edge_bytes=dict(edge_bytes),
            edge_busy_us=dict(edge_busy_us),
            edge_peak_concurrency=dict(edge_peak_concurrency),
        )

    def _action_duration(
        self,
        action: ScheduledAction,
        edge_concurrency: Mapping[str, int],
    ) -> tuple[float, float, float, float, int]:
        serialization_us = max(
            (
                self.topology.edges[edge_name].serialization_time_us(
                    action.size_bytes
                )
                * edge_concurrency[edge_name]
                for edge_name in action.physical_edges
            ),
            default=0.0,
        )
        max_edge_concurrency = max(
            (
                edge_concurrency[edge_name]
                for edge_name in action.physical_edges
            ),
            default=0,
        )
        path_latency_us = self.topology.route_latency_us(action.physical_edges)
        processing_overhead_us = 0.0
        if action.relay:
            processing_overhead_us += self.config.relay_kernel_overhead_us
        if action.phase == "combine" and action.reduction_inputs > 0:
            reduction_bytes = action.size_bytes * action.reduction_inputs
            processing_overhead_us += (
                self.config.reduction_launch_overhead_us
                + reduction_bytes
                / (self.config.reduction_bandwidth_GBps * 1_000.0)
            )
        duration_us = (
            serialization_us + path_latency_us + processing_overhead_us
        )
        return (
            duration_us,
            serialization_us,
            path_latency_us,
            processing_overhead_us,
            max_edge_concurrency,
        )
