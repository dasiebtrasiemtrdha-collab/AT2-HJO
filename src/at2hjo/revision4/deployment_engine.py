"""Causal, persistent deployment traffic for manuscript 4.0.

The caller supplies *current-slot reserved* directed-link rates and electrical
energy per bit. This module does not inspect future links, choose task traffic,
or debit a battery. Its receipt is the actual traffic/energy reservation for
the execution certificate. Initial C(0) is installed without a transfer charge.

Call ``begin_slot(t)`` before observing installed services, ``boundary(C)`` at
an LT boundary, and ``advance_slot(t, dt, links, energy_budget_j=...)`` once per
ST slot. ``advance_slot`` calls ``begin_slot`` if necessary. Completion during
t is held until ``begin_slot(t + 1)``. A whole image must arrive before it is
forwarded, but a later hop may use the remaining time in the same ST slot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
from numbers import Integral
from typing import Any, Callable, Mapping, Optional, Sequence


Matrix = tuple[tuple[int, ...], ...]
ImageKey = tuple[int, int]
RouteCallback = Callable[[str, Mapping[str, "DeploymentLink40"]], Optional[Sequence[str]]]
_STATE_SCHEMA = "at2hjo.deployment40.v1"


def _nonnegative(value: Any, name: str, *, positive: bool = False,
                 unbounded: bool = False) -> float:
    result = float(value)
    if math.isnan(result) or result < 0 or (positive and result <= 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    if not math.isfinite(result) and not (unbounded and result == math.inf):
        raise ValueError(f"{name} must be finite")
    return result


def _slot(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("st_index must be a nonnegative integer")
    return int(value)


def _mask(values: Any, shape: tuple[int, int]) -> Matrix:
    try:
        rows = tuple(tuple(row) for row in values)
    except TypeError as exc:
        raise ValueError("Deployment mask must be a binary matrix") from exc
    if len(rows) != shape[0] or any(len(row) != shape[1] for row in rows):
        raise ValueError("Deployment mask has the wrong shape")
    if any(value not in (0, 1) for row in rows for value in row):
        raise ValueError("Deployment mask must be binary")
    return tuple(tuple(int(value) for value in row) for row in rows)


@dataclass(frozen=True)
class DeploymentLink40:
    """A reserved directed link, identified by its key in the supplied mapping.

    tx_j_bit/rx_j_bit include active circuit power divided by active rate.
    Gateway coefficients may be zero if gateway energy is outside the model.
    """

    src: str
    dst: str
    rate_bps: float
    tx_j_bit: float = 0.0
    rx_j_bit: float = 0.0
    available: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.src, str) or not self.src or not isinstance(self.dst, str) or not self.dst:
            raise ValueError("Link endpoints must be nonempty strings")
        if self.src == self.dst:
            raise ValueError("A deployment link needs distinct endpoints")
        for name in ("rate_bps", "tx_j_bit", "rx_j_bit"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))


@dataclass(frozen=True)
class DeploymentSlot40:
    st_index: int
    dt_s: float
    installed_at_start: Matrix
    pending_residence_s: float
    pending_residence_by_node_s: tuple[float, ...]
    bits_by_link: Mapping[str, float]
    written_bits_by_image: Mapping[ImageKey, float]
    communication_energy_by_node_j: Mapping[str, float]
    write_energy_by_node_j: Mapping[str, float]
    completed_images: tuple[ImageKey, ...]
    events: tuple[Mapping[str, Any], ...] = ()

    @property
    def actual_bits_by_link(self) -> Mapping[str, float]:
        return self.bits_by_link

    @property
    def energy_by_node_j(self) -> dict[str, float]:
        nodes = self.communication_energy_by_node_j.keys() | self.write_energy_by_node_j.keys()
        return {node: self.communication_energy_by_node_j.get(node, 0.0)
                + self.write_energy_by_node_j.get(node, 0.0) for node in sorted(nodes)}

    @property
    def total_energy_j(self) -> float:
        return math.fsum(self.energy_by_node_j.values())

    def to_dict(self) -> dict[str, Any]:
        return {"st_index": self.st_index, "dt_s": self.dt_s,
                "installed_at_start": [list(row) for row in self.installed_at_start],
                "pending_residence_s": self.pending_residence_s,
                "pending_residence_by_node_s": list(self.pending_residence_by_node_s),
                "actual_bits_by_link": dict(self.bits_by_link),
                "written_bits_by_image": [{"node_index": n, "service_index": k, "bits": bits}
                                          for (n, k), bits in sorted(self.written_bits_by_image.items())],
                "communication_energy_by_node_j": dict(self.communication_energy_by_node_j),
                "write_energy_by_node_j": dict(self.write_energy_by_node_j),
                "energy_by_node_j": self.energy_by_node_j, "total_energy_j": self.total_energy_j,
                "completed_images": [list(key) for key in self.completed_images],
                "events": [dict(event) for event in self.events]}


@dataclass
class _Image:
    destination_index: int
    service_index: int
    size_bits: float
    route_link_ids: tuple[str, ...] = ()
    route_nodes: tuple[str, ...] = ()
    hop_index: int = 0
    residual_bits: float = 0.0
    installation_bits: float = 0.0
    buffer_nodes: set[str] = field(default_factory=set)
    completed_slot: int | None = None

    @property
    def key(self) -> ImageKey:
        return self.destination_index, self.service_index

    def to_dict(self) -> dict[str, Any]:
        return {"destination_index": self.destination_index, "service_index": self.service_index,
                "size_bits": self.size_bits, "route_link_ids": list(self.route_link_ids),
                "route_nodes": list(self.route_nodes), "hop_index": self.hop_index,
                "residual_bits": self.residual_bits, "installation_bits": self.installation_bits,
                "buffer_nodes": sorted(self.buffer_nodes), "completed_slot": self.completed_slot}


class DeploymentEngine40:
    """Persistent FIFO deployment queues, relay ownership, and actual-bit costs.

    Each selected image reserves its full destination storage. Relay capacity is
    a separate finite, preallocated buffer: an image reserves its full size
    before receiving its first bit, and releases a relay only after its outgoing
    hop finishes. A blocked FIFO head cannot be overtaken. Existing routes stay
    fixed on outages; a source without a route retries using only current links.

    Nodes missing from ``relay_buffer_capacity_bits`` have no relay space. Write
    rates default to unbounded for small fixtures; production callers should
    supply physical destination rates. ``energy_budget_j`` is the affordable
    deployment budget after idle/other mandatory energy has been reserved.
    """

    def __init__(self, node_ids: Sequence[str], service_sizes_bits: Sequence[float],
                 storage_capacity_bits: Sequence[float], *, initial_installed: Any = None,
                 gateway_ids: Sequence[str] = ("GS:0",),
                 relay_buffer_capacity_bits: Mapping[str, float] | None = None,
                 write_rate_bps: Mapping[str, float] | None = None,
                 write_energy_j_bit: Mapping[str, float] | None = None,
                 node_order: Sequence[str] | None = None):
        self.node_ids = tuple(node_ids)
        if not self.node_ids or len(set(self.node_ids)) != len(self.node_ids) or any(
                not isinstance(node, str) or not node for node in self.node_ids):
            raise ValueError("node_ids must be nonempty unique strings")
        self.gateway_ids = tuple(gateway_ids)
        if not self.gateway_ids or len(set(self.gateway_ids)) != len(self.gateway_ids) or any(
                not isinstance(node, str) or not node for node in self.gateway_ids):
            raise ValueError("gateway_ids must be nonempty unique strings")
        if set(self.gateway_ids) & set(self.node_ids):
            raise ValueError("Gateway nodes cannot be deployment destinations")
        self.service_sizes_bits = tuple(_nonnegative(size, "service size", positive=True)
                                        for size in service_sizes_bits)
        if not self.service_sizes_bits:
            raise ValueError("At least one service is required")
        self.storage_capacity_bits = tuple(_nonnegative(size, "storage capacity")
                                           for size in storage_capacity_bits)
        if len(self.storage_capacity_bits) != len(self.node_ids):
            raise ValueError("Storage capacities have the wrong shape")
        self.relay_buffer_capacity_bits = {str(node): _nonnegative(value, "relay buffer capacity")
                                           for node, value in (relay_buffer_capacity_bits or {}).items()}
        self.write_rate_bps = {node: _nonnegative((write_rate_bps or {}).get(node, math.inf),
                                                "write rate", unbounded=True) for node in self.node_ids}
        self.write_energy_j_bit = {node: _nonnegative((write_energy_j_bit or {}).get(node, 0.0),
                                                    "write energy") for node in self.node_ids}
        self.node_order = tuple(node_order) if node_order is not None else self.gateway_ids + self.node_ids
        if len(set(self.node_order)) != len(self.node_order):
            raise ValueError("node_order must contain unique IDs")
        self._node_rank = {node: i for i, node in enumerate(self.node_order)}
        self._shape = len(self.node_ids), len(self.service_sizes_bits)
        empty = tuple((0,) * self._shape[1] for _ in self.node_ids)
        self._installed = self._validate_target(empty if initial_installed is None else initial_installed)
        self._target = self._installed
        self._images: dict[ImageKey, _Image] = {}
        self._queues: dict[str, list[ImageKey]] = {}
        self._current_slot: int | None = None
        self._last_advanced_slot: int | None = None
        self.cumulative_pending_residence_s = 0.0
        self.cumulative_communication_energy_by_node_j: dict[str, float] = {}
        self.cumulative_write_energy_by_node_j: dict[str, float] = {}

    @property
    def target(self) -> Matrix:
        return self._target

    @property
    def installed(self) -> Matrix:
        return self._installed

    @property
    def pending_transfers(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._images[key].to_dict() for key in sorted(self._images))

    @property
    def link_queues(self) -> dict[str, tuple[ImageKey, ...]]:
        return {identifier: tuple(queue) for identifier, queue in sorted(self._queues.items()) if queue}

    @property
    def relay_buffer_ownership(self) -> dict[str, tuple[ImageKey, ...]]:
        owners: dict[str, list[ImageKey]] = {}
        for key, image in sorted(self._images.items()):
            for node in sorted(image.buffer_nodes):
                owners.setdefault(node, []).append(key)
        return {node: tuple(keys) for node, keys in sorted(owners.items())}

    @property
    def relay_buffer_occupancy_bits(self) -> dict[str, float]:
        owners = self.relay_buffer_ownership
        return {node: math.fsum(self._images[key].size_bits for key in owners.get(node, ()))
                for node in sorted(self.relay_buffer_capacity_bits.keys() | owners.keys())}

    @property
    def reserved_storage_bits(self) -> tuple[float, ...]:
        return tuple(math.fsum(bit * size for bit, size in zip(row, self.service_sizes_bits))
                     for row in self._target)

    def _validate_target(self, values: Any) -> Matrix:
        result = _mask(values, self._shape)
        if any(math.fsum(bit * size for bit, size in zip(row, self.service_sizes_bits)) > capacity
               for row, capacity in zip(result, self.storage_capacity_bits)):
            raise ValueError("storage_capacity_exceeded")
        return result

    def boundary(self, target: Any) -> dict[str, Any]:
        """Apply a storage-feasible LT target without resetting retained progress."""
        after = self._validate_target(target)
        before = self._target
        canceled = tuple(key for key in sorted(self._images) if not after[key[0]][key[1]])
        for key in canceled:
            del self._images[key]  # drops all of this image's relay ownership
        canceled_set = set(canceled)
        self._queues = {identifier: [key for key in queue if key not in canceled_set]
                        for identifier, queue in self._queues.items()}
        self._queues = {identifier: queue for identifier, queue in self._queues.items() if queue}
        self._installed = tuple(tuple(int(bit and after[n][k]) for k, bit in enumerate(row))
                                for n, row in enumerate(self._installed))
        self._target = after
        added = []
        for n, row in enumerate(after):
            for k, selected in enumerate(row):
                if selected and not self._installed[n][k] and (n, k) not in self._images:
                    size = self.service_sizes_bits[k]
                    self._images[n, k] = _Image(n, k, size, residual_bits=size)
                    added.append((n, k))
        return {"previous_target": before, "target": after, "enqueued_images": tuple(added),
                "canceled_images": canceled, "reserved_storage_bits": self.reserved_storage_bits}

    def begin_slot(self, st_index: int) -> Matrix:
        """Make prior-slot completions usable before current-slot observations."""
        index = _slot(st_index)
        if self._current_slot == index:
            return self._installed
        if self._current_slot is not None and self._last_advanced_slot != self._current_slot:
            raise ValueError("Cannot start another slot before advancing the current slot")
        if self._last_advanced_slot is not None and index != self._last_advanced_slot + 1:
            raise ValueError("Deployment ST slots must be consecutive")
        rows = [list(row) for row in self._installed]
        for key, image in list(self._images.items()):
            if image.completed_slot is not None and image.completed_slot < index:
                n, k = key
                rows[n][k] = self._target[n][k]
                del self._images[key]
        self._installed = tuple(tuple(row) for row in rows)
        self._current_slot = index
        return self._installed

    def _order(self, node: str) -> tuple[int, str]:
        return self._node_rank.get(node, len(self._node_rank)), node

    def shortest_route(self, destination: str, links: Mapping[str, DeploymentLink40]) -> tuple[str, ...] | None:
        """Minimum current hop count, then lexicographic node-index route."""
        outgoing: dict[str, list[tuple[str, DeploymentLink40]]] = {}
        for identifier, link in links.items():
            if link.available and link.rate_bps > 0:
                outgoing.setdefault(link.src, []).append((identifier, link))
        heap = [(0, (self._order(source),), (), source, (source,)) for source in self.gateway_ids]
        heapq.heapify(heap)
        visited: set[str] = set()
        while heap:
            hops, order, route, node, path = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            if node == destination:
                return route
            for identifier, link in outgoing.get(node, ()):
                if link.dst not in path:
                    heapq.heappush(heap, (hops + 1, order + (self._order(link.dst),),
                                          route + (identifier,), link.dst, path + (link.dst,)))
        return None

    def _route_waiting(self, links: Mapping[str, DeploymentLink40], callback: RouteCallback | None) -> None:
        arrivals: dict[str, list[ImageKey]] = {}
        assignments: list[tuple[_Image, tuple[str, ...], tuple[str, ...]]] = []
        for key, image in sorted(self._images.items()):
            if image.route_link_ids or image.completed_slot is not None:
                continue
            destination = self.node_ids[image.destination_index]
            selected = callback(destination, links) if callback is not None else self.shortest_route(destination, links)
            if selected is None:
                continue
            route = tuple(selected)
            if not route:
                raise ValueError("A gateway-to-destination route must contain a link")
            if any(identifier not in links for identifier in route):
                raise ValueError("Route callback selected a link outside the current graph")
            first = links[route[0]]
            nodes = [first.src]
            for identifier in route:
                link = links[identifier]
                if link.src != nodes[-1] or not link.available or link.rate_bps <= 0:
                    raise ValueError("Route callback selected an unavailable or discontinuous route")
                nodes.append(link.dst)
            if nodes[0] not in self.gateway_ids or nodes[-1] != destination or len(set(nodes)) != len(nodes):
                raise ValueError("Route callback must return a simple gateway-to-destination path")
            assignments.append((image, route, tuple(nodes)))
            arrivals.setdefault(route[0], []).append(key)
        for image, route, nodes in assignments:
            image.route_link_ids, image.route_nodes = route, nodes
        for identifier, keys in sorted(arrivals.items()):
            self._queues.setdefault(identifier, []).extend(sorted(keys))

    def advance_slot(self, st_index: int, dt_s: float, links: Mapping[str, DeploymentLink40], *,
                     energy_budget_j: Mapping[str, float] | None = None,
                     route_callback: RouteCallback | None = None) -> DeploymentSlot40:
        """Advance affordable actual traffic, retaining queues when progress pauses.

        Different directed links operate simultaneously. Events are whole-hop
        completions, slot end, or exhaustion of a node's aggregate bit-energy
        budget. Final-hop rates share the destination's write throughput. This
        computes a reservation from actual traffic, including every relay TX/RX.
        """
        index, duration = _slot(st_index), _nonnegative(dt_s, "dt_s", positive=True)
        current_links = dict(links)
        if any(not isinstance(identifier, str) or not identifier or not isinstance(link, DeploymentLink40)
               for identifier, link in current_links.items()):
            raise ValueError("links must map nonempty physics IDs to DeploymentLink40 objects")
        budgets = {str(node): _nonnegative(value, "energy budget", unbounded=True)
                   for node, value in (energy_budget_j or {}).items()}
        if self._last_advanced_slot == index:
            raise ValueError("A deployment slot can only be advanced once")
        for image in self._images.values():
            for h, identifier in enumerate(image.route_link_ids):
                link = current_links.get(identifier)
                if link is not None and (link.src, link.dst) != image.route_nodes[h:h + 2]:
                    raise ValueError("A persistent physics link ID changed its endpoints")
        installed = self.begin_slot(index)
        self._route_waiting(current_links, route_callback)
        residence_by_node = tuple(duration * sum(selected and not available for selected, available in zip(target, ready))
                                  for target, ready in zip(self._target, installed))
        carried: dict[str, float] = {}
        written: dict[ImageKey, float] = {}
        communication: dict[str, float] = {}
        write_energy: dict[str, float] = {}
        completed: list[ImageKey] = []
        events: list[dict[str, Any]] = []
        elapsed = 0.0

        def remaining_budget(node: str) -> float:
            return max(0.0, budgets.get(node, math.inf) - communication.get(node, 0.0) - write_energy.get(node, 0.0))

        while elapsed < duration:
            heads = []
            for identifier, queue in self._queues.items():
                link = current_links.get(identifier)
                if queue and link is not None and link.available and link.rate_bps > 0:
                    image = self._images[queue[0]]
                    final = image.hop_index == len(image.route_link_ids) - 1
                    coefficients = [(link.src, link.tx_j_bit), (link.dst, link.rx_j_bit)]
                    if final:
                        coefficients.append((link.dst, self.write_energy_j_bit[link.dst]))
                    if any(coefficient > 0 and remaining_budget(node) <= 0 for node, coefficient in coefficients):
                        continue
                    if final and self.write_rate_bps[link.dst] <= 0:
                        continue
                    heads.append((image.key, identifier, image, link, final))
            heads.sort(key=lambda head: (head[0], head[1]))
            rates: dict[str, float] = {}
            occupancy = self.relay_buffer_occupancy_bits
            # A full receiver buffer is reserved before the first incoming bit.
            # Cross-link contenders use the same (destination, service) tie rule.
            for _, identifier, image, link, final in sorted(heads):
                if not final and link.dst not in image.buffer_nodes:
                    capacity = self.relay_buffer_capacity_bits.get(link.dst, 0.0)
                    used = occupancy.get(link.dst, 0.0)
                    if used + image.size_bits > capacity:
                        continue
                    image.buffer_nodes.add(link.dst)
                    occupancy[link.dst] = used + image.size_bits
                rates[identifier] = link.rate_bps
            if not rates:
                break
            writer_groups: dict[str, list[str]] = {}
            for _, identifier, _, link, final in heads:
                if identifier in rates and final:
                    writer_groups.setdefault(link.dst, []).append(identifier)
            for node, identifiers in writer_groups.items():
                total = math.fsum(rates[identifier] for identifier in identifiers)
                factor = min(1.0, self.write_rate_bps[node] / total)
                for identifier in identifiers:
                    rates[identifier] *= factor
            power: dict[str, float] = {}
            advance = duration - elapsed
            for _, identifier, image, link, final in heads:
                rate = rates.get(identifier, 0.0)
                if rate <= 0:
                    continue
                advance = min(advance, image.residual_bits / rate)
                power[link.src] = power.get(link.src, 0.0) + rate * link.tx_j_bit
                power[link.dst] = power.get(link.dst, 0.0) + rate * link.rx_j_bit
                if final:
                    power[link.dst] += rate * self.write_energy_j_bit[link.dst]
            for node, watts in power.items():
                if watts > 0:
                    advance = min(advance, remaining_budget(node) / watts)
            if advance <= 0 or elapsed + advance == elapsed:
                break
            finished = []
            for _, identifier, image, link, final in heads:
                rate = rates.get(identifier, 0.0)
                if rate <= 0:
                    continue
                bits = min(image.residual_bits, rate * advance)
                image.residual_bits = max(0.0, image.residual_bits - bits)
                carried[identifier] = carried.get(identifier, 0.0) + bits
                communication[link.src] = communication.get(link.src, 0.0) + bits * link.tx_j_bit
                communication[link.dst] = communication.get(link.dst, 0.0) + bits * link.rx_j_bit
                if final:
                    image.installation_bits = min(image.size_bits, image.installation_bits + bits)
                    written[image.key] = written.get(image.key, 0.0) + bits
                    write_energy[link.dst] = write_energy.get(link.dst, 0.0) + bits * self.write_energy_j_bit[link.dst]
                if image.residual_bits <= 1e-12 * max(1.0, image.size_bits):
                    # Do not fabricate or charge residual bits at roundoff scale.
                    image.residual_bits = 0.0
                    finished.append((image.key, identifier, image, link, final))
                events.append({"node_index": image.destination_index, "service_index": image.service_index,
                               "link_id": identifier, "src": link.src, "dst": link.dst,
                               "hop_index": image.hop_index, "bits": bits,
                               "start_s": elapsed, "end_s": elapsed + advance,
                               "final_hop": final, "hop_complete": image.residual_bits == 0.0})
            elapsed += advance
            arrivals = {}
            for _, identifier, image, link, final in sorted(finished):
                self._queues[identifier].pop(0)
                image.buffer_nodes.discard(link.src)
                image.hop_index += 1
                if final:
                    image.completed_slot = index
                    image.buffer_nodes.clear()
                    completed.append(image.key)
                else:
                    image.residual_bits = image.size_bits
                    next_link = image.route_link_ids[image.hop_index]
                    arrivals.setdefault(next_link, []).append(image.key)
            for identifier, keys in sorted(arrivals.items()):
                self._queues.setdefault(identifier, []).extend(sorted(keys))
            # Consume the remaining budget exactly when its event is reached,
            # preventing a numerical residual from causing a zero-time loop.
            for node, watts in power.items():
                remaining = remaining_budget(node)
                budget = budgets.get(node, math.inf)
                if watts > 0 and math.isfinite(budget) and remaining <= 1e-12 * max(1.0, budget):
                    budgets[node] = communication.get(node, 0.0) + write_energy.get(node, 0.0)
        self._queues = {identifier: queue for identifier, queue in self._queues.items() if queue}
        self._last_advanced_slot = index
        residence = math.fsum(residence_by_node)
        self.cumulative_pending_residence_s += residence
        for ledger, charges in ((self.cumulative_communication_energy_by_node_j, communication),
                                (self.cumulative_write_energy_by_node_j, write_energy)):
            for node, charge in charges.items():
                ledger[node] = ledger.get(node, 0.0) + charge
        return DeploymentSlot40(index, duration, installed, residence, residence_by_node,
                                dict(sorted(carried.items())), dict(sorted(written.items())),
                                dict(sorted(communication.items())), dict(sorted(write_energy.items())),
                                tuple(sorted(completed)), tuple(events))

    def _configuration(self) -> dict[str, Any]:
        return {"node_ids": list(self.node_ids), "service_sizes_bits": list(self.service_sizes_bits),
                "storage_capacity_bits": list(self.storage_capacity_bits), "gateway_ids": list(self.gateway_ids),
                "relay_buffer_capacity_bits": dict(self.relay_buffer_capacity_bits),
                "write_rate_bps": {node: rate if math.isfinite(rate) else None for node, rate in self.write_rate_bps.items()},
                "write_energy_j_bit": dict(self.write_energy_j_bit), "node_order": list(self.node_order)}

    def state_dict(self) -> dict[str, Any]:
        """JSON-safe checkpoint including all FIFO and relay/installation state."""
        return {"schema": _STATE_SCHEMA, "configuration": self._configuration(),
                "target": [list(row) for row in self._target], "installed": [list(row) for row in self._installed],
                "current_slot": self._current_slot, "last_advanced_slot": self._last_advanced_slot,
                "pending_transfers": list(self.pending_transfers),
                "link_queues": {identifier: [list(key) for key in queue] for identifier, queue in self.link_queues.items()},
                "relay_buffer_ownership": {node: [list(key) for key in keys] for node, keys in self.relay_buffer_ownership.items()},
                "relay_buffer_occupancy_bits": self.relay_buffer_occupancy_bits,
                "cumulative_pending_residence_s": self.cumulative_pending_residence_s,
                "cumulative_communication_energy_by_node_j": dict(self.cumulative_communication_energy_by_node_j),
                "cumulative_write_energy_by_node_j": dict(self.cumulative_write_energy_by_node_j)}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "DeploymentEngine40":
        config = dict(state["configuration"])
        config["write_rate_bps"] = {node: math.inf if value is None else value
                                    for node, value in config["write_rate_bps"].items()}
        engine = cls(**config)
        engine.load_state_dict(state)
        return engine

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a matching engine; reject incomplete/corrupt queue ownership."""
        if state.get("schema") != _STATE_SCHEMA or state.get("configuration") != self._configuration():
            raise ValueError("Deployment checkpoint configuration/schema differs")
        target, installed = self._validate_target(state["target"]), self._validate_target(state["installed"])
        if any(ready > selected for a, b in zip(installed, target) for ready, selected in zip(a, b)):
            raise ValueError("Installed image is excluded from target")
        images = {}
        for raw in state["pending_transfers"]:
            n, k = raw["destination_index"], raw["service_index"]
            if not isinstance(n, int) or not isinstance(k, int) or not 0 <= n < self._shape[0] or not 0 <= k < self._shape[1]:
                raise ValueError("Invalid checkpoint image index")
            if (n, k) in images or not target[n][k] or installed[n][k]:
                raise ValueError("Invalid or duplicate pending image")
            image = _Image(n, k, _nonnegative(raw["size_bits"], "image size", positive=True),
                           tuple(raw["route_link_ids"]), tuple(raw["route_nodes"]), int(raw["hop_index"]),
                           _nonnegative(raw["residual_bits"], "residual bits"),
                           _nonnegative(raw["installation_bits"], "installation bits"),
                           set(raw["buffer_nodes"]), raw["completed_slot"])
            if image.size_bits != self.service_sizes_bits[k] or image.residual_bits > image.size_bits or image.installation_bits > image.size_bits:
                raise ValueError("Invalid checkpoint image progress")
            if not image.route_link_ids:
                if image.route_nodes or image.hop_index or image.buffer_nodes or image.completed_slot is not None or image.residual_bits != image.size_bits or image.installation_bits:
                    raise ValueError("Invalid unrouted checkpoint image")
            else:
                h, length = image.hop_index, len(image.route_link_ids)
                if len(image.route_nodes) != length + 1 or image.route_nodes[0] not in self.gateway_ids or image.route_nodes[-1] != self.node_ids[n] or len(set(image.route_nodes)) != length + 1 or not 0 <= h <= length:
                    raise ValueError("Invalid checkpoint route")
                if image.completed_slot is not None:
                    _slot(image.completed_slot)
                    if h != length or image.residual_bits or image.buffer_nodes or not math.isclose(image.installation_bits, image.size_bits):
                        raise ValueError("Invalid completed checkpoint image")
                elif h == length or image.residual_bits <= 0:
                    raise ValueError("Invalid in-flight checkpoint image")
                else:
                    required = {image.route_nodes[h]} if h > 0 else set()
                    allowed = required | ({image.route_nodes[h + 1]} if h < length - 1 else set())
                    if not required <= image.buffer_nodes <= allowed:
                        raise ValueError("Invalid checkpoint relay ownership")
                    if image.residual_bits < image.size_bits and h < length - 1 and image.route_nodes[h + 1] not in image.buffer_nodes:
                        raise ValueError("Partial incoming image lacks relay ownership")
                    progress_valid = (math.isclose(image.installation_bits + image.residual_bits,
                                                   image.size_bits, rel_tol=1e-12, abs_tol=1e-9)
                                      if h == length - 1 else image.installation_bits == 0.0)
                    if not progress_valid:
                        raise ValueError("Invalid checkpoint installation progress")
            images[n, k] = image
        if set(images) != {(n, k) for n, row in enumerate(target) for k, selected in enumerate(row) if selected and not installed[n][k]}:
            raise ValueError("Checkpoint is missing a pending selected image")
        queues = {identifier: [tuple(key) for key in raw] for identifier, raw in state["link_queues"].items()}
        seen = []
        for identifier, queue in queues.items():
            for key in queue:
                image = images.get(key)
                if image is None or image.completed_slot is not None or not image.route_link_ids or image.route_link_ids[image.hop_index] != identifier:
                    raise ValueError("Checkpoint queue does not match current hop")
                seen.append(key)
        expected = {key for key, image in images.items() if image.route_link_ids and image.completed_slot is None}
        if len(set(seen)) != len(seen) or set(seen) != expected:
            raise ValueError("Checkpoint FIFO membership is incomplete or duplicated")
        owners: dict[str, list[ImageKey]] = {}
        for key, image in sorted(images.items()):
            for node in image.buffer_nodes:
                owners.setdefault(node, []).append(key)
        occupancy = {node: math.fsum(images[key].size_bits for key in owners.get(node, ()))
                     for node in sorted(self.relay_buffer_capacity_bits.keys() | owners.keys())}
        if any(used > self.relay_buffer_capacity_bits.get(node, 0.0) for node, used in occupancy.items()):
            raise ValueError("Checkpoint relay buffer exceeds capacity")
        serialized_owners = {node: [list(key) for key in keys] for node, keys in sorted(owners.items())}
        if state["relay_buffer_ownership"] != serialized_owners or state["relay_buffer_occupancy_bits"] != occupancy:
            raise ValueError("Checkpoint relay ownership/occupancy is inconsistent")
        current = None if state["current_slot"] is None else _slot(state["current_slot"])
        last = None if state["last_advanced_slot"] is None else _slot(state["last_advanced_slot"])
        if last is not None and current not in (last, last + 1):
            raise ValueError("Invalid checkpoint slot clock")
        if any(image.completed_slot is not None and (last is None or image.completed_slot != last)
               for image in images.values()):
            raise ValueError("Invalid checkpoint visibility clock")
        residence = _nonnegative(state["cumulative_pending_residence_s"], "pending residence")
        communication = {node: _nonnegative(value, "communication energy")
                         for node, value in state["cumulative_communication_energy_by_node_j"].items()}
        writes = {node: _nonnegative(value, "write energy")
                  for node, value in state["cumulative_write_energy_by_node_j"].items()}
        self._target, self._installed, self._images, self._queues = target, installed, images, queues
        self._current_slot, self._last_advanced_slot = current, last
        self.cumulative_pending_residence_s = residence
        self.cumulative_communication_energy_by_node_j = communication
        self.cumulative_write_energy_by_node_j = writes
