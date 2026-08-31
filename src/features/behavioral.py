"""Calculate exact past-only destination behavioural features."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


TRANSACTION_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
HISTORY_FEATURE_NAMES = (
    "dest_tx_count_log1p_24h",
    "dest_tx_count_log1p_168h",
    "dest_amount_sum_log1p_24h",
    "dest_amount_sum_log1p_168h",
    "dest_unique_origins_log1p_24h",
    "dest_unique_origins_log1p_168h",
    "dest_hours_since_previous_activity",
    "dest_history_available",
    "amount_log_ratio_to_dest_mean_168h",
    "dest_cash_in_count_log1p_24h",
    "dest_cash_out_count_log1p_24h",
    "dest_debit_count_log1p_24h",
    "dest_payment_count_log1p_24h",
    "dest_transfer_count_log1p_24h",
)


@dataclass
class _RollingWindow:
    """Bound one destination's exact rolling state to its active time window."""

    events: deque[tuple[int, str, float, str]] = field(default_factory=deque)
    origins: Counter[str] = field(default_factory=Counter)
    transaction_types: Counter[str] = field(default_factory=Counter)
    amount_sum: float = 0.0

    def expire(self, earliest_step: int) -> None:
        """Remove events older than the reviewed inclusive window boundary."""
        while self.events and self.events[0][0] < earliest_step:
            _, origin, amount, transaction_type = self.events.popleft()
            self.amount_sum -= amount
            self.transaction_types[transaction_type] -= 1
            self.origins[origin] -= 1
            if self.origins[origin] == 0:
                del self.origins[origin]

    def add(self, event: tuple[int, str, float, str]) -> None:
        """Add one event after its entire PaySim step has been evaluated."""
        self.events.append(event)
        self.origins[event[1]] += 1
        self.transaction_types[event[3]] += 1
        self.amount_sum += event[2]


def calculate_destination_history(destination_rows: pd.DataFrame) -> pd.DataFrame:
    """Return reviewed features for one destination in source-row order.

    All rows sharing a PaySim step are evaluated before state is updated. This is
    essential because PaySim has hourly steps and provides no ordering within an hour.
    """
    required = {"source_row_number", "step", "name_orig", "amount", "transaction_type"}
    missing = required - set(destination_rows)
    if missing:
        raise ValueError(f"Destination history input is missing: {sorted(missing)}")
    frame = destination_rows.sort_values(["step", "source_row_number"], kind="mergesort")
    rows = len(frame)
    output = {name: np.zeros(rows, dtype="float32") for name in HISTORY_FEATURE_NAMES}
    output["dest_history_available"] = np.zeros(rows, dtype="int8")

    window_24 = _RollingWindow()
    window_168 = _RollingWindow()
    last_step: int | None = None

    write_offset = 0
    for step, current in frame.groupby("step", sort=False):
        step = int(step)
        window_24.expire(step - 24)
        window_168.expire(step - 168)
        count_24 = len(window_24.events)
        count_168 = len(window_168.events)
        indexes = slice(write_offset, write_offset + len(current))
        output["dest_tx_count_log1p_24h"][indexes] = np.log1p(count_24)
        output["dest_tx_count_log1p_168h"][indexes] = np.log1p(count_168)
        output["dest_amount_sum_log1p_24h"][indexes] = np.log1p(window_24.amount_sum)
        output["dest_amount_sum_log1p_168h"][indexes] = np.log1p(window_168.amount_sum)
        output["dest_unique_origins_log1p_24h"][indexes] = np.log1p(len(window_24.origins))
        output["dest_unique_origins_log1p_168h"][indexes] = np.log1p(len(window_168.origins))
        output["dest_history_available"][indexes] = int(last_step is not None)
        output["dest_hours_since_previous_activity"][indexes] = (
            min(step - last_step, 744) if last_step is not None else 744
        )
        previous_mean = window_168.amount_sum / count_168 if count_168 else 0.0
        output["amount_log_ratio_to_dest_mean_168h"][indexes] = (
            np.log1p(current["amount"].to_numpy(dtype="float64"))
            - np.log1p(previous_mean)
            if count_168
            else 0.0
        )
        for transaction_type in TRANSACTION_TYPES:
            name = f"dest_{transaction_type.lower()}_count_log1p_24h"
            output[name][indexes] = np.log1p(
                window_24.transaction_types[transaction_type]
            )
        # Update only after every same-step feature has been calculated.
        for row in current.itertuples(index=False):
            event = (step, str(row.name_orig), float(row.amount), str(row.transaction_type))
            window_24.add(event)
            window_168.add(event)
        last_step = step
        write_offset += len(current)

    result = pd.DataFrame(output)
    result.insert(0, "source_row_number", frame["source_row_number"].to_numpy(dtype="int64"))
    return result.sort_values("source_row_number", kind="mergesort", ignore_index=True)


def calculate_bucket_history(bucket_rows: pd.DataFrame) -> pd.DataFrame:
    """Calculate many destination groups without constructing tiny DataFrames.

    Most PaySim destinations occur only once. Pre-filling their cold-start values and
    scanning only repeated groups keeps the 6.3-million-row build practical.
    """
    frame = bucket_rows.sort_values(
        ["name_dest", "step", "source_row_number"], kind="mergesort"
    ).reset_index(drop=True)
    rows = len(frame)
    output = {name: np.zeros(rows, dtype="float32") for name in HISTORY_FEATURE_NAMES}
    output["dest_history_available"] = np.zeros(rows, dtype="int8")
    output["dest_hours_since_previous_activity"] = np.full(rows, 744, dtype="float32")

    destinations = frame["name_dest"].astype(str).to_numpy()
    origins = frame["name_orig"].astype(str).to_numpy()
    steps = frame["step"].to_numpy(dtype="int64")
    amounts = frame["amount"].to_numpy(dtype="float64")
    transaction_types = frame["transaction_type"].astype(str).to_numpy()
    boundaries = np.r_[0, np.flatnonzero(destinations[1:] != destinations[:-1]) + 1, rows]

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start == 1:
            continue
        window_24 = _RollingWindow()
        window_168 = _RollingWindow()
        last_step: int | None = None
        position = int(start)
        while position < end:
            step = int(steps[position])
            after_step = position + 1
            while after_step < end and steps[after_step] == step:
                after_step += 1
            indexes = slice(position, after_step)
            window_24.expire(step - 24)
            window_168.expire(step - 168)
            count_24 = len(window_24.events)
            count_168 = len(window_168.events)
            output["dest_tx_count_log1p_24h"][indexes] = np.log1p(count_24)
            output["dest_tx_count_log1p_168h"][indexes] = np.log1p(count_168)
            output["dest_amount_sum_log1p_24h"][indexes] = np.log1p(window_24.amount_sum)
            output["dest_amount_sum_log1p_168h"][indexes] = np.log1p(window_168.amount_sum)
            output["dest_unique_origins_log1p_24h"][indexes] = np.log1p(len(window_24.origins))
            output["dest_unique_origins_log1p_168h"][indexes] = np.log1p(len(window_168.origins))
            output["dest_history_available"][indexes] = int(last_step is not None)
            output["dest_hours_since_previous_activity"][indexes] = (
                min(step - last_step, 744) if last_step is not None else 744
            )
            if count_168:
                previous_mean = window_168.amount_sum / count_168
                output["amount_log_ratio_to_dest_mean_168h"][indexes] = (
                    np.log1p(amounts[indexes]) - np.log1p(previous_mean)
                )
            for transaction_type in TRANSACTION_TYPES:
                output[f"dest_{transaction_type.lower()}_count_log1p_24h"][indexes] = (
                    np.log1p(window_24.transaction_types[transaction_type])
                )
            for index in range(position, after_step):
                event = (step, origins[index], float(amounts[index]), transaction_types[index])
                window_24.add(event)
                window_168.add(event)
            last_step = step
            position = after_step

    result = pd.DataFrame(output)
    result.insert(0, "source_row_number", frame["source_row_number"].to_numpy(dtype="int64"))
    return result.sort_values("source_row_number", kind="mergesort", ignore_index=True)
