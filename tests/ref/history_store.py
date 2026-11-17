import numpy as np
import pandas as pd


class HistoryStore:
    """Earlier titles per IMDb person id (name id), stored as sorted arrays for fast lookups."""

    def __init__(self, df: pd.DataFrame | None):
        if df is None or df.empty:
            self.names = np.array([], dtype="U10"); self.year = np.array([]); self.rating = np.array([]); self.votes = np.array([])
            self.rows = 0
            return
        df = df.sort_values("nconst", kind="stable")
        self.names = df["nconst"].to_numpy().astype("U10")
        self.year = df["year"].to_numpy(dtype="int32")
        self.rating = df["rating"].to_numpy(dtype="float32")
        self.votes = df["votes"].to_numpy(dtype="int64")
        self.rows = len(df)

    def person(self, nm: str):
        lo = np.searchsorted(self.names, nm, "left"); hi = np.searchsorted(self.names, nm, "right")
        return self.year[lo:hi], self.rating[lo:hi], self.votes[lo:hi]


