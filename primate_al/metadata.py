import os
import re
import json
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple


class MetadataManager:
    """
    Manages camera-trap metadata (site, timestamp, burst events, and cost proxies).
    Enforces clean fallbacks without inventing missing data.
    """
    def __init__(self, index_json_path: str, burst_interval_sec: float = 60.0):
        self.index_json_path = index_json_path
        self.burst_interval_sec = burst_interval_sec
        self.items: Dict[int, Dict[str, Any]] = {}
        self.site_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.event_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.index_to_event: Dict[int, str] = {}
        self._load_and_process()

    def _load_and_process(self):
        with open(self.index_json_path, "r") as f:
            data = json.load(f)
            
        images = data.get("images", [])
        
        # Sort images by location and timestamp for temporal burst grouping
        # Handle missing timestamps gracefully by grouping into subfolder clusters
        loc_groups = defaultdict(list)
        
        for img in images:
            idx = int(img["idx"])
            self.items[idx] = img
            loc = str(img.get("location", "UNKNOWN"))
            self.site_to_indices[loc].append(idx)
            
            dt_str = img.get("datetime")
            dt_val = None
            if dt_str:
                try:
                    dt_val = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    dt_val = None
                    
            folder = os.path.dirname(img.get("file_name", ""))
            loc_groups[loc].append({
                "idx": idx,
                "datetime": dt_val,
                "folder": folder,
                "file_name": img.get("file_name", "")
            })

        # Process burst events per location
        event_counter = 0
        for loc, group in loc_groups.items():
            # Separate images with valid datetimes and missing datetimes
            with_dt = [item for item in group if item["datetime"] is not None]
            without_dt = [item for item in group if item["datetime"] is None]
            
            # Sort chronological items
            with_dt.sort(key=lambda x: x["datetime"])
            current_event_id = None
            last_dt = None
            
            for item in with_dt:
                idx = item["idx"]
                dt = item["datetime"]
                if (last_dt is None) or ((dt - last_dt).total_seconds() > self.burst_interval_sec):
                    event_counter += 1
                    current_event_id = f"evt_{loc}_{event_counter:06d}"
                    
                self.index_to_event[idx] = current_event_id
                self.event_to_indices[current_event_id].append(idx)
                last_dt = dt
                
            # For images without datetime, group by subfolder and extract sequence number if present
            folder_groups = defaultdict(list)
            for item in without_dt:
                folder_groups[item["folder"]].append(item)
                
            for folder, f_items in folder_groups.items():
                event_counter += 1
                folder_event_id = f"evt_{loc}_nodt_{event_counter:06d}"
                for item in f_items:
                    idx = item["idx"]
                    self.index_to_event[idx] = folder_event_id
                    self.event_to_indices[folder_event_id].append(idx)

    def get_site(self, idx: int) -> str:
        return self.items.get(idx, {}).get("location", "UNKNOWN")

    def get_event(self, idx: int) -> str:
        return self.index_to_event.get(idx, f"evt_single_{idx}")

    def compute_cost_proxy(
        self,
        num_boxes: int,
        crowding: float = 0.0,
        cost_base: float = 1.0,
        cost_box_weight: float = 0.5,
        cost_crowd_weight: float = 0.25,
        crowding_overlap: float = 0.0,
        overlap_only: bool = False
    ) -> float:
        """
        Computes annotation cost proxy:
        If overlap_only is True:
            cost = base + crowd_weight * crowding_overlap
            (Does not penalize distinct multiple animals; only penalizes occluding overlaps)
        Else:
            cost = base + box_weight * num_boxes + crowd_weight * crowding
        Strictly a proxy based on object count and crowding; not human clock timing.
        """
        if overlap_only:
            eff_crowd = crowding_overlap if crowding_overlap > 0.0 else crowding
            return float(cost_base + cost_crowd_weight * eff_crowd)
        return float(cost_base + cost_box_weight * num_boxes + cost_crowd_weight * crowding)

