from datetime import datetime
import pandas as pd


def add_event(events: list, category: str, message: str):
    events.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "category": category,
        "message": message,
    })


def events_to_frame(events: list) -> pd.DataFrame:
    return pd.DataFrame(events, columns=["time", "category", "message"])
