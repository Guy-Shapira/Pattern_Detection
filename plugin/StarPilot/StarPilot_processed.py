from datetime import datetime, timedelta

from base.DataFormatter import DataFormatter, EventTypeClassifier
from misc.Utils import str_to_number

# COLUMN_KEYS = ["sid", "ts", "x", "y", "z", "vx", "vy", "vz"]
COLUMN_KEYS = ["ts", "sid", "x", "y", "vx", "vy", "health"]
TICKER_KEY = "sid"
EVENT_TIMESTAMP_KEY = "ts"


class ByTickerEventTypeClassifier(EventTypeClassifier):
    """
    This type classifier assigns a dedicated event type to each event in the log.
    """

    def get_event_type(self, event_payload: dict):
        """
        The type of a  event is equal to the ticker .
        """
        return event_payload[TICKER_KEY]


class DataFormatter(DataFormatter):
    def __init__(
        self, event_type_classifier: EventTypeClassifier = ByTickerEventTypeClassifier()
    ):
        super().__init__(event_type_classifier)

    def parse_event(self, raw_data: str):
        """
        Parses a formatted string into an event.
        """
        event_attributes = raw_data.replace("\n", "").split(",")
        for j in range(len(event_attributes)):
            event_attributes[j] = str_to_number(event_attributes[j])
        return dict(zip(COLUMN_KEYS, event_attributes))

    def get_event_timestamp(self, event_payload: dict):
        time_stamp = int(event_payload[EVENT_TIMESTAMP_KEY])
        secs = int(time_stamp // 15)
        milisecs = (time_stamp / 15 * 1000) % 1000
        years = 2021
        months = 1
        days = 1
        hours = 0
        mins = 0
        if secs >= 60:
            mins = int(secs // 60)
            secs %= 60
        if mins >= 60:
            hours = int(mins // 60)
            mins %= 60
        if hours >= 24:
            days += int(hours // 24)
            hours %= 24


        time = datetime(year=years, month=months, day=days,
                        hour=hours, minute=mins ,second=secs, microsecond=int(milisecs * 1000))
        return time
