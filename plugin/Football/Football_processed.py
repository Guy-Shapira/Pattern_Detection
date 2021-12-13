from datetime import datetime, timedelta

from base.DataFormatter import DataFormatter, EventTypeClassifier
from misc.Utils import str_to_number

COLUMN_KEYS = ["sid", "ts", "x", "y", "z", "vx", "vy", "vz"]

TICKER_KEY = "sid"
EVENT_TIMESTAMP_KEY = "ts"
START_OF_PLAY = 10711885168794964


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
        time_from_game_start = event_payload[EVENT_TIMESTAMP_KEY] - START_OF_PLAY
        microseconds = time_from_game_start / 1000000
        time = datetime(year=2021, month=1, day=1,
                        hour=0, minute=0,second=0,microsecond=0) + timedelta(microseconds=microseconds)

        return time
