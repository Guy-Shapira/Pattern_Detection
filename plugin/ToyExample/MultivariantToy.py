from datetime import datetime

from base.DataFormatter import DataFormatter, EventTypeClassifier
from misc.Utils import str_to_number

COLUMN_KEYS = ["Type", "Value1", "Value2", "Count"]

TICKER_KEY = "Type"
EVENT_TIMESTAMP_KEY = "Count"
fmt = "%Y-%m-%d %H:%M:%S.%f"


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
        timestamp_str = str(event_payload[EVENT_TIMESTAMP_KEY])
        return datetime.strptime(timestamp_str, fmt)
