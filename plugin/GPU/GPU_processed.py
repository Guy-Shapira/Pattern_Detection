from datetime import datetime, timedelta

from base.DataFormatter import DataFormatter, EventTypeClassifier
from misc.Utils import str_to_number

COLUMN_KEYS = ['Timestamp', 'FB Memory Usage Used GPU_0', 'Power Samples Max GPU_0',
       'Power Samples Min GPU_0', 'Power Samples Avg GPU_0',
       'FB Memory Usage Used GPU_1', 'Power Samples Max GPU_1',
       'Power Samples Min GPU_1', 'Power Samples Avg GPU_1',
       'FB Memory Usage Used GPU_2', 'Power Samples Max GPU_2',
       'Power Samples Min GPU_2', 'Power Samples Avg GPU_2',
       'FB Memory Usage Used GPU_3', 'Power Samples Max GPU_3',
       'Power Samples Min GPU_3', 'Power Samples Avg GPU_3',
       'FB Memory Usage Used GPU_4', 'Power Samples Max GPU_4',
       'Power Samples Min GPU_4', 'Power Samples Avg GPU_4',
       'FB Memory Usage Used GPU_5', 'Power Samples Max GPU_5',
       'Power Samples Min GPU_5', 'Power Samples Avg GPU_5',
       'FB Memory Usage Used GPU_6', 'Power Samples Max GPU_6',
       'Power Samples Min GPU_6', 'Power Samples Avg GPU_6',
       'FB Memory Usage Used GPU_7', 'Power Samples Max GPU_7',
       'Power Samples Min GPU_7', 'Power Samples Avg GPU_7', 'Server']

TICKER_KEY = "Server"
EVENT_TIMESTAMP_KEY = "Timestamp"


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
        time = event_payload[EVENT_TIMESTAMP_KEY]
        return time
