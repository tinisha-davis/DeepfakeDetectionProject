class Confidence:
    def __init__(self, rating):
        self._rating = rating  # Private variable convention
        self._alert_level = self._set_alert_level()

    def _set_alert_level(self):
        if self._rating >= 90:
            return "Green"
        elif self._rating >= 60:
            return "Yellow"
        else:
            return "Red"

    def get_alert_level(self):
        return self._alert_level

    def get_rating(self):
        return self._rating
