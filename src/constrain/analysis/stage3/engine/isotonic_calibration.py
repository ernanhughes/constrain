from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_true, probs):
        self.model.fit(probs, y_true)

    def transform(self, probs):
        return self.model.transform(probs)