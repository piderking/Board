
def meters_to_inches(meters: float) -> float:
    return meters* 39.37


def frange(start: float, stop: float, step: float):
    while start < stop:
        yield start
        start += step
    yield stop