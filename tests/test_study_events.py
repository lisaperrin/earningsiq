import numpy as np
import pandas as pd
import pytest

from sec_filing_analysis.study.events import attach_earnings_releases, event_session
from sec_filing_analysis.study.returns import EventStudy, MARKET_TICKER

# Thu 2 May, Fri 3 May, Mon 6 May 2024
SESSIONS = pd.DatetimeIndex(["2024-05-02", "2024-05-03", "2024-05-06"])


@pytest.mark.parametrize("accepted, expected", [
    ("2024-05-02 08:00:00", "2024-05-02"),  # before the open
    ("2024-05-02 15:59:59", "2024-05-02"),  # during the session
    ("2024-05-02 16:00:00", "2024-05-03"),  # at the close
    ("2024-05-02 18:04:25", "2024-05-03"),  # after hours
    ("2024-05-03 17:30:00", "2024-05-06"),  # Friday evening -> Monday
    ("2024-05-04 10:00:00", "2024-05-06"),  # Saturday -> Monday
])
def test_event_session(accepted, expected):
    assert event_session(pd.Timestamp(accepted), SESSIONS) == pd.Timestamp(expected)


def test_event_session_after_last_known_session():
    assert event_session(pd.Timestamp("2024-05-06 17:00:00"), SESSIONS) is None


def test_filing_is_matched_to_the_release_of_its_own_quarter():
    events = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"],
        "filing_accepted": pd.to_datetime(["2024-05-03 06:00", "2024-08-02 06:00", "2024-05-03 06:00"]),
    })
    releases = pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA"],
        "release_accepted": pd.to_datetime(["2024-02-01 16:30", "2024-05-02 16:30", "2024-08-01 16:30"]),
    })

    matched = attach_earnings_releases(events, releases)["release_accepted"]

    assert matched[0] == pd.Timestamp("2024-05-02 16:30")
    assert matched[1] == pd.Timestamp("2024-08-01 16:30")
    assert pd.isna(matched[2])  # other company's releases are not used


def make_prices(n_days=400, beta=1.5, event_position=350, jump=0.05, seed=0):
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0005, 0.01, n_days)
    stock = beta * market + rng.normal(0, 0.0001, n_days)
    stock[event_position] += jump
    sessions = pd.bdate_range("2020-01-01", periods=n_days)
    prices = pd.DataFrame({
        "AAA": 100 * np.exp(np.cumsum(stock)),
        MARKET_TICKER: 100 * np.exp(np.cumsum(market)),
    }, index=sessions)
    return prices, sessions[event_position]


def test_car_recovers_the_jump_and_removes_the_market_move():
    prices, session = make_prices(jump=0.05)
    study = EventStudy(prices)

    assert study.car("AAA", session, 0, 1) == pytest.approx(0.05, abs=0.002)
    # no abnormal move outside the event day although the market keeps moving
    assert study.car("AAA", session, 2, 21) == pytest.approx(0.0, abs=0.003)


def test_move_size_is_large_on_the_event_day_and_ordinary_otherwise():
    prices, session = make_prices(jump=0.05)
    study = EventStudy(prices)
    quiet_day = prices.index[prices.index.get_loc(session) + 5]

    assert study.move_size("AAA", session) > 100  # a 5% jump against 0.01% daily noise
    assert study.move_size("AAA", quiet_day) < 4


def test_car_is_nan_without_enough_history():
    prices, _ = make_prices()
    study = EventStudy(prices)

    assert np.isnan(study.car("AAA", prices.index[100], 0, 1))
    assert np.isnan(study.car("AAA", prices.index[-1], 0, 1))  # window runs past the data
