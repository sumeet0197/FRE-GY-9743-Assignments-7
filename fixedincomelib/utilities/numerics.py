import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'
    MONOTONE_CONVEX = 'MONOTONE_CONVEX'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        return float(self.values_[idx])

    def integrate(self, start_x: float, end_x: float) -> float:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        overlaps = np.maximum(hi - lo, 0.0)
        return float(np.dot(np.asarray(self.values_, dtype=float), overlaps))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        idx = min(int(np.searchsorted(self.axis1_, x, side='left')), self.length_ - 1)
        grad = np.zeros(self.length_)
        grad[idx] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        edges = np.concatenate(([-np.inf], np.asarray(self.axis1_, dtype=float)[:-1], [np.inf]))
        lo = np.maximum(start_x, edges[:-1])
        hi = np.minimum(end_x, edges[1:])
        return np.maximum(hi - lo, 0.0)



class Interpolator1DMonotoneConvex(Interpolator1D):
    """Shape-preserving quadratic spline on the instantaneous forward rate.

    The curve is C1, passes through every supplied IFR point, and on each node
    interval is both monotone and convex (or concave) whenever the data allows it.

    Method
    ------
    Hagan and West build their monotone convex curve from *discrete* forwards: the
    defining constraint is that the interpolant's average over each interval
    reproduces the input, and the instantaneous forwards at the nodes are derived.
    Here the inputs are the instantaneous forwards themselves and must be matched
    exactly, so that constraint is not available and their formulae do not transfer
    unchanged. What does transfer is the device that makes their method work: allow
    a single extra breakpoint inside an interval, so each piece can be shaped on its
    own instead of a single polynomial having to satisfy every condition at once.

    That is Schumaker's shape-preserving quadratic spline (Schumaker 1983). On each
    interval the spline is two quadratics joined at an interior knot, with the slopes
    at the nodes chosen locally. Concretely, on [t_i, t_{i+1}] with h = t_{i+1} - t_i
    and secant slope D = (f_{i+1} - f_i) / h, the knot splits the interval into
    lengths a and b, and matching the right-hand value forces the slope at the knot:

        mbar = 2 D - (a m_i + b m_{i+1}) / h.

    The derivative is then piecewise linear, running m_i -> mbar -> m_{i+1}, which is
    what makes the shape easy to control and the integral exact.

    Why the shape holds
    -------------------
    Interior slopes are taken as a length-weighted average of the two adjacent secant
    slopes, so each m_i lies between them. On any interval the node slopes therefore
    straddle that interval's secant, and with Schumaker's knot placement the algebra
    collapses to mbar = D exactly (see _interval_pieces). The derivative path is then
    m_i -> D -> m_{i+1}:

      * monotone, because all three share the sign of D, so f' never changes sign;
      * convex, because for convex data the secants increase, the slopes inherit
        m_i <= D <= m_{i+1}, and a non-decreasing derivative is convexity.

    Positivity comes free: with no overshoot, f stays inside [f_i, f_{i+1}] on each
    interval, so positive inputs give a positive curve.

    Data that cannot satisfy every condition
    ----------------------------------------
    Where consecutive secants have opposite signs the data has a local extremum, and
    no interpolant through it can be monotone there. The slope at that node is set to
    zero, which turns the extremum into a smooth turning point instead of letting the
    spline overshoot and manufacture a forward rate the data never implied. The curve
    stays continuous, still interpolates, and remains monotone on every interval that
    is monotone in the data; only convexity is given up, and only locally. Likewise a
    flat segment (D = 0) forces both of its slopes to zero so the piece is exactly
    constant rather than wandering between equal endpoints.

    Degenerate inputs: one node gives a constant curve, two give a straight line, and
    repeated node times are collapsed before anything is built.

    Extrapolation is FLAT at both ends, matching the piecewise-constant interpolator.
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.MONOTONE_CONVEX,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

        t = np.asarray(self.axis1_, dtype=float)
        f = np.asarray(self.values_, dtype=float)

        # Repeated node times would divide by zero. Keep the last value at each time,
        # which is the same rule a left-continuous reading of the data would apply.
        if len(t) > 1:
            keep = np.concatenate((np.diff(t) > 0.0, [True]))
            t, f = t[keep], f[keep]

        self.nodes_ = t
        self.node_values_ = f
        self.slopes_ = self._node_slopes(t, f)

        # One flat left wing, two quadratic pieces per interval, one flat right wing.
        # Everything the class does is read off this one table, so interpolate and
        # integrate cannot disagree about the curve.
        self.breaks_, self.coeffs_ = self._build_pieces(t, f, self.slopes_)

        # Integral from the first node to each later breakpoint, so a partial integral
        # is one table lookup plus one quadratic, rather than a loop over pieces. Only
        # the interior pieces are accumulated: the wings are unbounded, and their
        # contribution depends on where the caller stops, so it is added at lookup.
        widths = np.diff(self.breaks_[1:-1])
        c0 = self.coeffs_[1:-1, 0]
        c1 = self.coeffs_[1:-1, 1]
        c2 = self.coeffs_[1:-1, 2]
        piece_integrals = widths * (c0 + widths * (c1 / 2.0 + widths * c2 / 3.0))
        self.cumulative_ = np.concatenate(([0.0], np.cumsum(piece_integrals)))

    @staticmethod
    def _node_slopes(t: np.ndarray, f: np.ndarray) -> np.ndarray:
        n = len(t)
        if n == 1:
            return np.zeros(1)

        h = np.diff(t)
        secants = np.diff(f) / h
        m = np.zeros(n)

        # Interior: a length-weighted average of the two adjacent secants. The point
        # of the weighting is not accuracy but placement -- the result always lies
        # between the two secants, which is exactly the condition the shape argument
        # in the class docstring needs.
        for i in range(1, n - 1):
            left, right = secants[i - 1], secants[i]
            if left * right <= 0.0:
                # Local extremum, or a flat run. No monotone interpolant passes
                # through it, so flatten the node rather than overshoot.
                m[i] = 0.0
            else:
                m[i] = (h[i - 1] * left + h[i] * right) / (h[i - 1] + h[i])

        if n == 2:
            # A single interval: the straight line is already monotone and convex.
            m[0] = m[1] = secants[0]
            return m

        # Ends: reflect the neighbouring slope through the end secant, which keeps the
        # end slope on the opposite side of the secant from its neighbour and so
        # preserves the straddling property there too. Then clip, so monotonicity is
        # never sacrificed to the reflection.
        m[0] = Interpolator1DMonotoneConvex._clip_between(
            2.0 * secants[0] - m[1], 0.0, secants[0])
        m[-1] = Interpolator1DMonotoneConvex._clip_between(
            2.0 * secants[-1] - m[-2], secants[-1], 2.0 * secants[-1])

        # A flat interval must stay exactly flat at both of its ends.
        for i, d in enumerate(secants):
            if d == 0.0:
                m[i] = m[i + 1] = 0.0

        return m

    @staticmethod
    def _clip_between(value: float, a: float, b: float) -> float:
        lo, hi = (a, b) if a <= b else (b, a)
        return float(min(max(value, lo), hi))

    @staticmethod
    def _interval_pieces(t_left, t_right, f_left, f_right, m_left, m_right):
        """The two quadratics covering one node interval.

        Returns (knot, mbar). Schumaker's knot placement, written so the two
        interesting cases are visible:

          * the node slopes straddle the secant (the shape-preserving case): the knot
            is placed so that mbar comes out exactly equal to the secant slope, which
            is what makes the derivative path m_left -> D -> m_right;
          * they sit on the same side (only possible at a flattened extremum here):
            the knot goes at the midpoint, and mbar follows from matching the
            right-hand value.
        """
        h = t_right - t_left
        secant = (f_right - f_left) / h
        d_left, d_right = m_left - secant, m_right - secant

        if d_left * d_right >= 0.0:
            knot = t_left + h / 2.0
        elif abs(d_right) < abs(d_left):
            knot = t_right + h * d_left / (m_right - m_left)
        else:
            knot = t_left + h * d_right / (m_right - m_left)

        a = knot - t_left
        b = h - a
        # Forced by s(t_right) = f_right once both pieces are quadratic and the slope
        # is continuous at the knot.
        mbar = 2.0 * secant - (a * m_left + b * m_right) / h
        return knot, mbar

    @classmethod
    def _build_pieces(cls, t: np.ndarray, f: np.ndarray, m: np.ndarray):
        """Breakpoints and quadratic coefficients, including both flat wings.

        Piece k covers [breaks[k], breaks[k+1]] and holds
        f(x) = c0 + c1 (x - breaks[k]) + c2 (x - breaks[k])^2.
        """
        breaks = [-np.inf, t[0]]
        coeffs = [(f[0], 0.0, 0.0)]  # flat left wing

        for i in range(len(t) - 1):
            knot, mbar = cls._interval_pieces(
                t[i], t[i + 1], f[i], f[i + 1], m[i], m[i + 1])
            a = knot - t[i]
            b = t[i + 1] - knot

            # Left piece: value f[i], slope m[i], reaching slope mbar at the knot.
            if a > 0.0:
                breaks.append(knot)
                coeffs.append((f[i], m[i], (mbar - m[i]) / (2.0 * a)))
                f_knot = f[i] + a * (m[i] + mbar) / 2.0
            else:
                f_knot = f[i]

            # Right piece: from the knot to the next node, ending on slope m[i+1].
            if b > 0.0:
                breaks.append(t[i + 1])
                coeffs.append((f_knot, mbar, (m[i + 1] - mbar) / (2.0 * b)))
            elif a > 0.0:
                breaks[-1] = t[i + 1]

        breaks.append(np.inf)
        coeffs.append((f[-1], 0.0, 0.0))  # flat right wing

        return np.asarray(breaks, dtype=float), np.asarray(coeffs, dtype=float)

    def _piece_index(self, x: float) -> int:
        # side='left' keeps a point sitting exactly on a breakpoint in the piece that
        # ends there. The spline is continuous, so either choice gives the same value;
        # this one matches the left-continuous reading used elsewhere in the library.
        idx = int(np.searchsorted(self.breaks_, x, side='left')) - 1
        return min(max(idx, 0), len(self.coeffs_) - 1)

    def interpolate(self, x: float) -> float:
        k = self._piece_index(x)
        c0, c1, c2 = self.coeffs_[k]
        # The wings carry c1 = c2 = 0, so an infinite breakpoint never reaches the
        # arithmetic and flat extrapolation needs no special case.
        dx = 0.0 if not np.isfinite(self.breaks_[k]) else x - self.breaks_[k]
        return float(c0 + dx * (c1 + dx * c2))

    def _integral_from_start(self, x: float) -> float:
        """Integral from the first node to x, with the flat wings included."""
        k = self._piece_index(x)
        if not np.isfinite(self.breaks_[k]):
            # Left wing: a constant run back from the first node, so the integral is
            # negative when x lies before it.
            return self.node_values_[0] * (x - self.nodes_[0])
        dx = x - self.breaks_[k]
        c0, c1, c2 = self.coeffs_[k]
        return float(self.cumulative_[k - 1] + dx * (c0 + dx * (c1 / 2.0 + dx * c2 / 3.0)))

    def integrate(self, start_x: float, end_x: float) -> float:
        if end_x == start_x:
            return 0.0
        if end_x < start_x:
            return -self.integrate(end_x, start_x)
        return self._integral_from_start(end_x) - self._integral_from_start(start_x)

    # Optional risk interface
    # The test cells for risk are not prepared but you can do it by your own.
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        raise NotImplementedError("Optional risk: ordinate gradient is not implemented")

    def gradient_of_integrated_value_wrt_ordinate(
        self, start_x: float, end_x: float,
    ) -> np.ndarray:
        raise NotImplementedError("Optional risk: integrated ordinate gradient is not implemented")


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        elif interpolation_method == InterpMethod.MONOTONE_CONVEX:
            return Interpolator1DMonotoneConvex(axis1_, values_, extrapolation_method)
        else:
            raise ValueError(f'Unsupported interpolation method: {interpolation_method}')
