# services/taguchi_qlf.py
# 田口质量损失函数核心计算类

import numpy as np
from scipy import integrate
from scipy.stats import norm
from config import FMEA_SEVERITY_K
import warnings

warnings.filterwarnings('ignore', category=integrate.IntegrationWarning)


class TaguchiQLFCore:
    def __init__(self):
        self.feature_calc_map = {
            "nominal": self._calc_nominal_loss,
            "smaller": self._calc_smaller_loss,
            "larger": self._calc_larger_loss,
        }

    # ---------- K 值计算 ----------
    @staticmethod
    def calc_k_value(feature_type: str, a0: float, delta0: float, a: float = None,
                     delta: float = None, fmea_severity: int = 5) -> float:
        severity_coef = FMEA_SEVERITY_K.get(fmea_severity, {}).get("a0_coef", 1.0)
        if a is not None and delta is not None and delta != 0:
            adjusted_a = a * severity_coef
            if feature_type in ("nominal", "smaller"):
                return adjusted_a / (delta ** 2)
            else:
                return adjusted_a * (delta ** 2)
        else:
            if delta0 == 0:
                return 0.0
            adjusted_a0 = a0 * severity_coef
            if feature_type in ("nominal", "smaller"):
                return adjusted_a0 / (delta0 ** 2)
            else:
                return adjusted_a0 * (delta0 ** 2)

    @staticmethod
    def calc_k_value_for_side(feature_type: str, a: float, delta: float, fmea_severity: int) -> float:
        if delta == 0:
            return 0.0
        severity_coef = FMEA_SEVERITY_K.get(fmea_severity, {}).get("a0_coef", 1.0)
        adjusted_a = a * severity_coef
        if feature_type in ("nominal", "smaller"):
            return adjusted_a / (delta ** 2)
        else:
            return adjusted_a * (delta ** 2)

    # ---------- 损失函数 ----------
    def _calc_nominal_loss(self, y: np.ndarray, m: float, k: float | tuple, asymmetric: bool = False) -> np.ndarray:
        if asymmetric and isinstance(k, (tuple, list)):
            k_upper, k_lower = k
            loss = np.where(y > m, k_upper * np.square(y - m), k_lower * np.square(y - m))
        else:
            k_val = k if isinstance(k, (int, float)) else k[0]
            loss = k_val * np.square(y - m)
        return loss

    def _calc_smaller_loss(self, y: np.ndarray, k: float) -> np.ndarray:
        return k * np.square(y)

    def _calc_larger_loss(self, y: np.ndarray, k: float) -> np.ndarray:
        y_safe = np.where(y <= 0, 1e-10, y)
        return k / np.square(y_safe)

    # ---------- 期望损失 ----------
    @staticmethod
    def expected_loss_nominal_symmetric(mu: float, sigma: float, m: float, k: float) -> float:
        return k * (sigma ** 2 + (mu - m) ** 2)

    @staticmethod
    def expected_loss_nominal_asymmetric(mu: float, sigma: float, m: float, k_lower: float, k_upper: float) -> float:
        if sigma == 0:
            k_avg = (k_lower + k_upper) / 2
            return k_avg * ((mu - m) ** 2)

        def loss_func(y):
            return np.where(y <= m, k_lower * (y - m) ** 2, k_upper * (y - m) ** 2)

        def pdf(y):
            return norm.pdf(y, loc=mu, scale=sigma)

        lower = mu - 6 * sigma
        upper = mu + 6 * sigma
        try:
            integral, _ = integrate.quad(lambda y: loss_func(y) * pdf(y), lower, upper, limit=200)
            return integral
        except Exception:
            k_avg = (k_lower + k_upper) / 2
            return k_avg * (sigma ** 2 + (mu - m) ** 2)

    @staticmethod
    def expected_loss_smaller(mu: float, sigma: float, k: float) -> float:
        return k * (mu ** 2 + sigma ** 2)

    @staticmethod
    def expected_loss_larger(mu: float, sigma: float, k: float) -> float:
        if sigma == 0:
            if mu == 0:
                return float("inf")
            return k / (mu ** 2)
        if mu < 1e-10:
            mu = 1e-10

        def loss_func(y):
            return k / (y ** 2)

        def pdf(y):
            return norm.pdf(y, loc=mu, scale=sigma)

        lower = max(1e-10, mu - 6 * sigma)
        upper = mu + 6 * sigma
        try:
            integral, _ = integrate.quad(lambda y: loss_func(y) * pdf(y), lower, upper, limit=200)
            return integral
        except Exception:
            return k / (mu ** 2)

    # ---------- 过程能力指数 ----------
    @staticmethod
    def calc_cpk(y: np.ndarray, usl: float, lsl: float, sigma_within: float = None) -> float:
        """
        计算过程能力指数 Cpk（需要组内标准差）
        :param y: 数据数组（用于计算均值，标准差使用传入的 sigma_within）
        :param usl: 上规格限
        :param lsl: 下规格限
        :param sigma_within: 组内标准差（若为 None 则返回 None）
        """
        if len(y) == 0 or sigma_within is None or sigma_within == 0:
            return None
        mean = np.mean(y)
        cpu = (usl - mean) / (3 * sigma_within)
        cpl = (mean - lsl) / (3 * sigma_within)
        return round(min(cpu, cpl), 4)

    @staticmethod
    def calc_ppk(y: np.ndarray, usl: float, lsl: float) -> float:
        """
        计算过程性能指数 Ppk（使用整体标准差）
        """
        if len(y) < 2:
            return None
        mean = np.mean(y)
        sigma = np.std(y, ddof=1)
        if sigma == 0:
            return 99.99
        cpu = (usl - mean) / (3 * sigma)
        cpl = (mean - lsl) / (3 * sigma)
        return round(min(cpu, cpl), 4)

    @staticmethod
    def calc_cpm(y: np.ndarray, usl: float, lsl: float, target: float):
        """
        计算 Cpm 指数（田口能力指数），要求样本数至少 2 个
        返回 None 表示无法计算（数据不足）
        """
        if len(y) < 2:
            return None
        mean = np.mean(y)
        var = np.var(y, ddof=1)
        bias_sq = (mean - target) ** 2
        if var + bias_sq == 0:
            return 99.99
        cpm = (usl - lsl) / (6 * np.sqrt(var + bias_sq))
        return round(cpm, 4)

    # ---------- 合并标准差 ----------
    @staticmethod
    def calculate_pooled_std(batch_groups):
        sum_sq = 0.0
        total_n_minus_1 = 0
        for group in batch_groups:
            if len(group) < 2:
                continue
            mean = np.mean(group)
            sum_sq += np.sum((group - mean) ** 2)
            total_n_minus_1 += len(group) - 1
        if total_n_minus_1 == 0:
            return 0.0
        pooled_var = sum_sq / total_n_minus_1
        return np.sqrt(pooled_var)