# services/target_calc.py
# 目标值优化核心算法（7种方法）

import numpy as np
from config import FEATURE_TYPE
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

class TargetCalcMethods:
    def __init__(self, ctq_row, production_data, decay_config=None, voc_data=None,
                 target_cpk=1.33, tendency=0.5, target_cpm=1.0):
        self.ctq = ctq_row
        self.ctq_id = getattr(ctq_row, 'ctq_id', 0)
        self.ctq_name = getattr(ctq_row, 'ctq_name', '未知CTQ')
        self.feature_type = getattr(ctq_row, 'feature_type', 'nominal')
        self.usl = float(ctq_row.usl) if ctq_row.usl is not None else np.nan
        self.lsl = float(ctq_row.lsl) if ctq_row.lsl is not None else np.nan
        self.target_m = float(ctq_row.target_m) if ctq_row.target_m is not None else np.nan
        self.k_upper = float(ctq_row.k_upper) if ctq_row.k_upper is not None else 0
        self.k_lower = float(ctq_row.k_lower) if ctq_row.k_lower is not None else 0
        self.fmea_severity = int(ctq_row.fmea_severity) if ctq_row.fmea_severity is not None else 5
        self.is_ccp = getattr(ctq_row, 'is_ccp', '否') == "是"
        self.a0 = float(ctq_row.a0) if ctq_row.a0 is not None else 0
        self.delta0 = float(ctq_row.delta0) if ctq_row.delta0 is not None else 0
        self.is_asymmetric = getattr(ctq_row, 'asymmetric_loss', '否') == "是"
        self.y_values = []
        for d in production_data:
            val = getattr(d, 'measured_value', None)
            if val is not None and val != '':
                try:
                    fval = float(val)
                    if not np.isnan(fval) and not np.isinf(fval):
                        self.y_values.append(fval)
                except:
                    continue
        self.y_values = np.array(self.y_values)
        self.data_count = len(self.y_values)
        if self.data_count > 0:
            self.mean = np.mean(self.y_values)
            self.std = np.std(self.y_values, ddof=1) if self.data_count >= 2 else 0.000001
        else:
            self.mean = np.nan
            self.std = 0.000001
        self.has_valid_data = self.data_count > 0 and not np.all(self.y_values == self.y_values[0]) if self.data_count > 0 else False
        self.target_cpk = float(target_cpk) if target_cpk is not None else 1.33
        self.tendency = max(0.0, min(1.0, float(tendency) if tendency is not None else 0.5))
        self.target_cpm = float(target_cpm) if target_cpm is not None else 1.0
        self.has_valid_spec = not np.isnan(self.usl) and not np.isnan(self.lsl) and self.usl > self.lsl
        self.tolerance_center = (self.usl + self.lsl) / 2 if self.has_valid_spec else np.nan
        self.tolerance_width = self.usl - self.lsl if self.has_valid_spec else np.nan
        self.has_valid_target = not np.isnan(self.target_m)
        self.has_valid_k = self.k_upper > 0 or self.k_lower > 0
        self.is_shelf_life_ctq = (self.feature_type == "larger" and
                                  ("活菌" in self.ctq_name or "乳酸菌" in self.ctq_name or "双歧杆菌" in self.ctq_name))
        self.decay_config = decay_config
        self.method_name_map = {
            "method_1": "国标约束保底法",
            "method_2": "CPK过程能力反推法",
            "method_3": "SN信噪比稳健优化法",
            "method_4": "CPK+SN双约束联合优化法",
            "method_5": "期望损失最小化法",
            "method_6": "Arrhenius货架期衰减动态法",
            "method_7": "VOC客户反馈拟合法"
        }

    def _base_check(self, need_spec=False, need_target=False, min_data_count=0,
                    need_k=False, only_nominal=False, only_larger=False):
        try:
            if only_nominal and self.feature_type != "nominal":
                feature_name = FEATURE_TYPE.get(self.feature_type, {}).get('name', '未知类型')
                return False, f"❌ 失败原因：该方法仅支持望目特性，当前CTQ【{self.ctq_name}】为{feature_name}"
            if only_larger and self.feature_type != "larger":
                feature_name = FEATURE_TYPE.get(self.feature_type, {}).get('name', '未知类型')
                return False, f"❌ 失败原因：该方法仅支持望大特性，当前CTQ【{self.ctq_name}】为{feature_name}"
            if need_k and self.feature_type == "nominal" and self.is_asymmetric:
                return False, "❌ 失败原因：当前CTQ为非对称损失，方法5仅支持对称望目特性"
            if need_spec and not self.has_valid_spec:
                return False, f"❌ 失败原因：当前CTQ【{self.ctq_name}】未正确配置上下规格限"
            if need_target and not self.has_valid_target:
                return False, f"❌ 失败原因：当前CTQ【{self.ctq_name}】未配置目标值m"
            if need_k and not self.has_valid_k:
                return False, f"❌ 失败原因：当前CTQ【{self.ctq_name}】未配置K值损失系数"
            if min_data_count > 0 and self.data_count < min_data_count:
                return False, f"❌ 失败原因：需要至少{min_data_count}组数据，当前仅{self.data_count}组"
            return True, "✅ 校验通过"
        except Exception as e:
            return False, f"❌ 校验异常：{str(e)}"

    def _predict_cpm(self, target):
        if self.std == 0 or np.isnan(self.std) or np.isnan(self.mean):
            return np.nan
        bias_sq = (self.mean - target) ** 2
        denom = 6 * np.sqrt(self.std**2 + bias_sq)
        if denom == 0:
            return 99.99
        return (self.usl - self.lsl) / denom

    def _select_optimal_in_range(self, lower, upper):
        if self.feature_type == "smaller":
            if lower <= 0:
                return min(upper, 0)
            return lower
        elif self.feature_type == "larger":
            return upper
        else:
            if lower > upper:
                return self.tolerance_center
            if self.is_asymmetric and self.has_valid_k:
                best_loss = float('inf')
                best_target = (lower + upper) / 2
                for candidate in np.linspace(lower, upper, 100):
                    from services.taguchi_qlf import TaguchiQLFCore
                    calc = TaguchiQLFCore()
                    loss = calc.expected_loss_nominal_asymmetric(self.mean, self.std, candidate,
                                                                 self.k_lower, self.k_upper)
                    if loss < best_loss:
                        best_loss = loss
                        best_target = candidate
                return best_target
            else:
                return lower + (upper - lower) * self.tendency

    def _cpk_cpm_optimize(self):
        offset = self.target_cpk * 3 * self.std
        cpk_lower = self.lsl + offset
        cpk_upper = self.usl - offset
        base_cpm_tolerance = self.tolerance_width / (6 * self.target_cpm)
        cpm_d_sq = base_cpm_tolerance**2 - self.std**2
        if cpm_d_sq < 0:
            cpm_lower, cpm_upper = None, None
            cpm_note = "⚠️ 当前波动过大，即使均值等于目标值也无法达到目标CPM"
        else:
            d = np.sqrt(cpm_d_sq)
            cpm_lower = max(self.lsl, self.mean - d)
            cpm_upper = min(self.usl, self.mean + d)
            cpm_note = ""
        combined_lower = cpk_lower
        combined_upper = cpk_upper
        if cpm_lower is not None:
            combined_lower = max(cpk_lower, cpm_lower)
            combined_upper = min(cpk_upper, cpm_upper)
        if combined_lower > combined_upper:
            final_target = self._select_optimal_in_range(cpk_lower, cpk_upper) if cpk_lower <= cpk_upper else self.tolerance_center
            predicted_cpm = self._predict_cpm(final_target)
            remark = "⚠️ 无法同时满足目标CPM，已回退到仅CPK约束"
            if cpm_note:
                remark += "；" + cpm_note
            return final_target, predicted_cpm, remark
        else:
            final_target = self._select_optimal_in_range(combined_lower, combined_upper)
            predicted_cpm = self._predict_cpm(final_target)
            if cpm_note:
                return final_target, predicted_cpm, "⚠️ " + cpm_note
            else:
                return final_target, predicted_cpm, "CPK与CPM联合约束满足"

    def _cpk_eval_benefit(self, optimal_target, predicted_cpm=None):
        expected_cpk = min((self.usl - optimal_target) / (3 * self.std),
                           (optimal_target - self.lsl) / (3 * self.std))
        if expected_cpk >= self.target_cpk:
            benefit = f"预期Ppk可达{round(expected_cpk,4)}，满足目标Ppk({self.target_cpk})"
        else:
            benefit = f"当前波动过大，预期Ppk仅{round(expected_cpk,4)}，无法达到目标Ppk({self.target_cpk})"
        if predicted_cpm is not None and not np.isnan(predicted_cpm):
            benefit += f"，预测CPM {round(predicted_cpm,4)}"
            if predicted_cpm >= self.target_cpm:
                benefit += " (达标)"
            else:
                benefit += " (未达标)"
        return benefit, expected_cpk

    def method_1_gb_base(self):
        check_pass, check_msg = self._base_check(need_spec=True)
        if not check_pass:
            return {"code": 400, "optimal_target": None, "core_index": "-", "benefit": "-", "remark": check_msg}
        optimal_target = self.tolerance_center
        predicted_cpm = self._predict_cpm(optimal_target)
        return {"code": 200, "optimal_target": round(optimal_target, 4),
                "core_index": f"预测CPM: {round(predicted_cpm,4)}",
                "benefit": "100%符合国标要求",
                "remark": "行业通用保底方案",
                "predicted_cpm": round(predicted_cpm,4)}

    def method_2_cpk_reverse(self):
        check_pass, check_msg = self._base_check(need_spec=True, min_data_count=30)
        if not check_pass:
            return {"code": 400, "optimal_target": None, "core_index": "-", "benefit": "-", "remark": check_msg}
        opt_target, predicted_cpm, cpm_remark = self._cpk_cpm_optimize()
        benefit, expected_cpk = self._cpk_eval_benefit(opt_target, predicted_cpm)
        return {"code": 200, "optimal_target": round(opt_target, 4),
                "core_index": f"Ppk: {round(expected_cpk,4)} | CPM: {round(predicted_cpm,4)}",
                "benefit": benefit,
                "remark": cpm_remark,
                "predicted_cpm": round(predicted_cpm,4)}

    def method_3_snr_robust(self):
        need_target = (self.feature_type == "nominal")
        check_pass, check_msg = self._base_check(need_target=need_target, min_data_count=5)
        if not check_pass:
            return {"code": 400, "optimal_target": None, "core_index": "-", "benefit": "-", "remark": check_msg}
        if self.feature_type == "nominal":
            snr = 10 * np.log10((self.mean ** 2) / (self.std ** 2)) if self.std > 0 else 99
            optimal_target = self.target_m
        elif self.feature_type == "smaller":
            mean_sq = np.mean(self.y_values ** 2)
            snr = -10 * np.log10(mean_sq) if mean_sq != 0 else float('inf')
            optimal_target = 0
        else:
            if self.has_valid_spec and not np.isnan(self.usl):
                optimal_target = self.usl
            else:
                optimal_target = np.max(self.y_values) * 1.2 if len(self.y_values) > 0 else 1e7
            valid_y = self.y_values[self.y_values != 0]
            if len(valid_y) == 0:
                return {"code": 400, "remark": "望大特性数据全为0"}
            mean_reciprocal_sq = np.mean(1 / (valid_y ** 2))
            snr = -10 * np.log10(mean_reciprocal_sq) if mean_reciprocal_sq != 0 else float('inf')
        predicted_cpm = self._predict_cpm(optimal_target)
        return {"code": 200, "optimal_target": round(optimal_target, 4),
                "core_index": f"SN比: {round(snr, 2)} dB",
                "benefit": "最大化抗干扰能力",
                "remark": "田口原生方法，适用于评估",
                "predicted_cpm": round(predicted_cpm,4) if not np.isnan(predicted_cpm) else None}

    def method_4_cpk_sn_joint(self):
        check_pass, check_msg = self._base_check(need_spec=True, need_target=True, min_data_count=30)
        if not check_pass:
            return {"code": 400, "optimal_target": None, "core_index": "-", "benefit": "-", "remark": check_msg}
        opt_target, predicted_cpm, cpm_remark = self._cpk_cpm_optimize()
        benefit, expected_cpk = self._cpk_eval_benefit(opt_target, predicted_cpm)
        snr = 10 * np.log10((opt_target ** 2) / (self.std ** 2)) if self.std > 0 else 99
        return {"code": 200, "optimal_target": round(opt_target, 4),
                "core_index": f"Ppk: {round(expected_cpk,4)} | CPM: {round(predicted_cpm,4)} | SN: {round(snr,2)}dB",
                "benefit": benefit,
                "remark": cpm_remark,
                "predicted_cpm": round(predicted_cpm,4)}

    def method_5_bayes_opt(self):
        check_pass, check_msg = self._base_check(need_spec=True, need_target=True, min_data_count=10, need_k=True, only_nominal=True)
        if not check_pass:
            return {"code": 400, "optimal_target": None, "core_index": "-", "benefit": "-", "remark": check_msg}
        k = (self.k_upper + self.k_lower) / 2
        def loss_func(t):
            return k * (self.std ** 2 + (self.mean - t) ** 2)
        result = optimize.minimize_scalar(loss_func, bounds=(self.lsl, self.usl), method='bounded')
        optimal_target = result.x if result.success else (self.lsl + self.usl) / 2
        min_loss = loss_func(optimal_target)
        predicted_cpm = self._predict_cpm(optimal_target)
        expected_cpk = min((self.usl - optimal_target) / (3 * self.std),
                           (optimal_target - self.lsl) / (3 * self.std))
        return {"code": 200, "optimal_target": round(optimal_target, 4),
                "core_index": f"最小期望损失: {round(min_loss, 6)}元 | Ppk: {round(expected_cpk, 4)}",
                "benefit": "全局最小化质量损失",
                "remark": "适用于对称望目特性",
                "predicted_cpm": round(predicted_cpm,4) if not np.isnan(predicted_cpm) else None}

    def method_6_arrhenius_decay(self):
        check_pass, check_msg = self._base_check(need_target=False, min_data_count=5, only_larger=True)
        if not check_pass:
            return {"code": 400, "optimal_target": None, "core_index": "-", "benefit": "-", "remark": check_msg}
        if not self.decay_config:
            return {"code": 400, "remark": "未配置衰减参数"}
        if self.lsl <= 0:
            return {"code": 400, "remark": "下规格限必须大于0才能计算货架期动态目标"}
        shelf_life = float(getattr(self.decay_config, 'shelf_life_days', 28))
        std_temp = float(getattr(self.decay_config, 'std_cold_temp', 4.0))
        actual_temp = float(getattr(self.decay_config, 'actual_avg_temp', 6.0))
        temp_coef = float(getattr(self.decay_config, 'temp_coef', 0.12))
        alpha0 = 1.0 / 30.0
        alpha = alpha0 * np.exp(temp_coef * (actual_temp - std_temp))
        optimal_target = self.lsl * np.exp(alpha * shelf_life)
        if optimal_target > self.usl * 1.5:
            optimal_target = self.usl * 1.5
        if optimal_target < self.lsl:
            optimal_target = self.lsl * 1.1
        end_life_value = optimal_target * np.exp(-alpha * shelf_life)
        predicted_cpm = self._predict_cpm(optimal_target)
        return {"code": 200, "optimal_target": round(optimal_target, 4),
                "core_index": f"终点预期值: {round(end_life_value, 4)}",
                "benefit": "确保保质期终点合规",
                "remark": "酸奶专属，基于Arrhenius模型",
                "predicted_cpm": round(predicted_cpm,4) if not np.isnan(predicted_cpm) else None}

    def method_7_voc_fit(self):
        return {"code": 400, "optimal_target": None, "remark": "VOC模块暂未开放"}

    def get_method_available_status(self):
        method_status = {}
        check1_pass, check1_msg = self._base_check(need_spec=True)
        method_status["method_1"] = {"available": check1_pass, "reason": check1_msg}
        check2_pass, check2_msg = self._base_check(need_spec=True, min_data_count=30)
        method_status["method_2"] = {"available": check2_pass, "reason": check2_msg}
        need_target_3 = (self.feature_type == "nominal")
        check3_pass, check3_msg = self._base_check(need_target=need_target_3, min_data_count=5)
        method_status["method_3"] = {"available": check3_pass, "reason": check3_msg}
        check4_pass, check4_msg = self._base_check(need_spec=True, need_target=True, min_data_count=30)
        method_status["method_4"] = {"available": check4_pass, "reason": check4_msg}
        check5_pass, check5_msg = self._base_check(need_spec=True, need_target=True, min_data_count=10, need_k=True, only_nominal=True)
        method_status["method_5"] = {"available": check5_pass, "reason": check5_msg}
        check6_pass, check6_msg = self._base_check(need_target=False, min_data_count=5, only_larger=True)
        if check6_pass and not self.decay_config:
            check6_pass = False
            check6_msg = "❌ 失败原因：未配置货架期衰减参数"
        method_status["method_6"] = {"available": check6_pass, "reason": check6_msg}
        method_status["method_7"] = {"available": False, "reason": "❌ 失败原因：VOC模块暂未开放"}
        return method_status

    def calc_by_method(self, method_key):
        method_map = {
            "method_1": self.method_1_gb_base,
            "method_2": self.method_2_cpk_reverse,
            "method_3": self.method_3_snr_robust,
            "method_4": self.method_4_cpk_sn_joint,
            "method_5": self.method_5_bayes_opt,
            "method_6": self.method_6_arrhenius_decay,
            "method_7": self.method_7_voc_fit,
        }
        method_status = self.get_method_available_status()
        if not method_status[method_key]["available"]:
            return {"code": 400, "optimal_target": None, "core_index": "-", "benefit": "-", "remark": method_status[method_key]["reason"]}
        return method_map[method_key]()

    def recommend_best_method(self):
        method_status = self.get_method_available_status()
        available = [k for k, v in method_status.items() if v["available"]]
        if not available:
            return {"code": 200, "recommend_method": "method_1", "method_name": "国标约束保底法", "recommend_reason": "无其他可用方法，默认推荐"}
        if self.is_shelf_life_ctq and "method_6" in available:
            return {"code": 200, "recommend_method": "method_6", "method_name": "Arrhenius衰减法", "recommend_reason": "活菌数专用"}
        if self.is_ccp and "method_4" in available:
            return {"code": 200, "recommend_method": "method_4", "method_name": "CPK+SN联合", "recommend_reason": "CCP关键点推荐"}
        if self.has_valid_k and self.is_asymmetric and "method_4" in available:
            return {"code": 200, "recommend_method": "method_4", "method_name": "CPK+SN联合（非对称优化）", "recommend_reason": "已配置非对称K值，自动寻优"}
        if self.has_valid_k and not self.is_asymmetric and "method_5" in available:
            return {"code": 200, "recommend_method": "method_5", "method_name": "期望损失最小化法", "recommend_reason": "已有K值，全局寻优"}
        if "method_2" in available:
            return {"code": 200, "recommend_method": "method_2", "method_name": "CPK反推法", "recommend_reason": "有充足数据"}
        return {"code": 200, "recommend_method": "method_1", "method_name": "国标保底法", "recommend_reason": "通用"}