import math
from statistics import NormalDist

def ab_test_conversion(control_conversions: int, control_n: int, treatment_conversions: int, treatment_n: int, alpha: float = 0.05) -> dict:
    """
    Returns lift, z, p_value, ci_low, ci_high and significant for a two-proportion z-test.
    """
    # 1. Compute rates and lift
    control_rate = control_conversions / control_n
    treatment_rate = treatment_conversions / treatment_n
    lift = treatment_rate - control_rate
    
    # 2. Compute pooled values for z and p-value
    pooled_rate = (control_conversions + treatment_conversions) / (control_n + treatment_n)
    pooled_se_squared = pooled_rate * (1.0 - pooled_rate) * (1.0 / control_n + 1.0 / treatment_n)
    pooled_se = math.sqrt(pooled_se_squared)
    
    if pooled_se == 0.0:
        z = 0.0
        p_value = 1.0
    else:
        z = lift / pooled_se
        p_value = math.erfc(abs(z) / math.sqrt(2))
        
    # 3. Compute unpooled values for confidence interval
    unpooled_se_squared = (control_rate * (1.0 - control_rate) / control_n) + (treatment_rate * (1.0 - treatment_rate) / treatment_n)
    unpooled_se = math.sqrt(unpooled_se_squared)
    
    # 4. Compute confidence intervals
    quantile = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    margin_of_error = quantile * unpooled_se
    
    ci_low = lift - margin_of_error
    ci_high = lift + margin_of_error
    
    # 5. Significance flag
    significant = p_value < alpha
    
    return {
        "lift": lift,
        "z": z,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": significant
    }