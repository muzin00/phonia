"""Keep the existing CV intervention while restoring non-CV controls exactly."""

import numpy as np


def restrict_to_cv(original, all_hard, cv_queries, prefix):
    np.testing.assert_array_equal(original["queries"], all_hard["queries"])
    for key in original:
        np.testing.assert_array_equal(original[key][:prefix], all_hard[key][:prefix])
    active = cv_queries[original["queries"]]
    negatives = np.where(
        active, all_hard["negative_claims"], original["negative_claims"]
    )
    return {"queries": original["queries"].copy(), "negative_claims": negatives}
