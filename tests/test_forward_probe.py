import numpy as np
from scipy import sparse

from pokefly.forward_probe import signed_contributions


def test_current_audit_uses_signed_incoming_rows_fractional_release_and_gain():
    rows = sparse.csr_matrix([[1.0, -2.0, 0], [-0.25, 0, 3]])
    contributions, positive, negative = signed_contributions(rows, np.array([0.5, 0.25, 0]), 3)
    np.testing.assert_allclose(contributions.toarray(), [[1.5, -1.5, 0], [-0.375, 0, 0]])
    np.testing.assert_allclose(positive, [1.5, 0])
    np.testing.assert_allclose(negative, [-1.5, -0.375])
