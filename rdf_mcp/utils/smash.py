import math

# generated from https://dl.acm.org/doi/pdf/10.14778/3685800.3685830 by Gemini


def _is_initial(s: str, k: int) -> bool:
    """
    Checks if the character at index k in string s is an initial letter of a word.
    According to the paper (Section 3.2), a character is an initial if it's
    the first character of the string or follows a space.

    Args:
      s: The string.
      k: The index of the character to check (0-based).

    Returns:
      True if s[k] is an initial, False otherwise.
    """
    if k < 0 or k >= len(s):
        # Should not happen if called correctly but good for safety
        return False
    # Character at index 0 is always considered an initial
    if k == 0:
        return True
    # Character is an initial if the preceding character is a space
    # Modify this if other separators should be considered word boundaries.
    return s[k - 1] == " "


def smash_distance(
    s1: str,
    s2: str,
    cost_del: float = 1.0,
    cost_ins: float = 1.0,
    cost_sub: float = 1.0,
    cost_acr: float = 1.0,
    case_sensitive: bool = False,
) -> float:
    """
    Calculates the SMASH distance between two strings based on Algorithm 1
    from the paper "Dealing with Acronyms, Abbreviations, and Typos in
    Real-World Entity Matching".

    This distance accounts for substitutions (typos), insertions/deletions
    (abbreviations), and acronym matching.

    Args:
      s1: The first string.
      s2: The second string.
      cost_del: The cost of deleting a character. Defaults to 1.0.
      cost_ins: The cost of inserting a character. Defaults to 1.0.
      cost_sub: The cost of substituting a character. Defaults to 1.0.
      cost_acr: The cost of matching a character as an acronym initial.
                Defaults to 1.0.
      case_sensitive: Whether the comparison should be case-sensitive.
                      Defaults to False.

    Returns:
      The SMASH distance between s1 and s2. Lower values indicate
      higher similarity.
    """
    if not case_sensitive:
        s1 = s1.lower()
        s2 = s2.lower()

    n = len(s1)
    m = len(s2)

    # Initialize DP table d[n+1][m+1]
    # d[i][j] will store the SMASH distance between s1[:i] and s2[:j]
    d = [[0.0] * (m + 1) for _ in range(n + 1)]

    # Initialize base cases
    # Cost of deleting characters from s1 to get an empty string
    for i in range(1, n + 1):
        d[i][0] = i * cost_del
    # Cost of inserting characters into an empty string to get s2
    for j in range(1, m + 1):
        d[0][j] = j * cost_ins

    # Precompute initial status for efficiency (Optional optimization,
    # but simple enough to include)
    s1_is_initial = [_is_initial(s1, i) for i in range(n)]
    s2_is_initial = [_is_initial(s2, j) for j in range(m)]

    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # --- Standard Edit Distance Operations ---
            # Cost of substitution/match
            sub_cost = 0.0 if s1[i - 1] == s2[j - 1] else cost_sub
            match_sub = d[i - 1][j - 1] + sub_cost

            # Cost of deletion (from s1)
            delete = d[i - 1][j] + cost_del

            # Cost of insertion (into s1 / from s2)
            insert = d[i][j - 1] + cost_ins

            # Initialize d[i][j] with the minimum of standard operations
            d[i][j] = min(match_sub, delete, insert)

            # --- Acronym Matching ---
            # Case 1: s2[j-1] is an initial character potentially matching an
            # initial s1[k] where k < i.
            # We match s1[k] with s2[j-1] (cost_acr) and delete s1[k+1...i-1]
            # (cost (i-k-1)*cost_del).
            # This corresponds to d[k][j-1] + cost_acr + (i - k - 1) * cost_del
            if s2_is_initial[j - 1]:
                cost_acronym1 = float("inf")
                for k in range(
                    i
                ):  # Corresponds to k in paper's Algorithm 1 (0 to i-1 for s1)
                    if s1_is_initial[k] and s1[k] == s2[j - 1]:
                        # Paper: d[k][j-1] + cost_acr + (i - (k+1)) * cost_del
                        # index k in s1 corresponds to length k+1 prefix
                        # index i-1 in s1 corresponds to length i prefix
                        # we are matching s1[k] (at index k) with s2[j-1] (at index j-1)
                        # we need the cost from d[k][j-1] (prefix s1[:k] to s2[:j-1])
                        # add cost_acr for the s1[k] == s2[j-1] acronym match
                        # add cost_del for deleting chars s1[k+1] through s1[i-1]
                        # Number of chars to delete = (i-1) - (k+1) + 1 = i - 1 - k
                        current_cost = d[k][j - 1] + cost_acr + (i - 1 - k) * cost_del
                        cost_acronym1 = min(cost_acronym1, current_cost)
                d[i][j] = min(d[i][j], cost_acronym1)

            # Case 2: s1[i-1] is an initial character potentially matching an
            # initial s2[k] where k < j.
            # We match s1[i-1] with s2[k] (cost_acr) and delete s2[k+1...j-1]
            # (cost (j-k-1)*cost_del).
            # This corresponds to d[i-1][k] + cost_acr + (j - k - 1) * cost_del
            if s1_is_initial[i - 1]:
                cost_acronym2 = float("inf")
                for k in range(
                    j
                ):  # Corresponds to k in paper's Algorithm 1 (0 to j-1 for s2)
                    if s2_is_initial[k] and s2[k] == s1[i - 1]:
                        # Paper: d[i-1][k] + cost_acr + (j - (k+1)) * cost_del
                        # Similar logic as above, symmetric case
                        # Match s1[i-1] with s2[k] (cost_acr)
                        # Need cost from d[i-1][k] (prefix s1[:i-1] to s2[:k])
                        # Delete chars s2[k+1] through s2[j-1]
                        # Number of chars to delete = (j-1) - (k+1) + 1 = j - 1 - k
                        current_cost = d[i - 1][k] + cost_acr + (j - 1 - k) * cost_del
                        cost_acronym2 = min(cost_acronym2, current_cost)
                d[i][j] = min(d[i][j], cost_acronym2)

    # The final distance is in the bottom-right cell
    return d[n][m]
