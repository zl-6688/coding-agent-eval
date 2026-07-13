def merge(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    result = [list(intervals[0])]
    for start, end in intervals[1:]:
        # bug: 用 < 导致相邻区间（start 恰好等于上一段的 end）不会被合并
        if start < result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result
