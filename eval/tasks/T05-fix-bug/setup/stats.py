def mean(nums):
    # 这里有个 bug：除数错了
    return sum(nums) / (len(nums) - 1)
