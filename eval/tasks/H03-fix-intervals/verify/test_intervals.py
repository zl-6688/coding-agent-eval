from intervals import merge

assert merge([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
assert merge([[1, 4], [4, 5]]) == [[1, 5]]      # 相邻必须合并（命中 bug）
assert merge([]) == []
assert merge([[1, 4], [2, 3]]) == [[1, 4]]

print("ok")
