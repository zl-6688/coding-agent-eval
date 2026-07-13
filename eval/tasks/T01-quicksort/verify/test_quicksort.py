from quicksort import quicksort

assert quicksort([3, 1, 2]) == [1, 2, 3]
assert quicksort([]) == []
assert quicksort([5, 5, 1]) == [1, 5, 5]
assert quicksort([-1, -3, 2]) == [-3, -1, 2]

src = [3, 1, 2]
quicksort(src)
assert src == [3, 1, 2], "不应修改入参"

print("ok")
