from topk import top_k

assert top_k("a a b b c", 2) == [("a", 2), ("b", 2)]      # 同次数按字母升序
assert top_k("the the the cat cat dog", 2) == [("the", 3), ("cat", 2)]
assert top_k("X x y", 2) == [("x", 2), ("y", 1)]          # 转小写
assert top_k("one", 5) == [("one", 1)]                    # k 超过词数
assert top_k("", 3) == []

print("ok")
