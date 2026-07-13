from lru import LRUCache

c = LRUCache(2)
c.put(1, 1)
c.put(2, 2)
assert c.get(1) == 1          # 1 变为最近使用
c.put(3, 3)                   # 容量超了，淘汰最久未用的 2
assert c.get(2) == -1
c.put(4, 4)                   # 淘汰 1
assert c.get(1) == -1
assert c.get(3) == 3
assert c.get(4) == 4

print("ok")
