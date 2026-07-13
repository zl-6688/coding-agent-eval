from flatten import flatten

assert flatten({"a": 1, "b": {"c": 2, "d": {"e": 3}}}) == {"a": 1, "b.c": 2, "b.d.e": 3}
assert flatten({}) == {}
assert flatten({"x": {"y": {"z": 1}}}) == {"x.y.z": 1}
assert flatten({"a": 1}) == {"a": 1}

print("ok")
