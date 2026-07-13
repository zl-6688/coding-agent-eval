from wordcount import word_count

assert word_count("a a b") == {"a": 2, "b": 1}
assert word_count("Hello hello") == {"hello": 2}
assert word_count("") == {}
assert word_count("one two two three three three") == {"one": 1, "two": 2, "three": 3}

print("ok")
