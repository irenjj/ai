1. chr(0) 返回哪个 unicode 字符:

> '\x00'

2. 该字符串表示 (__repr__()) 与打印出来的表示有什么不同

```
>>> '\x00'.__repr__()
"'\\x00'"
>>> print('\x00')

```

3. 当这个字符出现在文本中时, 会发生什么? 

```
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```

4. 与 utf-16 或 utf-32 相比, utf-8 编码的字节上训练分词器有哪些优势?

> utf8 字节表示更少相同 token 可以用更少字节表示, 训练效率更高

5. 考虑下面这个意图将 UTF-8 字节串解码为 Unicode 字符串的（错误）函数。为什么它不正
确？请给出一个会产生错误结果的输入字节串示例。

```
>>> def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
...     return "".join([bytes([b]).decode("utf-8") for b in bytestring])
...
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
>>> decode_utf8_bytes_to_str_wrong("hello".encode("は"))
Traceback (most recent call last):
  File "<python-input-31>", line 1, in <module>
    decode_utf8_bytes_to_str_wrong("hello".encode("は"))
                                   ~~~~~~~~~~~~~~^^^^^^
LookupError: unknown encoding: は
```
は 是多字节组合成的

6. 给出一个无法解码为任何 Unicode 字符的双字节序列。

```
>>> wrong_uni = b"\xb3\xa9"
>>> wrong_uni.decode("utf-8")
Traceback (most recent call last):
  File "<python-input-38>", line 1, in <module>
    wrong_uni.decode("utf-8")
    ~~~~~~~~~~~~~~~~^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb3 in position 0: invalid start byte
```
