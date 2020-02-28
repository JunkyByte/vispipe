class Hash:

    def __init__(self, maxsize):
        self.m = maxsize
        self.A = []
        self.V = []
        self.empty = set()
        self.deleted = set()
        for i in range(maxsize):
            self.V.append(None)
            self.empty.add(i)
            self.A.append('')

    def __H(self, key: str) -> int:
        C = 0.6180339887498949
        b = ord(key[0]) * C
        for i in range(1, len(key)):
            b = ((b * 256) + (ord(key[i]) * C)) % 1

        return int(self.m * b)

    def __H1(self, key: str) -> int:
        b = ord(key[0])
        for i in range(1, len(key)):
            b = ((b * 256) + (ord(key[i]))) % self.m
        return int(b)

    def scan(self, key: str, insert: bool) -> int:
        c = self.m
        i = 0
        j = self.__H(key)
        while j not in self.empty and self.A[j] != key and i < self.m:
            if j in self.deleted and c == self.m:
                c = j
            j = (j + self.__H1(key)) % self.m
            i = i + 1
        if insert and self.A[j] != key and c < self.m:
            j = c
        return j

    def lookup(self, key: str) -> int:
        assert isinstance(key, str)
        i = self.scan(key, False)
        if self.A[i] == key:
            return self.V[i]
        return None

    def insert(self, key: str, value) -> bool:
        i = self.scan(key, True)
        if i in self.empty or i in self.deleted or self.A[i] == key:
            self.A[i] = key
            self.V[i] = value
            try:
                self.empty.remove(i)
            except KeyError:
                pass
            try:
                self.deleted.remove(i)
            except KeyError:
                pass
            return True
        else:
            return False

    def remove(self, key: str):
        i = self.scan(key, False)
        if i not in self.deleted and i not in self.empty and self.A[i] == key:
            self.deleted.add(i)


def main():
    myhash = Hash(500)
    myhash.insert("90's plaid", 3)
    myhash.insert("hipster ipsum", 1)
    myhash.remove("90's plaid")
    myhash.insert("Wes Anderson ", 2)
    myhash.insert("lomo", 4)
    myhash.insert("McSweeney's Tonx hashtag twee", 5)
    myhash.insert("vegan craft beer", 6)
    #myhash.insert("90's plaid", 3);

    assert myhash.lookup("non esisto") is None

    print(myhash.lookup("hipster ipsum"))
    print(myhash.lookup("Wes Anderson "))
    print(myhash.lookup("90's plaid"))
    print(myhash.lookup("lomo"))
    print(myhash.lookup("McSweeney's Tonx hashtag twee"))
    print(myhash.lookup("vegan craft beer"))


if __name__ == "__main__":
    main()
