from bitarray import bitarray
import sys
def GetData(data):
    d = bitarray('1'*8*4096+ '1111'*6 + '00001001' + '0000'*8 + '00000000' + '00011111' + '00111101' + data)
    return d

def GetHamming(indexData):
    #d = GetData(indexData)
    data = list(indexData)
    c, ch, j, r, h = 0, 0, 0, 0, []
    result = []
    while ((len(indexData) + r + 1) > (pow(2, r))):
        r = r + 1

    for i in range(0, (r + len(data))):
        p = (2 ** c)

        if (p == (i + 1)):
            h.append(0)
            c = c + 1

        else:
            h.append(int(data[j]))
            j = j + 1

    for parity in range(0, (len(h))):
        ph = (2 ** ch)
        if (ph == (parity + 1)):
            startIndex = ph - 1
            i = startIndex
            toXor = []

            while (i < len(h)):
                block = h[i:i + ph]
                toXor.extend(block)
                i += 2 * ph

            for z in range(1, len(toXor)):
                h[startIndex] = h[startIndex] ^ toXor[z]

            result.append(h[startIndex])
            ch += 1

    #print('Hamming code generated would be: ', end="")
    #print(result, len(result))
    final = (bitarray(result)).tobytes().hex()
    hh = (bitarray(result[:8][::-1])+bitarray(result[8:][::-1])).tobytes().hex()
    #print(final,hh,bitarray(result))
    return bitarray(result)

if __name__ == "__main__":
    for i in range(2047432,2047432+30):
        data = GetData("{:0>32b}".format(i))
        sys.stdout.write(bitarray("{:0>32b}".format(i)).tobytes().hex()+" ")
        GetHamming(data)
    print(int.from_bytes(bytes.fromhex("15a30000"),"big"))
    print("{:0>32b}".format(int.from_bytes(bytes.fromhex("15a30000"),"big")))
    print("{:0>32b}".format(int.from_bytes(bytes.fromhex("bd208000"),"big")))
    print("{:0>32b}".format(int.from_bytes(bytes.fromhex("bd238000"),"big")))
    print("{:0>32b}".format(int.from_bytes(bytes.fromhex("15a00000"),"big")))
    #print(bitarray.frombytes(bytes.fromhex("15a30000")))
