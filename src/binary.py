from struct import unpack

def read8(str, offset=0):
    return unpack('B', str[offset:offset + 1])[0]

def read16be(str, offset=0):
    return unpack('>H', str[offset:offset + 2])[0]

def read16le(str, offset=0):
    return unpack('<H', str[offset:offset + 2])[0]

def read32be(str, offset=0):
    return unpack('>I', str[offset:offset + 4])[0]

def read32le(str, offset=0):
    return unpack('<I', str[offset:offset + 4])[0]

def read8s(str, length, offset=0):
    return unpack('{}B'.format(length), str[offset:offset + length])

def read16bes(str, length, offset=0):
    return unpack('>{}H'.format(length), str[offset:offset + 2 * length])

def read16les(str, length, offset=0):
    return unpack('<{}H'.format(length), str[offset:offset + 2 * length])

def read32bes(str, length, offset=0):
    return unpack('>{}I'.format(length), str[offset:offset + 4 * length])

def read32les(str, length, offset=0):
    return unpack('<{}I'.format(length), str[offset:offset + 4 * length])
