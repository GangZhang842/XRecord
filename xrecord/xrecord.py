import io
import struct
import typing
import collections


class XRecord(object):
    def __init__(self, fname, mode='r'):
        self.file_key = None
        self.file_value = None
        self.key_dic = collections.OrderedDict()
        self.pos = None
        self.template_bytes = None

        self.fname = fname
        self.mode = mode
        self.open(fname, mode)

        # for iterator
        self.keys = [key for key in self.key_dic]
        self._cur = 0

    def open(self, fname, mode='r'):
        self.fname = fname
        self.mode = mode
        self.pos = 0
        if mode == 'r':
            self.file_key = open(fname + '.key', "r")
            self.file_value = io.open(fname + '.value', "rb")
            self.template_bytes = bytearray(1024 * 1024)
            for line in self.file_key:
                s = line.split('\t')
                assert len(s) == 3
                self.key_dic[s[0]] = (int(s[1]), int(s[2]))
        elif mode == 'w':
            self.file_key = open(fname + '.key', "w")
            self.file_value = io.open(fname + '.value', "wb")
        elif mode == 'a':
            self.file_key = open(fname + '.key', "a")
            self.file_value = io.open(fname + '.value', "ab")
        else:
            raise Exception("invalid mode: {}".format(mode))

    def close(self):
        self.file_key.close()
        self.file_value.close()

    def __del__(self):
        self.close()

    def write(self, key, value):
        assert type(value) == bytes
        length = len(value)
        # record data
        self.file_value.write(value)

        # record key
        self.file_key.write("{}\t{}\t{}\n".format(key, self.pos, self.pos + length))
        self.pos = self.pos + length

    def read(self, key):
        start_byte, end_byte = self.key_dic[key]
        length = end_byte - start_byte

        # loc start
        self.file_value.seek(start_byte)
        if length > len(self.template_bytes):
            self.template_bytes = self.template_bytes.zfill(int(length * 1.5))

        template_bytes_view = memoryview(self.template_bytes)[:length]
        if self.file_value.readinto(template_bytes_view) != length:
            raise RuntimeError("Failed to read data!")

        return template_bytes_view
    
    # define iterator
    def __iter__(self):
        return self

    def reset(self):
        self._cur = 0

    def __next__(self):
        if self._cur < len(self):
            key = self.keys[self._cur]
            self._cur += 1
            start_byte, end_byte = self.key_dic[key]
            length = end_byte - start_byte

            if length > len(self.template_bytes):
                self.template_bytes = self.template_bytes.zfill(int(length * 1.5))

            template_bytes_view = memoryview(self.template_bytes)[:length]
            if self.file_value.readinto(template_bytes_view) != length:
                raise RuntimeError("Failed to read data!")
            return template_bytes_view
        else:
            raise StopIteration

    def __len__(self):
        return len(self.keys)