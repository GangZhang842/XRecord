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

    def open(self, fname, mode='r'):
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
    
    def __seek_key(self, key):
        start_byte, _ = self.key_dic[key]
        self.file_value.seek(start_byte)

    def __seek_idx(self, idx):
        key = self.keys[idx]
        self.__seek_key(key)

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
    
    def read_iter(self, part=None):
        """
        part=(part_id, part_num) means the total dataset is split into "part_num" parts
        and the $part_id_{th}$ part is used, which is required in the distributed training system.
        By default, part is set as None, meaning that the whole dataset is loaded.
        """
        if part == None:
            part = (0, 1)
        
        assert type(part) in [list, tuple]
        assert len(part) == 2

        part_id = part[0]
        part_num = part[1]
        assert type(part_id) == int
        assert type(part_num) == int
        assert part_id < part_num

        sample_num = len(self)
        start_idx = (sample_num * part_id) // part_num
        end_idx = min((sample_num * (part_id + 1)) // part_num, sample_num)
        self.__seek_idx(start_idx)
        for idx in range(start_idx, end_idx):
            key_tmp = self.keys[idx]
            start_byte, end_byte = self.key_dic[key_tmp]
            length = end_byte - start_byte

            if length > len(self.template_bytes):
                self.template_bytes = self.template_bytes.zfill(int(length * 1.5))
            
            template_bytes_view = memoryview(self.template_bytes)[:length]
            if self.file_value.readinto(template_bytes_view) != length:
                raise RuntimeError("Failed to read data!")
            
            yield template_bytes_view

    def __len__(self):
        return len(self.keys)


class XRecordReadList(object):
    def __init__(self, fname_list):
        self.fname_list = fname_list
        assert type(self.fname_list) in [list, tuple]
        self.file_value_list = None
        self.key_dic = collections.OrderedDict()
        self.template_bytes = None

        self.open(self.fname_list)

        # for iterator
        self.keys = [key for key in self.key_dic]
    
    def open(self, fname_list):
        self.file_value_list = []
        self.template_bytes = bytearray(1024 * 1024)
        for i, fname in enumerate(self.fname_list):
            with open(fname + '.key', "r") as file_key:
                file_value = io.open(fname + '.value', "rb")
                self.file_value_list.append(file_value)
                for line in file_key:
                    s = line.split('\t')
                    assert len(s) == 3
                    self.key_dic[s[0]] = (i, int(s[1]), int(s[2]))
    
    def close(self):
        for i in range(len(self.file_value_list)):
            self.file_value_list[i].close()
    
    def __del__(self):
        self.close()

    def read(self, key):
        i, start_byte, end_byte = self.key_dic[key]
        length = end_byte - start_byte

        # loc start
        self.file_value_list[i].seek(start_byte)
        if length > len(self.template_bytes):
            self.template_bytes = self.template_bytes.zfill(int(length * 1.5))

        template_bytes_view = memoryview(self.template_bytes)[:length]
        if self.file_value_list[i].readinto(template_bytes_view) != length:
            raise RuntimeError("Failed to read data!")

        return template_bytes_view
    
    def read_iter(self, part=None):
        """
        part=(part_id, part_num) means the total dataset is split into "part_num" parts
        and the $part_id_{th}$ part is used, which is required in the distributed training system.
        By default, part is set as None, meaning that the whole dataset is loaded.
        """
        if part == None:
            part = (0, 1)
        
        assert type(part) in [list, tuple]
        assert len(part) == 2

        part_id = part[0]
        part_num = part[1]
        assert type(part_id) == int
        assert type(part_num) == int
        assert part_id < part_num

        sample_num = len(self)
        start_idx = (sample_num * part_id) // part_num
        end_idx = min((sample_num * (part_id + 1)) // part_num, sample_num)

        key_start = self.keys[start_idx]
        i_start, start_byte_start, end_byte_start = self.key_dic[key_start]
        file_cur = self.file_value_list[i_start]

        # loc start
        file_cur.seek(start_byte_start)

        i_last = i_start
        for idx in range(start_idx, end_idx):
            key_tmp = self.keys[idx]
            i, start_byte, end_byte = self.key_dic[key_tmp]
            length = end_byte - start_byte

            # reloc file
            if i != i_last:
                file_cur = self.file_value_list[i]
                file_cur.seek(0)
                i_last = i

            if length > len(self.template_bytes):
                self.template_bytes = self.template_bytes.zfill(int(length * 1.5))
            
            template_bytes_view = memoryview(self.template_bytes)[:length]
            if file_cur.readinto(template_bytes_view) != length:
                raise RuntimeError("Failed to read data!")
            
            yield template_bytes_view
    
    def __len__(self):
        return len(self.keys)