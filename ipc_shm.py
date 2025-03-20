from ctypes import *
import time
import numpy
import sys
import os
import mmap

IPC_RMID = 0
IPC_STAT = 2


class ipc_perm(Structure):
    _fields_ = [
        ("key", c_int),
        ("uid", c_uint16),
        ("gid", c_uint16),
        ("cuid", c_uint16),
        ("cgid", c_uint16),
        ("mode", c_uint16),
        ("seq", c_uint16)
    ]


class shmid_ds(Structure):
    _fields_ = [("shm_perm", ipc_perm),
                ("shm_segsz", c_int),
                ("shm_atime", c_uint),
                ("shm_dtime", c_uint),
                ("shm_ctime", c_uint),
                ("shm_cpid", c_uint16),
                ("shm_lpid", c_uint16),
                ("shm_nattch", c_int16),
                ("shm_npages", c_uint16),
                ("shm_lpid", POINTER(c_uint16)),
                ("attaches", c_char_p)
                ]


class ipc_shm:
    def __init__(self, SHM_KEY, SHM_SIZE, create=False, force_mmap=True):
        self.is_mmap = False  # 标记是否使用mmap回退
        self.mmap_file = None
        self.mmap_fd = None
        self.shm_size = SHM_SIZE

        if force_mmap:
            self._init_mmap(SHM_KEY, SHM_SIZE, create=create)
        else:
            try:
                # 尝试加载共享库
                try:
                    rt = CDLL('librt.so')
                except:
                    try:
                        rt = CDLL('librt.so.1')
                    except:
                        rt = CDLL('/usr/lib/libSystem.B.dylib')

                # 设置函数原型
                self.shmget = rt.shmget
                self.shmget.argtypes = [c_int, c_size_t, c_int]
                self.shmget.restype = c_int

                self.shmat = rt.shmat
                self.shmat.argtypes = [c_int, POINTER(c_void_p), c_int]
                self.shmat.restype = c_void_p

                self.shmctl = rt.shmctl
                self.shmctl.argtypes = [c_int, c_int, POINTER(shmid_ds)]
                self.shmctl.restype = c_int

                # 尝试获取共享内存
                flags = 0o666 | (0o1000 if create else 0)
                self.shmid = self.shmget(SHM_KEY, SHM_SIZE, flags)

                if self.shmid < 0:
                    # System V共享内存失败，回退到mmap文件
                    self._init_mmap(SHM_KEY, SHM_SIZE, create)
                else:
                    # 正常附加共享内存
                    self.addr = self.shmat(self.shmid, None, 0)
                    if self.addr == -1:
                        raise MemoryError("shmat failed")

            except Exception as e:
                # 如果仍然失败，尝试mmap回退
                if not self.is_mmap:
                    self._init_mmap(SHM_KEY, SHM_SIZE, create)
                else:
                    raise e

    @staticmethod
    def ftok(name, proj_id):
        try:
            rt = CDLL('librt.so')
        except:
            try:
                rt = CDLL('librt.so.1')
            except:
                rt = CDLL('/usr/lib/libSystem.B.dylib')

        name_p = create_string_buffer(name.encode('utf-8'))
        rt.ftok.argtypes = [c_char_p, c_int]
        rt.ftok.restype = c_int

        return rt.ftok(name_p.raw, proj_id)

    def _init_mmap(self, SHM_KEY, SHM_SIZE, create):
        """使用mmap文件回退的实现"""
        self.is_mmap = True

        # 生成唯一文件名
        filename = f"/dev/shm/shm_fallback_{SHM_KEY}"
        self.mmap_file = filename

        # 文件打开模式
        flags = os.O_CREAT | os.O_RDWR
        if not create:
            flags = os.O_RDWR

        try:
            # 打开/创建文件
            self.mmap_fd = os.open(filename, flags, 0o666)

            # 调整文件大小
            os.ftruncate(self.mmap_fd, SHM_SIZE)

            # 内存映射
            self.addr = mmap.mmap(
                self.mmap_fd,
                SHM_SIZE,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE
            ).__enter__()  # 保持mmap对象打开

        except Exception as e:
            raise RuntimeError(f"mmap fallback failed: {str(e)}")

    def remove(self):
        """清理资源"""
        if self.is_mmap:
            # 关闭mmap和文件
            if self.addr:
                self.addr.close()
            if self.mmap_fd:
                os.close(self.mmap_fd)
            if self.mmap_file and os.path.exists(self.mmap_file):
                os.unlink(self.mmap_file)
        else:
            # 原始共享内存清理
            rt = CDLL('librt.so')
            rt.shmdt(c_void_p(self.addr))
            rt.shmctl(self.shmid, IPC_RMID, None)

    # 以下原有方法保持不变，因为地址访问方式兼容
    def read(self, size, offset=None):
        offset = offset or 0
        if self.is_mmap:
            return bytes(self.addr[offset:offset+size])
        else:
            return string_at(self.addr + offset, size)

    def memory(self, offset=None):
        if self.is_mmap:
            return cast(c_void_p(id(self.addr) + (offset or 0)), POINTER(c_ubyte))
        else:
            return cast(self.addr + (offset or 0), POINTER(c_ubyte))

    def nd_array(self, shape, offset=None, dtype=numpy.uint8):
        offset = offset or 0
        if self.is_mmap:
            # mmap模式直接使用内存视图
            return numpy.frombuffer(
                self.addr,
                dtype=dtype,
                count=numpy.prod(shape),
                offset=offset
            ).reshape(shape)
        else:
            # System V模式使用ctypes指针
            ptr_type = POINTER(c_ubyte * (numpy.prod(shape) * numpy.dtype(dtype).itemsize))
            c_array = cast(self.addr + offset, ptr_type)
            np_array = numpy.ctypeslib.as_array(c_array.contents)
            return np_array.view(dtype).reshape(shape)

    def write(self, src, length=None, offset=None):
        length = length or len(src)
        offset = offset or 0
        if self.is_mmap:
            self.addr[offset:offset+length] = src[:length]
        else:
            memmove(c_void_p(self.addr + offset), src, length)
