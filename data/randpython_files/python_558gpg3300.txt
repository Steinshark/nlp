
import ctypes
import numpy

import pyinterface
import libgpg3300
pIE = pyinterface.IdentiferElement


# Identifer Wrapper
# =================

class OverlappedProcess(pyinterface.Identifer):
    FLAG_SYNC = pIE('LAG_SYNC', libgpg3300.FLAG_SYNC)
    FLAG_ASYNC = pIE('FLAG_ASYNC', libgpg3300.FLAG_ASYNC)
    pass

class FileFormat(pyinterface.Identifer):
    FLAG_BIN = pIE('FLAG_BIN', libgpg3300.FLAG_BIN)
    FLAG_CSV = pIE('FLAG_CSV', libgpg3300.FLAG_CSV)
    pass

class DataFormat(pyinterface.Identifer):
    CONV_BIN = pIE('CONV_BIN', libgpg3300.CONV_BIN)
    CONV_PHYS = pIE('CONV_PHYS', libgpg3300.CONV_PHYS)
    pass

class AnalogOutputStatus(pyinterface.Identifer):
    DA_STATUS_STOP_SAMPLING = pIE('DA_STATUS_STOP_SAMPLING', libgpg3300.DA_STATUS_STOP_SAMPLING)
    DA_STATUS_WAIT_TRIGGER = pIE('DA_STATUS_WAIT_TRIGGER', libgpg3300.DA_STATUS_WAIT_TRIGGER)
    DA_STATUS_NOW_SAMPLING = pIE('DA_STATUS_NOW_SAMPLING', libgpg3300.DA_STATUS_NOW_SAMPLING)
    pass

class EventFactor(pyinterface.Identifer):
    DA_EVENT_STOP_TRIGGER = pIE('DA_EVENT_STOP_TRIGGER', libgpg3300.DA_EVENT_STOP_TRIGGER)
    DA_EVENT_STOP_FUNCTION = pIE('DA_EVENT_STOP_FUNCTION', libgpg3300.DA_EVENT_STOP_FUNCTION)
    DA_EVENT_STOP_SAMPLING = pIE('DA_EVENT_STOP_SAMPLING', libgpg3300.DA_EVENT_STOP_SAMPLING)
    DA_EVENT_RESET_IN = pIE('DA_EVENT_RESET_IN', libgpg3300.DA_EVENT_RESET_IN)
    DA_EVENT_CURRENT_OFF = pIE('DA_EVENT_CURRENT_OFF', libgpg3300.DA_EVENT_CURRENT_OFF)
    DA_EVENT_FIFO_EMPTY = pIE('DA_EVENT_FIFO_EMPTY', libgpg3300.DA_EVENT_FIFO_EMPTY)
    DA_EVENT_EX_INT = pIE('DA_EVENT_EX_INT', libgpg3300.DA_EVENT_EX_INT)
    DA_EVENT_EXOV_OFF = pIE('DA_EVENT_EXOV_OFF', libgpg3300.DA_EVENT_EXOV_OFF)
    DA_EVENT_OV_OFF = pIE('DA_EVENT_OV_OFF', libgpg3300.DA_EVENT_OV_OFF)
    pass

class OutputRangeElement(pyinterface.IdentiferElement):
    def __init__(self, name, id, min, max):
        self.name = name
        self.id = id
        self.min = min
        self.max = max
        pass
    
    def digitize(self, value, resolution):
        flag_not_array = False
        if type(value) in [int, float]: 
            flag_not_array = True
            value = [value]
            pass
        value = numpy.array(value)
        shape = value.shape
        value1d = value.ravel()
        bit = numpy.linspace(self.min, self.max, resolution)
        bins = bit[:-1] + (bit[1] - bit[0])/2.
        ret = numpy.digitize(value1d, bins)
        if flag_not_array: ret = ret[0]
        else: ret = ret.reshape(shape)
        return ret

_DA_0_1V = OutputRangeElement('DA_0_1V', libgpg3300.DA_0_1V, 0.0, 1.0)
_DA_0_2P5V = OutputRangeElement('DA_0_2P5V', libgpg3300.DA_0_2P5V, 0.0, 2.5)
_DA_0_5V = OutputRangeElement('DA_0_5V', libgpg3300.DA_0_5V, 0.0, 5.0)
_DA_0_10V = OutputRangeElement('DA_0_10V', libgpg3300.DA_0_10V, 0.0, 10.0)
_DA_1_5V = OutputRangeElement('DA_1_5V', libgpg3300.DA_1_5V, 1.0, 5.0)
_DA_0_20mA = OutputRangeElement('DA_0_20mA', libgpg3300.DA_0_20mA, 0.0, 0.020)
_DA_4_20mA = OutputRangeElement('DA_4_20mA', libgpg3300.DA_4_20mA, 0.004, 0.020)
_DA_0_1mA = OutputRangeElement('DA_0_1mA', libgpg3300.DA_0_1mA, 0.000, 0.001)
_DA_0_100mA = OutputRangeElement('DA_0_100mA', libgpg3300.DA_0_100mA, 0.000, 0.100)
_DA_1V = OutputRangeElement('DA_1V', libgpg3300.DA_1V, -1.0, 1.0)
_DA_2P5V = OutputRangeElement('DA_2P5V', libgpg3300.DA_2P5V, -2.5, 2.5)
_DA_5V = OutputRangeElement('DA_5V', libgpg3300.DA_5V, -5.0, 5.0)
_DA_10V = OutputRangeElement('DA_10V', libgpg3300.DA_10V, -10.0, 10.0)
_DA_20mA = OutputRangeElement('DA_20mA', libgpg3300.DA_20mA, -0.020, 0.020)

class OutputRange(pyinterface.Identifer):
    DA_0_1V = _DA_0_1V
    DA_0_2P5V = _DA_0_2P5V
    DA_0_5V = _DA_0_5V
    DA_0_10V = _DA_0_10V
    DA_1_5V = _DA_1_5V
    DA_0_20mA = _DA_0_20mA
    DA_4_20mA = _DA_4_20mA
    DA_0_1mA = _DA_0_1mA
    DA_0_100mA = _DA_0_100mA
    DA_1V = _DA_1V
    DA_2P5V = _DA_2P5V
    DA_5V = _DA_5V
    DA_10V = _DA_10V
    DA_20mA = _DA_20mA
    
    def __init__(self, ids):
        self.available = []
        for key in dir(self):
            if not key.isupper(): continue
            value = self.__getattribute__(key)
            if value & ids: self.available.append(value)
            continue
        
        def verify(to_be_verified):
            for id in self.available:
                if id == to_be_verified: return id
                continue
            raise ValueError, '%s is not available'%(to_be_verified)
            pass

        self.verify = verify
        pass

    def __repr__(self):
        maxlen = max([len(str(d)) for d in self.available])
        fmt = '%%-%ds'%(maxlen)
        ret = '%s\n'%(self.__class__)
        for value in self.available:
            ret += fmt%(value) + ' :  %d\n'%(value)
            continue
        return ret


class DataTransferArchtecture(pyinterface.Identifer):
    DA_IO_SAMPLING = pIE('DA_IO_SAMPLING', libgpg3300.DA_IO_SAMPLING)
    DA_FIFO_SAMPLING = pIE('DA_FIFO_SAMPLING', libgpg3300.DA_FIFO_SAMPLING)
    DA_MEM_SAMPLING = pIE('DA_MEM_SAMPLING', libgpg3300.DA_MEM_SAMPLING)
    pass

class TriggerPoint(pyinterface.Identifer):
    DA_TRIG_START = pIE('DA_TRIG_START', libgpg3300.DA_TRIG_START)
    DA_TRIG_STOP = pIE('DA_TRIG_STOP', libgpg3300.DA_TRIG_STOP)
    DA_TRIG_START_STOP = pIE('DA_TRIG_START_STOP', libgpg3300.DA_TRIG_START_STOP)
    pass

class TriggerMode(pyinterface.Identifer):
    DA_FREERUN = pIE('DA_FREERUN', libgpg3300.DA_FREERUN)
    DA_EXTTRG = pIE('DA_EXTTRG', libgpg3300.DA_EXTTRG)
    DA_EXTTRG_DI = pIE('DA_EXTTRG_DI', libgpg3300.DA_EXTTRG_DI)
    pass

class Polarity(pyinterface.Identifer):
    DA_DOWN_EDGE = pIE('DA_DOWN_EDGE', libgpg3300.DA_DOWN_EDGE)
    DA_UP_EDGE = pIE('DA_UP_EDGE', libgpg3300.DA_UP_EDGE)
    pass

class PulsePolarity(pyinterface.Identifer):
    DA_LOW_PULSE = pIE('DA_LOW_PULSE', libgpg3300.DA_LOW_PULSE)
    DA_HIGH_PULSE = pIE('DA_HIGH_PULSE', libgpg3300.DA_HIGH_PULSE)
    pass

class SimultaneousOutput(pyinterface.Identifer):
    DA_NORMAL_MODE = pIE('DA_NORMAL_MODE', libgpg3300.DA_NORMAL_MODE)
    DA_FAST_MODE = pIE('DA_FAST_MODE', libgpg3300.DA_FAST_MODE)
    pass

class Isolation(pyinterface.Identifer):
    DA_ISOLATION = pIE('DA_ISOLATION', libgpg3300.DA_ISOLATION)
    DA_NOT_ISOLATION = pIE('DA_NOT_ISOLATION', libgpg3300.DA_NOT_ISOLATION)
    pass

class RangeType(pyinterface.Identifer):
    DA_RANGE_UNIPOLAR = pIE('DA_RANGE_UNIPOLAR', libgpg3300.DA_RANGE_UNIPOLAR)
    DA_RANGE_BIPOLAR = pIE('DA_RANGE_BIPOLAR', libgpg3300.DA_RANGE_BIPOLAR)
    pass

class Filter(pyinterface.Identifer):
    DA_FILTER_OFF = pIE('DA_FILTER_OFF', libgpg3300.DA_FILTER_OFF)
    DA_FILTER_ON = pIE('DA_FILTER_ON', libgpg3300.DA_FILTER_ON)
    pass

class WaveformGenerationMode(pyinterface.Identifer):
    DA_MODE_CUT = pIE('DA_MODE_CUT', libgpg3300.DA_MODE_CUT)
    DA_MODE_SYNTHE = pIE('DA_MODE_SYNTHE', libgpg3300.DA_MODE_SYNTHE)
    pass

class RepeateMode(pyinterface.Identifer):
    DA_MODE_REPEAT = pIE('DA_MODE_REPEAT', libgpg3300.DA_MODE_REPEAT)
    DA_MODE_SINGLE = pIE('DA_MODE_SINGLE', libgpg3300.DA_MODE_SINGLE)
    pass

class CounterClear(pyinterface.Identifer):
    DA_COUNTER_CLEAR = pIE('DA_COUNTER_CLEAR', libgpg3300.DA_COUNTER_CLEAR)
    DA_COUNTER_NONCLEAR = pIE('DA_COUNTER_NONCLEAR', libgpg3300.DA_COUNTER_NONCLEAR)
    pass

class DaLatch(pyinterface.Identifer):
    DA_LATCH_CLEAR = pIE('DA_LATCH_CLEAR', libgpg3300.DA_LATCH_CLEAR)
    DA_LATCH_NONCLEAR = pIE('DA_LATCH_NONCLEAR', libgpg3300.DA_LATCH_NONCLEAR)
    pass

class ClockSource(pyinterface.Identifer):
    DA_CLOCK_TIMER = pIE('DA_CLOCK_TIMER', libgpg3300.DA_CLOCK_TIMER)
    DA_CLOCK_FIXED = pIE('DA_CLOCK_FIXED', libgpg3300.DA_CLOCK_FIXED)
    pass

class TriggerConfig(pyinterface.Identifer):
    DA_EXTRG_IN = pIE('DA_EXTRG_IN', libgpg3300.DA_EXTRG_IN)
    DA_EXTRG_OUT = pIE('DA_EXTRG_OUT', libgpg3300.DA_EXTRG_OUT)
    DA_EXINT_IN = pIE('DA_EXINT_IN', libgpg3300.DA_EXINT_IN)
    pass

class ClockConfig(pyinterface.Identifer):
    DA_EXCLK_IN = pIE('DA_EXCLK_IN', libgpg3300.DA_EXCLK_IN)
    DA_EXCLK_OUT = pIE('DA_EXCLK_OUT', libgpg3300.DA_EXCLK_OUT)
    pass

class ResetPolarity(pyinterface.Identifer):
    DA_RESET_DOWN_EDGE = pIE('DA_RESET_DOWN_EDGE', libgpg3300.DA_RESET_DOWN_EDGE)
    DA_RESET_UP_EDGE = pIE('DA_RESET_UP_EDGE', libgpg3300.DA_RESET_UP_EDGE)
    pass

class ExternalTriggerPolarity(pyinterface.Identifer):
    DA_EXTRG_DOWN_EDGE = pIE('DA_EXTRG_DOWN_EDGE', libgpg3300.DA_EXTRG_DOWN_EDGE)
    DA_EXTRG_UP_EDGE = pIE('DA_EXTRG_UP_EDGE', libgpg3300.DA_EXTRG_UP_EDGE)
    pass

class SynchronousAnalogOutput(pyinterface.Identifer):
    DA_MASTER_MODE = pIE('DA_MASTER_MODE', libgpg3300.DA_MASTER_MODE)
    DA_SLAVE_MODE = pIE('DA_SLAVE_MODE', libgpg3300.DA_SLAVE_MODE)
    pass

class SynchronousNumber(pyinterface.Identifer):
    DA_SYNC_NUM_1 = pIE('DA_SYNC_NUM_1', libgpg3300.DA_SYNC_NUM_1)
    DA_SYNC_NUM_2 = pIE('DA_SYNC_NUM_2', libgpg3300.DA_SYNC_NUM_2)
    DA_SYNC_NUM_3 = pIE('DA_SYNC_NUM_3', libgpg3300.DA_SYNC_NUM_3)
    DA_SYNC_NUM_4 = pIE('DA_SYNC_NUM_4', libgpg3300.DA_SYNC_NUM_4)
    DA_SYNC_NUM_5 = pIE('DA_SYNC_NUM_5', libgpg3300.DA_SYNC_NUM_5)
    DA_SYNC_NUM_6 = pIE('DA_SYNC_NUM_6', libgpg3300.DA_SYNC_NUM_6)
    DA_SYNC_NUM_7 = pIE('DA_SYNC_NUM_7', libgpg3300.DA_SYNC_NUM_7)
    pass

class AfterCloseExOutput(pyinterface.Identifer):
    DA_OUTPUT_RESET = pIE('DA_OUTPUT_RESET', libgpg3300.DA_OUTPUT_RESET)
    DA_OUTPUT_MAINTAIN = pIE('DA_OUTPUT_MAINTAIN', libgpg3300.DA_OUTPUT_MAINTAIN)
    pass

class Reset(pyinterface.Identifer):
    DA_RESET_ON = pIE('DA_RESET_ON', libgpg3300.DA_RESET_ON)
    DA_RESET_OFF = pIE('DA_RESET_OFF', libgpg3300.DA_RESET_OFF)
    pass

class CPZ360810DioFunction(pyinterface.Identifer):
    DA_EX_DIO1 = pIE('DA_EX_DIO1', libgpg3300.DA_EX_DIO1)
    DA_EX_DIO2 = pIE('DA_EX_DIO2', libgpg3300.DA_EX_DIO2)
    DA_EX_DIO3 = pIE('DA_EX_DIO3', libgpg3300.DA_EX_DIO3)
    DA_EX_DIO4 = pIE('DA_EX_DIO4', libgpg3300.DA_EX_DIO4)
    DA_EX_DIO5 = pIE('DA_EX_DIO5', libgpg3300.DA_EX_DIO5)
    pass

class PCI3525CnFunction(pyinterface.Identifer):
    DA_CN_FREE = pIE('DA_CN_FREE', libgpg3300.DA_CN_FREE)
    DA_CN_EXTRG_IN = pIE('DA_CN_EXTRG_IN', libgpg3300.DA_CN_EXTRG_IN)
    DA_CN_EXTRG_OUT = pIE('DA_CN_EXTRG_OUT', libgpg3300.DA_CN_EXTRG_OUT)
    DA_CN_EXCLK_IN = pIE('DA_CN_EXCLK_IN', libgpg3300.DA_CN_EXCLK_IN)
    DA_CN_EXCLK_OUT = pIE('DA_CN_EXCLK_OUT', libgpg3300.DA_CN_EXCLK_OUT)
    DA_CN_EXINT_IN = pIE('DA_CN_EXINT_IN', libgpg3300.DA_CN_EXINT_IN)
    DA_CN_ATRG_IN = pIE('DA_CN_ATRG_IN', libgpg3300.DA_CN_ATRG_IN)
    DA_CN_DI = pIE('DA_CN_DI', libgpg3300.DA_CN_DI)
    DA_CN_DO = pIE('DA_CN_DO', libgpg3300.DA_CN_DO)
    DA_CN_DAOUT = pIE('DA_CN_DAOUT', libgpg3300.DA_CN_DAOUT)
    DA_CN_OPEN = pIE('DA_CN_OPEN', libgpg3300.DA_CN_OPEN)
    DA_CN_EX_OUT1 = pIE('DA_CN_EX_OUT1', libgpg3300.DA_CN_EX_OUT1)
    DA_CN_EX_OUT2 = pIE('DA_CN_EX_OUT2', libgpg3300.DA_CN_EX_OUT2)
    pass

class PCI3525ExternalTriggerPolarity(pyinterface.Identifer):
    DA_START_DOWN_EDGE = pIE('DA_START_DOWN_EDGE', libgpg3300.DA_START_DOWN_EDGE)
    DA_START_UP_EDGE = pIE('DA_START_UP_EDGE', libgpg3300.DA_START_UP_EDGE)
    DA_STOP_DOWN_EDGE = pIE('DA_STOP_DOWN_EDGE', libgpg3300.DA_STOP_DOWN_EDGE)
    DA_STOP_UP_EDGE = pIE('DA_STOP_UP_EDGE', libgpg3300.DA_STOP_UP_EDGE)
    pass

class FifoTriggerLevel(pyinterface.Identifer):
    DA_TRG_FREERUN = pIE('DA_TRG_FREERUN', libgpg3300.DA_TRG_FREERUN)
    DA_TRG_EXTTRG = pIE('DA_TRG_EXTTRG', libgpg3300.DA_TRG_EXTTRG)
    DA_TRG_ATRG = pIE('DA_TRG_ATRG', libgpg3300.DA_TRG_ATRG)
    DA_TRG_SIGTIMER = pIE('DA_TRG_SIGTIMER', libgpg3300.DA_TRG_SIGTIMER)
    DA_TRG_CNT_EQ = pIE('DA_TRG_CNT_EQ', libgpg3300.DA_TRG_CNT_EQ)
    DA_TRG_Z_CLR = pIE('DA_TRG_Z_CLR', libgpg3300.DA_TRG_Z_CLR)
    DA_TRG_AD_START = pIE('DA_TRG_AD_START', libgpg3300.DA_TRG_AD_START)
    DA_TRG_AD_STOP = pIE('DA_TRG_AD_STOP', libgpg3300.DA_TRG_AD_STOP)
    DA_TRG_AD_PRETRG = pIE('DA_TRG_AD_PRETRG', libgpg3300.DA_TRG_AD_PRETRG)
    DA_TRG_AD_POSTTRG = pIE('DA_TRG_AD_POSTTRG', libgpg3300.DA_TRG_AD_POSTTRG)
    DA_TRG_SMPLNUM = pIE('DA_TRG_SMPLNUM', libgpg3300.DA_TRG_SMPLNUM)
    DA_TRG_FIFO_EMPTY = pIE('DA_TRG_FIFO_EMPTY', libgpg3300.DA_TRG_FIFO_EMPTY)
    DA_TRG_SYNC1 = pIE('DA_TRG_SYNC1', libgpg3300.DA_TRG_SYNC1)
    DA_TRG_SYNC2 = pIE('DA_TRG_SYNC2', libgpg3300.DA_TRG_SYNC2)
    DA_FIFORESET = pIE('DA_FIFORESET', libgpg3300.DA_FIFORESET)
    DA_RETRG = pIE('DA_RETRG', libgpg3300.DA_RETRG)
    pass

class SimultaneousOutputSet(pyinterface.Identifer):
    DA_NORMAL_OUTPUT = pIE('DA_NORMAL_OUTPUT', libgpg3300.DA_NORMAL_OUTPUT)
    DA_SYNC_OUTPUT = pIE('DA_SYNC_OUTPUT', libgpg3300.DA_SYNC_OUTPUT)
    pass

class Volume(pyinterface.Identifer):
    DA_ADJUST_BIOFFSET = pIE('DA_ADJUST_BIOFFSET', libgpg3300.DA_ADJUST_BIOFFSET)
    DA_ADJUST_UNIOFFSET = pIE('DA_ADJUST_UNIOFFSET', libgpg3300.DA_ADJUST_UNIOFFSET)
    DA_ADJUST_BIGAIN = pIE('DA_ADJUST_BIGAIN', libgpg3300.DA_ADJUST_BIGAIN)
    DA_ADJUST_UNIGAIN = pIE('DA_ADJUST_UNIGAIN', libgpg3300.DA_ADJUST_UNIGAIN)
    pass

class CalibrationItem(pyinterface.Identifer):
    DA_ADJUST_UP = pIE('DA_ADJUST_UP', libgpg3300.DA_ADJUST_UP)
    DA_ADJUST_DOWN = pIE('DA_ADJUST_DOWN', libgpg3300.DA_ADJUST_DOWN)
    DA_ADJUST_STORE = pIE('DA_ADJUST_STORE', libgpg3300.DA_ADJUST_STORE)
    DA_ADJUST_STANDBY = pIE('DA_ADJUST_STANDBY', libgpg3300.DA_ADJUST_STANDBY)
    DA_ADJUST_NOT_STORE = pIE('DA_ADJUST_NOT_STORE', libgpg3300.DA_ADJUST_NOT_STORE)
    DA_ADJUST_STORE_INITAREA = pIE('DA_ADJUST_STORE_INITAREA', libgpg3300.DA_ADJUST_STORE_INITAREA)
    DA_ADJUST_READ_FACTORY = pIE('DA_ADJUST_READ_FACTORY', libgpg3300.DA_ADJUST_READ_FACTORY)
    DA_ADJUST_READ_USER = pIE('DA_ADJUST_READ_USER', libgpg3300.DA_ADJUST_READ_USER)
    pass

class DataIdentifer(pyinterface.Identifer):
    DA_DATA_PHYSICAL = pIE('DA_DATA_PHYSICAL', libgpg3300.DA_DATA_PHYSICAL)
    DA_DATA_BIN8 = pIE('DA_DATA_BIN8', libgpg3300.DA_DATA_BIN8)
    DA_DATA_BIN12 = pIE('DA_DATA_BIN12', libgpg3300.DA_DATA_BIN12)
    DA_DATA_BIN16 = pIE('DA_DATA_BIN16', libgpg3300.DA_DATA_BIN16)
    DA_DATA_BIN24 = pIE('DA_DATA_BIN24', libgpg3300.DA_DATA_BIN24)
    DA_DATA_BIN14 = pIE('DA_DATA_BIN14', libgpg3300.DA_DATA_BIN14)
    pass

class DataConversion(pyinterface.Identifer):
    DA_CONV_SMOOTH = pIE('DA_CONV_SMOOTH', libgpg3300.DA_CONV_SMOOTH)
    DA_CONV_AVERAGE1 = pIE('DA_CONV_AVERAGE1', libgpg3300.DA_CONV_AVERAGE1)
    DA_CONV_AVERAGE2 = pIE('DA_CONV_AVERAGE2', libgpg3300.DA_CONV_AVERAGE2)
    pass

class SamplingFreq(object):
    @classmethod
    def verify(cls, value):
        value = float(value)
        if value < 0:
            errmsg = '%s.%s: value should be 0.0 or >0.01 (value=%d)'%(cls.__module__, cls.__name__, value)
            raise ValueError, errmsg
        return value

class SamplingRepeat(object):
    @classmethod
    def verify(cls, value):
        value = int(value)
        if (value < 0)or(value > 65535):
            errmsg = '%s.%s: value should be 0 - 65535 (value=%d)'%(cls.__module__, cls.__name__, value)
            raise ValueError, errmsg
        return value

class TriggerDelay(object):
    @classmethod
    def verify(cls, value):
        value = int(value)
        if (value < 0) or (value > 1073741824):
            errmsg = '%s.%s: value should be 0 - 1073741824 (value=%d)'%(cls.__module__, cls.__name__, value)
            raise ValueError, errmsg
        return value


# Error Wrapper
# =============

class ErrorGPG3300(pyinterface.ErrorCode):
    DA_ERROR_SUCCESS               = libgpg3300.DA_ERROR_SUCCESS
    DA_ERROR_NOT_DEVICE            = libgpg3300.DA_ERROR_NOT_DEVICE
    DA_ERROR_NOT_OPEN              = libgpg3300.DA_ERROR_NOT_OPEN
    DA_ERROR_INVALID_DEVICE_NUMBER = libgpg3300.DA_ERROR_INVALID_DEVICE_NUMBER
    DA_ERROR_ALREADY_OPEN          = libgpg3300.DA_ERROR_ALREADY_OPEN
    DA_ERROR_NOT_SUPPORTED         = libgpg3300.DA_ERROR_NOT_SUPPORTED
    DA_ERROR_NOW_SAMPLING          = libgpg3300.DA_ERROR_NOW_SAMPLING
    DA_ERROR_STOP_SAMPLING         = libgpg3300.DA_ERROR_STOP_SAMPLING
    DA_ERROR_START_SAMPLING        = libgpg3300.DA_ERROR_START_SAMPLING
    DA_ERROR_SAMPLING_TIMEOUT      = libgpg3300.DA_ERROR_SAMPLING_TIMEOUT
    DA_ERROR_INVALID_PARAMETER     = libgpg3300.DA_ERROR_INVALID_PARAMETER
    DA_ERROR_ILLEGAL_PARAMETER     = libgpg3300.DA_ERROR_ILLEGAL_PARAMETER
    DA_ERROR_NULL_POINTER          = libgpg3300.DA_ERROR_NULL_POINTER
    DA_ERROR_SET_DATA              = libgpg3300.DA_ERROR_SET_DATA
    DA_ERROR_USED_AD               = libgpg3300.DA_ERROR_USED_AD
    DA_ERROR_FILE_OPEN             = libgpg3300.DA_ERROR_FILE_OPEN
    DA_ERROR_FILE_CLOSE            = libgpg3300.DA_ERROR_FILE_CLOSE
    DA_ERROR_FILE_READ             = libgpg3300.DA_ERROR_FILE_READ
    DA_ERROR_FILE_WRITE            = libgpg3300.DA_ERROR_FILE_WRITE
    DA_ERROR_INVALID_DATA_FORMAT   = libgpg3300.DA_ERROR_INVALID_DATA_FORMAT
    DA_ERROR_INVALID_AVERAGE_OR_SMOOTHING = libgpg3300.DA_ERROR_INVALID_AVERAGE_OR_SMOOTHING
    DA_ERROR_INVALID_SOURCE_DATA   = libgpg3300.DA_ERROR_INVALID_SOURCE_DATA
    DA_ERROR_NOT_ALLOCATE_MEMORY   = libgpg3300.DA_ERROR_NOT_ALLOCATE_MEMORY
    DA_ERROR_NOT_LOAD_DLL          = libgpg3300.DA_ERROR_NOT_LOAD_DLL
    DA_ERROR_CALL_DLL              = libgpg3300.DA_ERROR_CALL_DLL
    DA_ERROR_CALIBRATION           = libgpg3300.DA_ERROR_CALIBRATION
    DA_ERROR_USBIO_FAILED          = libgpg3300.DA_ERROR_USBIO_FAILED
    DA_ERROR_USBIO_TIMEOUT         = libgpg3300.DA_ERROR_USBIO_TIMEOUT
    DA_ERROR_USBLIB_LOAD_FAILED    = libgpg3300.DA_ERROR_USBLIB_LOAD_FAILED
    _success = DA_ERROR_SUCCESS
    pass



# ==========================
# GPG-3300 Python Controller
# ==========================

class gpg3300(object):
    def __init__(self, ndev=1, remote=False):
        initialize = not remote
        self.ctrl = gpg3300_controller(ndev, initialize)
        self._board_name = self.ctrl._latest_device_info.ulBoardType
        self._board_id = self.ctrl._latest_device_info.ulBoardID
        self._sampling_mode = self.ctrl._latest_device_info.ulSamplingMode
        self._ch_count = self.ctrl._latest_device_info.ulChCount
        self._resolution = self.ctrl._latest_device_info.ulResolution
        self._output_range = OutputRange(self.ctrl._latest_device_info.ulRange)
        self._isolation = self.ctrl._latest_device_info.ulIsolation
        self._ch_di = self.ctrl._latest_device_info.ulDi
        self._ch_do = self.ctrl._latest_device_info.ulDo
        
        self._status_open = True
        self._status_output = False
        self._output_ch = range(self._ch_count)
        self._status_output_value = [0] * self._ch_count
        self._status_output_value_digit = [0] * self._ch_count
        self._status_output_range = [0] * self._ch_count
        self._status_di = [0] * self._ch_di
        self._status_do = [0] * self._ch_do
        self.set_range(self._output_range.available[0])
        pass
    
    def open(self):
        self.ctrl.open()
        self._status_open = True
        return
    
    def close(self, output='terminate'):
        """
        output : 'terminate', 'continue'
        """
        if output=='terminate':
            self.ctrl.close()
            self._status_output = False
        elif output=='continue':
            self.ctrl.close_ex(AfterCloseExOutput.DA_OUTPUT_MAINTAIN)
        else:
            raise ValueError, "output = 'terminate', 'conitnue'"
        self._status_open = False
        return
        
    def set_da_value(self, value, ch=None):
        self._set_output(self._status_output_value, value, ch, dtype=float)
        if self._status_output: self.output()
        return
        
    def set_range(self, da_range, ch=None):
        self._set_output(self._status_output_range, da_range, ch, self._output_range.verify)
        if self._status_output: self.output()
        return

    def _set_output(self, config, value, ch, checker=None, dtype=None):
        value_is_list = (type(value) in [list, tuple, numpy.ndarray])
        ch_is_list = (type(ch) in [list, tuple])
        if (not value_is_list) & (not ch_is_list):
            if ch is None:
                value = [value] * self._ch_count
                ch = range(self._ch_count)
            else:
                value = [value]
                ch = [ch]
                pass
        elif (not value_is_list) & (ch_is_list):
            value = [value] * len(ch)
        elif (value_is_list) & (not ch_is_list):
            ch = range(len(value))
            pass
        
        if not type(checker) in [list, tuple]: checker = [checker] * len(value)
        
        for c, v, check in zip(ch, value, checker):
            if check is not None: v = check(v)
            if dtype is not None: v = dtype(v)
            config[c] = v
            continue
        return
    
    def output(self):
        output_data = self._digitize(self._status_output_value, self._status_output_range)
        self._output(output_data, self._output_ch, self._status_output_range)
        self._status_output = True
        return
        
    def stop_output(self):
        output_data = self._digitize([0]*self._ch_count, self._status_output_range)
        self._output(output_data, self._output_ch, self._status_output_range)
        self._status_output = False
        return
    
    def _output(self, data, ch, range):
        ch = [i+1 for i in ch]
        self.ctrl.output_da(data, ch, range)
        self._status_output_value_digit = data
        return

    def _digitize(self, data, drange):
        return [r.digitize(d, 2**self._resolution) for d,r in zip(data, drange)]
    
    def output_series(self, data, ch, range, freq, repeat, syncflag):
        datasize = len(data)
        data = OutputRange.verify(range).digitize(data, 2**self._resolution)
        self.ctrl.set_board_config(datasize, None, 0)
        self.ctrl.set_sampling_config(ch, range, freq, repeat)
        self.ctrl.clear_sampling_data()
        self.ctrl.set_sampling_data(data)
        self.ctrl.start_sampling(syncflag)
        return
    
    def output_sync(self, data, ch, range, freq, repeat, master_slave):
        datasize = len(data)
        data = OutputRange.verify(range).digitize(data, 2**self._resolution)
        self.ctrl.set_board_config(datasize, None, 0)
        self.ctrl.set_sampling_config(ch, range, freq, repeat)
        self.ctrl.clear_sampling_data()
        self.ctrl.set_sampling_data(data)
        self.ctrl.sync_sampling(master_slave)
        return
    
    def stop_sampling(self):
        self.ctrl.stop_sampling()
        self.stop_output()
        return
    
    def get_status(self):
        return self.ctrl.get_status()
    
    def read_board_name(self):
        return self._board_name
    
    def read_board_id(self):
        return self._board_id
        
    def read_sampling_mode(self):
        return self._sampling_mode
        
    def read_ch_count(self):
        return self._ch_count

    def read_resolution(self):
        return self._resolution

    def read_output_range(self):
        return self._output_range

    def read_isolation(self):
        return self._isolation

    def read_ch_di(self):
        return self._ch_di

    def read_ch_do(self):
        return self._ch_do

    def read_status_open(self):
        return self._status_open

    def read_status_output(self):
        return self._status_output

    def read_output_ch(self):
        return self._output_ch

    def read_status_output_value(self):
        return self._status_output_value

    def read_status_output_value_digit(self):
        return self._status_output_value_digit

    def read_status_output_range(self):
        return self._status_output_range

    def read_status_di(self):
        return self._status_di

    def read_status_do(self):
        return self._status_do

    def print_status(self):
        msg = 'open: %s, output: %s\n'%(self._status_open, self._status_output)
        msg += 'ch: %s\n'%(self._output_ch)
        msg += 'output(value): %s\n'%(self._status_output_value)
        msg += 'output(bit): %s\n'%(self._status_output_value_digit)
        msg += 'output(range): %s\n'%([str(r) for r in self._status_output_range])
        msg += 'DI: %s, DO: %s\n'%(self._status_di, self._status_do)
        print(msg)
        return msg


class gpg3300_controller(object):
    ndev = int()
    
    def __init__(self, ndev=1, initialize=True):
        self.ndev = ndev
        if initialize: self.initialize()
        return
    
    def _log(self, msg):
        print('Interface GPG3300(%d): %s'%(self.ndev, msg))
        return
        
    def _error_check(self, error_no):
        ErrorGPG3300.check(error_no)
        return
        
    def initialize(self):
        self.open()
        self.get_device_info()
        self.get_sampling_config()
        try:
            self.get_mode()
        except pyinterface.InterfaceError: pass
        return
        
    def open(self):
        self._log('open')
        ret = libgpg3300.DaOpen(self.ndev)
        self._error_check(ret)
        return
    
    def close(self):
        self._log('close')
        ret = libgpg3300.DaClose(self.ndev)
        self._error_check(ret)
        return
        
    def close_ex(self, final_state):
        AfterCloseExOutput.verify(final_state)
        identifer = AfterCloseExOutput.get_id(final_state)
        self._log('close_ex, %s %s'%(final_state, identifer))
        ret = libgpg3300.DaCloseEx(self.ndev, final_state)
        self._error_check(ret)
        return
    
    def get_device_info(self):
        self._log('get_device_info')
        info = libgpg3300.DABOARDSPEC()
        ret = libgpg3300.DaGetDeviceInfo(self.ndev, info)
        self._error_check(ret)
        self._latest_device_info = info
        resolution = info.ulResolution
        if resolution==8: self._data_size = ctypes.c_ubyte
        if resolution==12: self._data_size = ctypes.c_ushort
        if resolution==16: self._data_size = ctypes.c_ushort
        if resolution==24: self._data_size = ctypes.c_ulong
        print(info)
        return info
    
    def set_board_config(self, buffer_size, callback_func, user_data):
        self._log('set_board_config')
        ret = libgpg3300.DaSetBoardConfig(self.ndev, buffer_size, None, callback_func, user_data)
        self._error_check(ret)
        return
    
    def get_board_config(self):
        self._log('get_board_config')
        size = ctypes.c_ulong(0)
        factor = ctypes.c_ulong(0)
        ret = libgpg3300.DaGetBoardConfig(self.ndev, size, factor)
        self._error_check(ret)
        size = size.value
        factor = factor.value
        print('SmplBufferSize = %d, SmplEventFactor = %d (%s)'%(size, factor, EventFactor.get_id(factor)))
        return size, factor

    def set_sampling_config(self, chs, ranges, freq=None, repeat=None, mode=None, trig_mode=None,
                            trig_point=None, trig_delay=None, clock_edge=None, trig_edge=None, trig_di=None):
        self._log('set_sampling_config')
        config = self._create_sampling_config(chs, ranges, freq, repeat, mode, trig_mode, trig_point, trig_delay,
                                              clock_edge, trig_edge, trig_di)
        ret = libgpg3300.DaSetSamplingConfig(self.ndev, config)
        self._error_check(ret)
        self._latest_sampling_config = config
        return
    
    def _create_sampling_config(self, chs, ranges, freq, repeat, mode, trig_mode,
                                trig_point, trig_delay, clock_edge, trig_edge, trig_di):
        ch_config = self._create_sampling_ch_config(chs, ranges)
        
        if mode is None: mode = self._latest_sampling_config.ulSamplingMode
        if freq is None: freq = self._latest_sampling_config.fSmplFreq
        if repeat is None: repeat = self._latest_sampling_config.ulSmplRepeat
        if trig_mode is None: trig_mode = self._latest_sampling_config.ulTrigMode
        if trig_point is None: trig_point = self._latest_sampling_config.ulTrigPoint
        if trig_delay is None: trig_delay = self._latest_sampling_config.ulTrigDelay
        if clock_edge is None: clock_edge = self._latest_sampling_config.ulEClkEdge
        if trig_edge is None: trig_edge = self._latest_sampling_config.ulTrigEdge
        if trig_di is None: trig_di = self._latest_sampling_config.ulTrigDI
        
        config = libgpg3300.DASMPLREQ()
        config.ulChCount = len(chs)
        for i in range(len(ch_config)): config.SmplChReq[i] = ch_config[i]
        config.ulSamplingMode = DataTransferArchtecture.verify(mode)
        config.fSmplFreq = SamplingFreq.verify(freq)
        config.ulSmplRepeat = SamplingRepeat.verify(repeat)
        config.ulTrigMode = TriggerMode.verify(trig_mode)
        config.ulTrigPoint = TriggerPoint.verify(trig_point)
        config.ulTrigDelay = TriggerDelay.verify(trig_delay)
        config.ulEClkEdge = Polarity.verify(clock_edge)
        config.ulTrigEdge = Polarity.verify(trig_edge)
        config.ulTrigDI = trig_di
        #print(config)
        return config
    
    def _create_sampling_ch_config(self, chs, ranges):
        if not type(chs) in [list, tuple]: chs = [chs,]
        ch_no = len(chs)
        
        if not type(ranges) in [list, tuple]: ranges = [ranges] * ch_no
        
        ch_config = (libgpg3300.DASMPLCHREQ * ch_no)()
        for i, (ch, rang) in enumerate(zip(chs, ranges)):
            ch_config[i].ulChNo = ch
            ch_config[i].ulRange = OutputRange.verify(rang)
            continue
        
        return ch_config
    
    def get_sampling_config(self):
        self._log('get_sampling_config')
        config = libgpg3300.DASMPLREQ()
        ret = libgpg3300.DaGetSamplingConfig(self.ndev, config)
        self._error_check(ret)
        self._latest_sampling_config = config
        print(config)
        return config

    def set_mode(self, mode):
        self._log('set_mode')
        ret = libgpg3300.DaSetMode(self.ndev, mode)
        self._error_check(ret)
        self._latest_mode = mode
        return

    def get_mode(self):
        self._log('get_mode')
        mode = libgpg3300.DAMODEREQ()
        ret = libgpg3300.DaGetMode(self.ndev, mode)
        self._latest_mode = mode
        self._error_check(ret)
        return

    def set_sampling_data(self, data):
        self._log('set_sampling_data, num=%d'%(len(data)))
        data = numpy.array(data)
        chnum = self._latest_sampling_config.ulChCount        
        
        if data.ndim==1: 
            data = data[:,None] * numpy.ones(chnum, int)[None,:]
            pass
        
        d = (ctypes.c_int * data.size)()
        for i in range(len(data)):
            for j in range(chnum): d[i*chnum+j] = data[i][j]
            continue
            
        ret = libgpg3300.DaSetSamplingData(self.ndev, ctypes.byref(d), len(data))
        self._error_check(ret)
        return

    def clear_sampling_data(self):
        self._log('clear_sampling_data')
        ret = libgpg3300.DaClearSamplingData(self.ndev)
        self._error_check(ret)
        return

    def start_sampling(self, syncflag=1):
        self._log('start_sampling')
        syncflag = OverlappedProcess.verify(syncflag)
        ret = libgpg3300.DaStartSampling(self.ndev, syncflag)
        self._error_check(ret)
        return

    def stop_sampling(self):
        self._log('stop_sampling')
        ret = libgpg3300.DaStopSampling(self.ndev)
        self._error_check(ret)
        return

    def start_file_sampling(self, path, format, num):
        self._log('start_file_sampling')
        FileFormat.verify(format)
        ret = libgpg3300.DaStartFileSampling(self.ndev, path, format, num)
        self._error_check(ret)
        return

    def sync_sampling(self, mode):
        self._log('sync_sampling')
        #SynchronousNumber.verify(mode & 0x1100)
        mode = SynchronousAnalogOutput.verify(mode)
        ret = libgpg3300.DaSyncSampling(self.ndev, mode)
        self._error_check(ret)
        return

    def get_status(self):
        self._log('get_status')
        status = ctypes.c_ulong(0)
        done_count = ctypes.c_ulong(0)
        rest_count = ctypes.c_ulong(0)
        rest_repeat = ctypes.c_ulong(0)
        ret = libgpg3300.DaGetStatus(self.ndev, status, done_count, rest_count, rest_repeat)
        self._error_check(ret)
        status = status.value
        done_count = done_count.value
        rest_count = rest_count.value
        rest_repeat = rest_repeat.value
        print('SmplStatus = %d (%s)'%(status, AnalogOutputStatus.get_id(status)))
        print('SmplCount = %d, AvailCount = %d, AvailRepeat = %d'%(done_count, rest_count, rest_repeat))
        return status, done_count, rest_count, rest_repeat
    
    def output_da(self, data, chs=None, ranges=None):
        self._log('output_da')
        ch_config = self._create_sampling_ch_config(chs, ranges)
        if len(ch_config)!=len(data):
            raise ValueError, 'Size dosen\'t match. len(data)=%d, len(chs)=%d'%(len(data), len(ch_config))
        d = (self._data_size * len(data))()
        for i in range(len(data)):
            print(i, data[i])
            d[i] = data[i]
        ret = libgpg3300.DaOutputDA(self.ndev, len(data), ch_config, d)
        self._error_check(ret)
        return
        
    def set_output_da_ex(self, chs, ranges):
        self._log('set_output_da_ex')
        ch_config = self._create_sampling_ch_config(chs, ranges)
        ret = libgpg3300.DaSetOutputDAEx(self.ndev, len(ch_config), ch_config)
        self._error_check(ret)
        return
        
    def output_da_ex(self, data):
        self._log('output_da_ex')
        d = (self._data_size * len(data))()
        for i in range(len(data)): d[i] = data[i]
        ret = libgpg3300.DaOutputDAEx(self.ndev, d)
        self._error_check(ret)
        return

    def input_di(self):
        self._log('input_di')
        di = ctypes.c_ulong(0)
        ret = libgpg3300.DaInputDI(self.ndev, di)
        self._error_check(ret)
        di = di.value
        print('Data = %s'%(bin(di)))
        return di
    
    def output_do(self, do):
        self._log('output_do')
        ret = libgpg3300.DaOutputDO(self.ndev, do)
        self._error_check(ret)
        return

    # for I/O devices
    # ---------------
    def set_output_mode(self, mode):
        self._log('set_output_mode')
        SimultaneousOutputSet.verify(mode)
        ret = libgpg3300.DaSetOutputMode(self.ndev, mode)
        self._error_check(ret)
        # The discription of official document (GPG-3300 Help for Linux, p.109) is incorrect.
        return
    
    def get_output_mode(self):
        self._log('get_output_mode')
        mode = ctypes.c_ulong(0)
        ret = libgpg3300.DaGetOutputMode(self.ndev, mode)
        self._error_check(ret)
        mode = mode.value
        print('Mode = %d (%s)'%(mode, SimultaneousOutputSet.verify(mode)))
        # The discription of official document (GPG-3300 Help for Linux, p.109) is incorrect.
        return
    

