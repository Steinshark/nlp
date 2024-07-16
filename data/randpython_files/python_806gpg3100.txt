
import ctypes
import numpy

import pyinterface
import libgpg3100


# Identifer Wrapper
# =================

class SamplingMode(pyinterface.Identifer):
    FLAG_SYNC = pyinterface.IdentiferElement('FLAG_SYNC', libgpg3100.FLAG_SYNC)
    FLAG_ASYNC = pyinterface.IdentiferElement('FLAG_ASYNC', libgpg3100.FLAG_ASYNC)
    FLAG_TRIGGER = pyinterface.IdentiferElement('FLAG_TRIGGER', libgpg3100.FLAG_TRIGGER)
    FLAG_MASTER_MODE = pyinterface.IdentiferElement('FLAG_MASTER_MODE', libgpg3100.FLAG_MASTER_MODE)
    FLAG_SLAVE_MODE = pyinterface.IdentiferElement('FLAG_SLAVE_MODE', libgpg3100.FLAG_SLAVE_MODE)
    pass

class FileType(pyinterface.Identifer):
    FLAG_BIN = pyinterface.IdentiferElement('FLAG_BIN', libgpg3100.FLAG_BIN)
    FLAG_CSV = pyinterface.IdentiferElement('FLAG_CSV', libgpg3100.FLAG_CSV)
    pass

class DataFormatType(pyinterface.Identifer):
    CONV_BIN = pyinterface.IdentiferElement('CONV_BIN', libgpg3100.CONV_BIN)
    CONV_PHYS = pyinterface.IdentiferElement('CONV_PHYS', libgpg3100.CONV_PHYS)
    pass

class SamplingStatus(pyinterface.Identifer):
    AD_STATUS_STOP_SAMPLING = pyinterface.IdentiferElement('AD_STATUS_STOP_SAMPLING', libgpg3100.AD_STATUS_STOP_SAMPLING)
    AD_STATUS_WAIT_TRIGGER = pyinterface.IdentiferElement('AD_STATUS_WAIT_TRIGGER', libgpg3100.AD_STATUS_WAIT_TRIGGER)
    AD_STATUS_NOW_SAMPLING = pyinterface.IdentiferElement('AD_STATUS_NOW_SAMPLING', libgpg3100.AD_STATUS_NOW_SAMPLING)
    pass

class SamplingEvent(pyinterface.Identifer):
    AD_EVENT_NONE = pyinterface.IdentiferElement('AD_EVENT_NONE', 0)
    AD_EVENT_SMPLNUM = pyinterface.IdentiferElement('AD_EVENT_SMPLNUM', libgpg3100.AD_EVENT_SMPLNUM)
    AD_EVENT_STOP_TRIGGER = pyinterface.IdentiferElement('AD_EVENT_STOP_TRIGGER', libgpg3100.AD_EVENT_STOP_TRIGGER)
    AD_EVENT_STOP_FUNCTION = pyinterface.IdentiferElement('AD_EVENT_STOP_FUNCTION', libgpg3100.AD_EVENT_STOP_FUNCTION)
    AD_EVENT_STOP_TIMEOUT = pyinterface.IdentiferElement('AD_EVENT_STOP_TIMEOUT', libgpg3100.AD_EVENT_STOP_TIMEOUT)
    AD_EVENT_STOP_SAMPLING = pyinterface.IdentiferElement('AD_EVENT_STOP_SAMPLING', libgpg3100.AD_EVENT_STOP_SAMPLING)
    AD_EVENT_STOP_SCER = pyinterface.IdentiferElement('AD_EVENT_STOP_SCER', libgpg3100.AD_EVENT_STOP_SCER)
    AD_EVENT_STOP_ORER = pyinterface.IdentiferElement('AD_EVENT_STOP_ORER', libgpg3100.AD_EVENT_STOP_ORER)
    AD_EVENT_SCER = pyinterface.IdentiferElement('AD_EVENT_SCER', libgpg3100.AD_EVENT_SCER)
    AD_EVENT_ORER = pyinterface.IdentiferElement('AD_EVENT_ORER', libgpg3100.AD_EVENT_ORER)
    AD_EVENT_STOP_LV_1 = pyinterface.IdentiferElement('AD_EVENT_STOP_LV_1', libgpg3100.AD_EVENT_STOP_LV_1)
    AD_EVENT_STOP_LV_2 = pyinterface.IdentiferElement('AD_EVENT_STOP_LV_2', libgpg3100.AD_EVENT_STOP_LV_2)
    AD_EVENT_STOP_LV_3 = pyinterface.IdentiferElement('AD_EVENT_STOP_LV_3', libgpg3100.AD_EVENT_STOP_LV_3)
    AD_EVENT_STOP_LV_4 = pyinterface.IdentiferElement('AD_EVENT_STOP_LV_4', libgpg3100.AD_EVENT_STOP_LV_4)
    AD_EVENT_RANGE = pyinterface.IdentiferElement('AD_EVENT_RANGE', libgpg3100.AD_EVENT_RANGE)
    AD_EVENT_STOP_RANGE = pyinterface.IdentiferElement('AD_EVENT_STOP_RANGE', libgpg3100.AD_EVENT_STOP_RANGE)
    AD_EVENT_OVPM = pyinterface.IdentiferElement('AD_EVENT_OVPM', libgpg3100.AD_EVENT_OVPM)
    AD_EVENT_STOP_OVPM = pyinterface.IdentiferElement('AD_EVENT_STOP_OVPM', libgpg3100.AD_EVENT_STOP_OVPM)
    pass

class SamplingMethod(pyinterface.Identifer):
    AD_INPUT_SINGLE = pyinterface.IdentiferElement('AD_INPUT_SINGLE', libgpg3100.AD_INPUT_SINGLE)
    AD_INPUT_DIFF = pyinterface.IdentiferElement('AD_INPUT_DIFF', libgpg3100.AD_INPUT_DIFF)
    AD_INPUT_GND = pyinterface.IdentiferElement('AD_INPUT_GND', libgpg3100.AD_INPUT_GND)
    AD_INPUT_REFP5V = pyinterface.IdentiferElement('AD_INPUT_REFP5V', libgpg3100.AD_INPUT_REFP5V)
    AD_INPUT_REFM5V = pyinterface.IdentiferElement('AD_INPUT_REFM5V', libgpg3100.AD_INPUT_REFM5V)
    AD_INPUT_DACONV = pyinterface.IdentiferElement('AD_INPUT_DACONV', libgpg3100.AD_INPUT_DACONV)
    AD_INPUT_REFP3V = pyinterface.IdentiferElement('AD_INPUT_REFP3V', libgpg3100.AD_INPUT_REFP3V)
    AD_INPUT_REFM3V = pyinterface.IdentiferElement('AD_INPUT_REFM3V', libgpg3100.AD_INPUT_REFM3V)
    AD_INPUT_DAC1 = pyinterface.IdentiferElement('AD_INPUT_DAC1', libgpg3100.AD_INPUT_DAC1)
    AD_INPUT_DAC2 = pyinterface.IdentiferElement('AD_INPUT_DAC2', libgpg3100.AD_INPUT_DAC2)
    AD_INPUT_DAC3 = pyinterface.IdentiferElement('AD_INPUT_DAC3', libgpg3100.AD_INPUT_DAC3)
    AD_INPUT_DAC4 = pyinterface.IdentiferElement('AD_INPUT_DAC4', libgpg3100.AD_INPUT_DAC4)
    AD_INPUT_DAC5 = pyinterface.IdentiferElement('AD_INPUT_DAC5', libgpg3100.AD_INPUT_DAC5)
    AD_INPUT_DAC6 = pyinterface.IdentiferElement('AD_INPUT_DAC6', libgpg3100.AD_INPUT_DAC6)
    pass

class VolumeAdjustType(pyinterface.Identifer):
    AD_ADJUST_BIOFFSET = pyinterface.IdentiferElement('AD_ADJUST_BIOFFSET', libgpg3100.AD_ADJUST_BIOFFSET)
    AD_ADJUST_UNIOFFSET = pyinterface.IdentiferElement('AD_ADJUST_UNIOFFSET', libgpg3100.AD_ADJUST_UNIOFFSET)
    AD_ADJUST_BIGAIN = pyinterface.IdentiferElement('AD_ADJUST_BIGAIN', libgpg3100.AD_ADJUST_BIGAIN)
    AD_ADJUST_UNIGAIN = pyinterface.IdentiferElement('AD_ADJUST_UNIGAIN', libgpg3100.AD_ADJUST_UNIGAIN)
    pass
    
class VolumeAdjustMode(pyinterface.Identifer):
    AD_ADJUST_UP = pyinterface.IdentiferElement('AD_ADJUST_UP', libgpg3100.AD_ADJUST_UP)
    AD_ADJUST_DOWN = pyinterface.IdentiferElement('AD_ADJUST_DOWN', libgpg3100.AD_ADJUST_DOWN)
    AD_ADJUST_STORE = pyinterface.IdentiferElement('AD_ADJUST_STORE', libgpg3100.AD_ADJUST_STORE)
    AD_ADJUST_STANDBY = pyinterface.IdentiferElement('AD_ADJUST_STANDBY', libgpg3100.AD_ADJUST_STANDBY)
    AD_ADJUST_NOT_STORE = pyinterface.IdentiferElement('AD_ADJUST_NOT_STORE', libgpg3100.AD_ADJUST_NOT_STORE)
    AD_ADJUST_STORE_INITAREA = pyinterface.IdentiferElement('AD_ADJUST_STORE_INITAREA', libgpg3100.AD_ADJUST_STORE_INITAREA)
    pass
    
class VolumeAdjustUser(pyinterface.Identifer):
    AD_ADJUST_READ_FACTORY = pyinterface.IdentiferElement('AD_ADJUST_READ_FACTORY', libgpg3100.AD_ADJUST_READ_FACTORY)
    AD_ADJUST_READ_USER = pyinterface.IdentiferElement('AD_ADJUST_READ_USER', libgpg3100.AD_ADJUST_READ_USER)
    pass

class DataConversion(pyinterface.Identifer):
    AD_DATA_PHYSICAL = pyinterface.IdentiferElement('AD_DATA_PHYSICAL', libgpg3100.AD_DATA_PHYSICAL)
    AD_DATA_BIN8 = pyinterface.IdentiferElement('AD_DATA_BIN8', libgpg3100.AD_DATA_BIN8)
    AD_DATA_BIN12 = pyinterface.IdentiferElement('AD_DATA_BIN12', libgpg3100.AD_DATA_BIN12)
    AD_DATA_BIN16 = pyinterface.IdentiferElement('AD_DATA_BIN16', libgpg3100.AD_DATA_BIN16)
    AD_DATA_BIN24 = pyinterface.IdentiferElement('AD_DATA_BIN24', libgpg3100.AD_DATA_BIN24)
    AD_DATA_BIN10 = pyinterface.IdentiferElement('AD_DATA_BIN10', libgpg3100.AD_DATA_BIN10)
    pass

class DataConvolution(pyinterface.Identifer):
    AD_CONV_SMOOTH = pyinterface.IdentiferElement('AD_CONV_SMOOTH', libgpg3100.AD_CONV_SMOOTH)
    AD_CONV_AVERAGE1 = pyinterface.IdentiferElement('AD_CONV_AVERAGE1', libgpg3100.AD_CONV_AVERAGE1)
    AD_CONV_AVERAGE2 = pyinterface.IdentiferElement('AD_CONV_AVERAGE2', libgpg3100.AD_CONV_AVERAGE2)
    pass

class SamplingFormula(pyinterface.Identifer):
    AD_IO_SAMPLING = pyinterface.IdentiferElement('AD_IO_SAMPLING', libgpg3100.AD_IO_SAMPLING)
    AD_FIFO_SAMPLING = pyinterface.IdentiferElement('AD_FIFO_SAMPLING', libgpg3100.AD_FIFO_SAMPLING)
    AD_MEM_SAMPLING = pyinterface.IdentiferElement('AD_MEM_SAMPLING', libgpg3100.AD_MEM_SAMPLING)
    AD_BM_SAMPLING = pyinterface.IdentiferElement('AD_BM_SAMPLING', libgpg3100.AD_BM_SAMPLING)
    pass

class TriggerPoint(pyinterface.Identifer):
    AD_TRIG_START = pyinterface.IdentiferElement('AD_TRIG_START', libgpg3100.AD_TRIG_START)
    AD_TRIG_STOP = pyinterface.IdentiferElement('AD_TRIG_STOP', libgpg3100.AD_TRIG_STOP)
    AD_TRIG_START_STOP = pyinterface.IdentiferElement('AD_TRIG_START_STOP', libgpg3100.AD_TRIG_START_STOP)
    pass

class TriggerMode(pyinterface.Identifer):
    AD_FREERUN = pyinterface.IdentiferElement('AD_FREERUN', libgpg3100.AD_FREERUN)
    AD_EXTTRG = pyinterface.IdentiferElement('AD_EXTTRG', libgpg3100.AD_EXTTRG)
    AD_EXTTRG_DI = pyinterface.IdentiferElement('AD_EXTTRG_DI', libgpg3100.AD_EXTTRG_DI)
    AD_LEVEL_P = pyinterface.IdentiferElement('AD_LEVEL_P', libgpg3100.AD_LEVEL_P)
    AD_LEVEL_M = pyinterface.IdentiferElement('AD_LEVEL_M', libgpg3100.AD_LEVEL_M)
    AD_LEVEL_D = pyinterface.IdentiferElement('AD_LEVEL_D', libgpg3100.AD_LEVEL_D)
    AD_INRANGE = pyinterface.IdentiferElement('AD_INRANGE', libgpg3100.AD_INRANGE)
    AD_OUTRANGE = pyinterface.IdentiferElement('AD_OUTRANGE', libgpg3100.AD_OUTRANGE)
    AD_ETERNITY = pyinterface.IdentiferElement('AD_ETERNITY', libgpg3100.AD_ETERNITY)
    AD_SMPLNUM = pyinterface.IdentiferElement('AD_SMPLNUM', libgpg3100.AD_SMPLNUM)
    
    AD_START_SIGTIMER = pyinterface.IdentiferElement('AD_START_SIGTIMER', libgpg3100.AD_START_SIGTIMER)
    AD_START_DA_START = pyinterface.IdentiferElement('AD_START_DA_START', libgpg3100.AD_START_DA_START)
    AD_START_DA_STOP = pyinterface.IdentiferElement('AD_START_DA_STOP', libgpg3100.AD_START_DA_STOP)
    AD_START_DA_IO = pyinterface.IdentiferElement('AD_START_DA_IO', libgpg3100.AD_START_DA_IO)
    AD_START_DA_SMPLNUM = pyinterface.IdentiferElement('AD_START_DA_SMPLNUM', libgpg3100.AD_START_DA_SMPLNUM)
    AD_STOP_SIGTIMER = pyinterface.IdentiferElement('AD_STOP_SIGTIMER', libgpg3100.AD_STOP_SIGTIMER)
    AD_STOP_DA_START = pyinterface.IdentiferElement('AD_STOP_DA_START', libgpg3100.AD_STOP_DA_START)
    AD_STOP_DA_STOP = pyinterface.IdentiferElement('AD_STOP_DA_STOP', libgpg3100.AD_STOP_DA_STOP)
    AD_STOP_DA_IO = pyinterface.IdentiferElement('AD_STOP_DA_IO', libgpg3100.AD_STOP_DA_IO)
    AD_STOP_DA_SMPLNUM = pyinterface.IdentiferElement('AD_STOP_DA_SMPLNUM', libgpg3100.AD_STOP_DA_SMPLNUM)
    
    AD_START_P1 = pyinterface.IdentiferElement('AD_START_P1', libgpg3100.AD_START_P1)
    AD_START_M1 = pyinterface.IdentiferElement('AD_START_M1', libgpg3100.AD_START_M1)
    AD_START_D1 = pyinterface.IdentiferElement('AD_START_D1', libgpg3100.AD_START_D1)
    AD_START_P2 = pyinterface.IdentiferElement('AD_START_P2', libgpg3100.AD_START_P2)
    AD_START_M2 = pyinterface.IdentiferElement('AD_START_M2', libgpg3100.AD_START_M2)
    AD_START_D2 = pyinterface.IdentiferElement('AD_START_D2', libgpg3100.AD_START_D2)
    AD_STOP_P1 = pyinterface.IdentiferElement('AD_STOP_P1', libgpg3100.AD_STOP_P1)
    AD_STOP_M1 = pyinterface.IdentiferElement('AD_STOP_M1', libgpg3100.AD_STOP_M1)
    AD_STOP_D1 = pyinterface.IdentiferElement('AD_STOP_D1', libgpg3100.AD_STOP_D1)
    AD_STOP_P2 = pyinterface.IdentiferElement('AD_STOP_P2', libgpg3100.AD_STOP_P2)
    AD_STOP_M2 = pyinterface.IdentiferElement('AD_STOP_M2', libgpg3100.AD_STOP_M2)
    AD_STOP_D2 = pyinterface.IdentiferElement('AD_STOP_D2', libgpg3100.AD_STOP_D2)
    AD_ANALOG_FILTER = pyinterface.IdentiferElement('AD_ANALOG_FILTER', libgpg3100.AD_ANALOG_FILTER)
    AD_START_CNT_EQ = pyinterface.IdentiferElement('AD_START_CNT_EQ', libgpg3100.AD_START_CNT_EQ)
    AD_STOP_CNT_EQ = pyinterface.IdentiferElement('AD_STOP_CNT_EQ', libgpg3100.AD_STOP_CNT_EQ)
    AD_START_DI_EQ = pyinterface.IdentiferElement('AD_START_DI_EQ', libgpg3100.AD_START_DI_EQ)
    AD_STOP_DI_EQ = pyinterface.IdentiferElement('AD_STOP_DI_EQ', libgpg3100.AD_STOP_DI_EQ)
    AD_STOP_SOFT = pyinterface.IdentiferElement('AD_STOP_SOFT', libgpg3100.AD_STOP_SOFT)
    AD_START_Z_CLR = pyinterface.IdentiferElement('AD_START_Z_CLR', libgpg3100.AD_START_Z_CLR)
    AD_STOP_Z_CLR = pyinterface.IdentiferElement('AD_STOP_Z_CLR', libgpg3100.AD_STOP_Z_CLR)
    AD_START_SYNC = pyinterface.IdentiferElement('AD_START_SYNC', libgpg3100.AD_START_SYNC)
    AD_STOP_SYNC = pyinterface.IdentiferElement('AD_STOP_SYNC', libgpg3100.AD_STOP_SYNC)
    AD_START_SYNC1 = pyinterface.IdentiferElement('AD_START_SYNC1', libgpg3100.AD_START_SYNC1)
    AD_START_SYNC2 = pyinterface.IdentiferElement('AD_START_SYNC2', libgpg3100.AD_START_SYNC2)
    AD_STOP_SYNC1 = pyinterface.IdentiferElement('AD_STOP_SYNC1', libgpg3100.AD_STOP_SYNC1)
    AD_STOP_SYNC2 = pyinterface.IdentiferElement('AD_STOP_SYNC2', libgpg3100.AD_STOP_SYNC2)
    pass

class TriggerEdge(pyinterface.Identifer):
    AD_DOWN_EDGE = pyinterface.IdentiferElement('AD_DOWN_EDGE', libgpg3100.AD_DOWN_EDGE)
    AD_UP_EDGE = pyinterface.IdentiferElement('AD_UP_EDGE', libgpg3100.AD_UP_EDGE)
    AD_EXTRG_IN = pyinterface.IdentiferElement('AD_EXTRG_IN', libgpg3100.AD_EXTRG_IN)
    AD_EXCLK_IN = pyinterface.IdentiferElement('AD_EXCLK_IN', libgpg3100.AD_EXCLK_IN)
    AD_LOW_LEVEL = pyinterface.IdentiferElement('AD_LOW_LEVEL', libgpg3100.AD_LOW_LEVEL)
    AD_HIGH_LEVEL = pyinterface.IdentiferElement('AD_HIGH_LEVEL', libgpg3100.AD_HIGH_LEVEL)
    
    AD_EDGE_P1 = pyinterface.IdentiferElement('AD_EDGE_P1', libgpg3100.AD_EDGE_P1)
    AD_EDGE_M1 = pyinterface.IdentiferElement('AD_EDGE_M1', libgpg3100.AD_EDGE_M1)
    AD_EDGE_D1 = pyinterface.IdentiferElement('AD_EDGE_D1', libgpg3100.AD_EDGE_D1)
    AD_EDGE_P2 = pyinterface.IdentiferElement('AD_EDGE_P2', libgpg3100.AD_EDGE_P2)
    AD_EDGE_M2 = pyinterface.IdentiferElement('AD_EDGE_M2', libgpg3100.AD_EDGE_M2)
    AD_EDGE_D2 = pyinterface.IdentiferElement('AD_EDGE_D2', libgpg3100.AD_EDGE_D2)
    AD_DISABLE = pyinterface.IdentiferElement('AD_DISABLE', libgpg3100.AD_DISABLE)
    pass

class TriggerState(pyinterface.Identifer):    
    AD_TRIG_MODE = pyinterface.IdentiferElement('AD_TRIG_MODE', libgpg3100.AD_TRIG_MODE)
    AD_BUSY_MODE = pyinterface.IdentiferElement('AD_BUSY_MODE', libgpg3100.AD_BUSY_MODE)
    AD_POST_MODE = pyinterface.IdentiferElement('AD_POST_MODE', libgpg3100.AD_POST_MODE)
    AD_ENABLE = pyinterface.IdentiferElement('AD_ENABLE', libgpg3100.AD_ENABLE)
    AD_SMP1_MODE = pyinterface.IdentiferElement('AD_SMP1_MODE', libgpg3100.AD_SMP1_MODE)
    AD_SMP2_MODE = pyinterface.IdentiferElement('AD_SMP2_MODE', libgpg3100.AD_SMP2_MODE)
    AD_ATRIG_MODE = pyinterface.IdentiferElement('AD_ATRIG_MODE', libgpg3100.AD_ATRIG_MODE)
    pass

class AnalogTriggerEdge(pyinterface.Identifer):
    AD_LOW_PULSE = pyinterface.IdentiferElement('AD_LOW_PULSE', libgpg3100.AD_LOW_PULSE)
    AD_HIGH_PULSE = pyinterface.IdentiferElement('AD_HIGH_PULSE', libgpg3100.AD_HIGH_PULSE)
    pass

class FastMode(pyinterface.Identifer):
    AD_NORMAL_MODE = pyinterface.IdentiferElement('AD_NORMAL_MODE', libgpg3100.AD_NORMAL_MODE)
    AD_FAST_MODE = pyinterface.IdentiferElement('AD_FAST_MODE', libgpg3100.AD_FAST_MODE)
    pass

class AdStatus(pyinterface.Identifer):
    AD_NO_STATUS = pyinterface.IdentiferElement('AD_NO_STATUS', libgpg3100.AD_NO_STATUS)
    AD_ADD_STATUS = pyinterface.IdentiferElement('AD_ADD_STATUS', libgpg3100.AD_ADD_STATUS)
    pass

class ErrorBusmasterClock(pyinterface.Identifer):
    AD_STOP_SCER = pyinterface.IdentiferElement('AD_STOP_SCER', libgpg3100.AD_STOP_SCER)
    AD_STOP_ORER = pyinterface.IdentiferElement('AD_STOP_ORER', libgpg3100.AD_STOP_ORER)
    pass

class BusmasterData(pyinterface.Identifer):
    AD_APPEND = pyinterface.IdentiferElement('AD_APPEND', libgpg3100.AD_APPEND)
    AD_OVERWRITE = pyinterface.IdentiferElement('AD_OVERWRITE', libgpg3100.AD_OVERWRITE)
    pass

class DigitalFilter(pyinterface.Identifer):
    AD_DF_8 = pyinterface.IdentiferElement('AD_DF_8', libgpg3100.AD_DF_8)
    AD_DF_16 = pyinterface.IdentiferElement('AD_DF_16', libgpg3100.AD_DF_16)
    AD_DF_32 = pyinterface.IdentiferElement('AD_DF_32', libgpg3100.AD_DF_32)
    AD_DF_64 = pyinterface.IdentiferElement('AD_DF_64', libgpg3100.AD_DF_64)
    AD_DF_128 = pyinterface.IdentiferElement('AD_DF_128', libgpg3100.AD_DF_128)
    AD_DF_256 = pyinterface.IdentiferElement('AD_DF_256', libgpg3100.AD_DF_256)
    pass


class RangeElement(pyinterface.IdentiferElement):
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
        bit = numpy.linspace(self.min, self.max, resolution)
        bins = bit[:-1] + (bit[1] - bit[0])/2.
        ret = numpy.digitize(value, bins)
        if flag_not_array: ret = ret[0]
        else: ret = list(ret)
        return ret
    
    def analogize(self, value, resolution):
        flag_not_array = False
        if type(value) in [int, float]: 
            flag_not_array = True
            value = [value]
            pass
        analog = numpy.linspace(self.min, self.max, resolution)
        ret = analog[value]
        if flag_not_array: ret = ret[0]
        else: ret = list(ret)
        return ret

class SamplingRange(pyinterface.Identifer):
    AD_0_1V = RangeElement('AD_0_1V', libgpg3100.AD_0_1V, 0.0, 1.0)
    AD_0_2P5V = RangeElement('AD_0_2P5V', libgpg3100.AD_0_2P5V, 0.0, 2.5)
    AD_0_5V = RangeElement('AD_0_5V', libgpg3100.AD_0_5V, 0.0, 5.0)
    AD_0_10V = RangeElement('AD_0_10V', libgpg3100.AD_0_10V, 0.0, 10.0)
    AD_1_5V = RangeElement('AD_1_5V', libgpg3100.AD_1_5V, 1.0, 5.0)
    AD_0_2V = RangeElement('AD_0_2V', libgpg3100.AD_0_2V, 0.0, 2.0)
    AD_0_0P125V = RangeElement('AD_0_0P125V', libgpg3100.AD_0_0P125V, 0.0, 0.125)
    AD_0_1P25V = RangeElement('AD_0_1P25V', libgpg3100.AD_0_1P25V, 0.0, 1.25)
    AD_0_0P625V = RangeElement('AD_0_0P625V', libgpg3100.AD_0_0P625V, 0.0, 0.625)
    AD_0_0P156V = RangeElement('AD_0_0P156V', libgpg3100.AD_0_0P156V, 0.0, 0.156)
    AD_0_20mA = RangeElement('AD_0_20mA', libgpg3100.AD_0_20mA, 0.000, 0.020)
    AD_4_20mA = RangeElement('AD_4_20mA', libgpg3100.AD_4_20mA, 0.004, 0.020)
    AD_20mA = RangeElement('AD_20mA', libgpg3100.AD_20mA, -0.020, 0.020)
    AD_1V = RangeElement('AD_1V', libgpg3100.AD_1V, -1.0, 1.0)
    AD_2P5V = RangeElement('AD_2P5V', libgpg3100.AD_2P5V, -2.5, 2.5)
    AD_5V = RangeElement('AD_5V', libgpg3100.AD_5V, -5.0, 5.0)
    AD_10V = RangeElement('AD_10V', libgpg3100.AD_10V, -10., 10.)
    AD_20V = RangeElement('AD_20V', libgpg3100.AD_20V, -20., 20.)
    AD_50V = RangeElement('AD_50V', libgpg3100.AD_50V, -50., 50.)
    AD_0P125V = RangeElement('AD_0P125V', libgpg3100.AD_0P125V, -0.125, 0.125)
    AD_1P25V = RangeElement('AD_1P25V', libgpg3100.AD_1P25V, -1.25, 1.25)
    AD_0P625V = RangeElement('AD_0P625V', libgpg3100.AD_0P625V, -0.625, 0.625)
    AD_0P156V = RangeElement('AD_0P156V', libgpg3100.AD_0P156V, -0.156, 0.156)
    AD_1P25V_AC = RangeElement('AD_1P25V_AC', libgpg3100.AD_1P25V_AC, -1.25, 1.25)
    AD_0P625V_AC = RangeElement('AD_0P625V_AC', libgpg3100.AD_0P625V_AC, -0.625, 0.625)
    AD_0P156V_AC = RangeElement('AD_0P156V_AC', libgpg3100.AD_0P156V_AC, -0.156, 0.156)
    AD_AC_COUPLING = RangeElement('AD_AC_COUPLING', libgpg3100.AD_AC_COUPLING, None, None)
    AD_GND = RangeElement('AD_GND', libgpg3100.AD_GND, None, None)
    
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

class Isolation(pyinterface.Identifer):
    AD_ISOLATION = pyinterface.IdentiferElement('AD_ISOLATION', libgpg3100.AD_ISOLATION)
    AD_NOT_ISOLATION = pyinterface.IdentiferElement('AD_NOT_ISOLATION', libgpg3100.AD_NOT_ISOLATION)
    pass

class SyncSampling(pyinterface.Identifer):
    AD_MASTER_MODE = pyinterface.IdentiferElement('AD_MASTER_MODE', libgpg3100.AD_MASTER_MODE)
    AD_SLAVE_MODE = pyinterface.IdentiferElement('AD_SLAVE_MODE', libgpg3100.AD_SLAVE_MODE)
    
    # for CTP- devices
    AD_SYNC_NUM_1 = pyinterface.IdentiferElement('AD_SYNC_NUM_1', libgpg3100.AD_SYNC_NUM_1)
    AD_SYNC_NUM_2 = pyinterface.IdentiferElement('AD_SYNC_NUM_2', libgpg3100.AD_SYNC_NUM_2)
    AD_SYNC_NUM_3 = pyinterface.IdentiferElement('AD_SYNC_NUM_3', libgpg3100.AD_SYNC_NUM_3)
    AD_SYNC_NUM_4 = pyinterface.IdentiferElement('AD_SYNC_NUM_4', libgpg3100.AD_SYNC_NUM_4)
    AD_SYNC_NUM_5 = pyinterface.IdentiferElement('AD_SYNC_NUM_5', libgpg3100.AD_SYNC_NUM_5)
    AD_SYNC_NUM_6 = pyinterface.IdentiferElement('AD_SYNC_NUM_6', libgpg3100.AD_SYNC_NUM_6)
    AD_SYNC_NUM_7 = pyinterface.IdentiferElement('AD_SYNC_NUM_7', libgpg3100.AD_SYNC_NUM_7)
    pass

class CalibrationMode(pyinterface.Identifer):
    AD_SELF_CALIBRATION = pyinterface.IdentiferElement('AD_SELF_CALIBRATION', libgpg3100.AD_SELF_CALIBRATION)
    AD_ZEROSCALE_CALIBRATION = pyinterface.IdentiferElement('AD_ZEROSCALE_CALIBRATION', libgpg3100.AD_ZEROSCALE_CALIBRATION)
    AD_FULLSCALE_CALIBRATION = pyinterface.IdentiferElement('AD_FULLSCALE_CALIBRATION', libgpg3100.AD_FULLSCALE_CALIBRATION)
    pass

class RangeStatus(pyinterface.Identifer):
    AD_STATUS_UNDER_RANGE = pyinterface.IdentiferElement('AD_STATUS_UNDER_RANGE', libgpg3100.AD_STATUS_UNDER_RANGE)
    AD_STATUS_OVER_RANGE = pyinterface.IdentiferElement('AD_STATUS_OVER_RANGE', libgpg3100.AD_STATUS_OVER_RANGE)
    pass

class CompareDataStatus(pyinterface.Identifer):
    AD_STATUS_UP_DATA = pyinterface.IdentiferElement('AD_STATUS_UP_DATA', libgpg3100.AD_STATUS_UP_DATA)
    AD_STATUS_DOWN_DATA = pyinterface.IdentiferElement('AD_STATUS_DOWN_DATA', libgpg3100.AD_STATUS_DOWN_DATA)
    pass

class Pci3525Func(pyinterface.Identifer):
    AD_CN_FREE = pyinterface.IdentiferElement('AD_CN_FREE', libgpg3100.AD_CN_FREE)
    AD_CN_EXTRG_IN = pyinterface.IdentiferElement('AD_CN_EXTRG_IN', libgpg3100.AD_CN_EXTRG_IN)
    AD_CN_EXTRG_OUT = pyinterface.IdentiferElement('AD_CN_EXTRG_OUT', libgpg3100.AD_CN_EXTRG_OUT)
    AD_CN_EXCLK_IN = pyinterface.IdentiferElement('AD_CN_EXCLK_IN', libgpg3100.AD_CN_EXCLK_IN)
    AD_CN_EXCLK_OUT = pyinterface.IdentiferElement('AD_CN_EXCLK_OUT', libgpg3100.AD_CN_EXCLK_OUT)
    AD_CN_EXINT_IN = pyinterface.IdentiferElement('AD_CN_EXINT_IN', libgpg3100.AD_CN_EXINT_IN)
    AD_CN_ATRG_OUT = pyinterface.IdentiferElement('AD_CN_ATRG_OUT', libgpg3100.AD_CN_ATRG_OUT)
    AD_CN_DI = pyinterface.IdentiferElement('AD_CN_DI', libgpg3100.AD_CN_DI)
    AD_CN_DO = pyinterface.IdentiferElement('AD_CN_DO', libgpg3100.AD_CN_DO)
    AD_CN_EXSMP1_OUT = pyinterface.IdentiferElement('AD_CN_EXSMP1_OUT', libgpg3100.AD_CN_EXSMP1_OUT)
    AD_CN_EXSMP2_OUT = pyinterface.IdentiferElement('AD_CN_EXSMP2_OUT', libgpg3100.AD_CN_EXSMP2_OUT)
    AD_CN_DIO = pyinterface.IdentiferElement('AD_CN_DIO', libgpg3100.AD_CN_DIO)
    AD_CN_CONTROL = pyinterface.IdentiferElement('AD_CN_CONTROL', libgpg3100.AD_CN_CONTROL)
    AD_CN_CNT = pyinterface.IdentiferElement('AD_CN_CNT', libgpg3100.AD_CN_CNT)
    pass

class Cpz360810Func(pyinterface.Identifer):
    AD_EX_DIO1 = pyinterface.IdentiferElement('AD_EX_DIO1', libgpg3100.AD_EX_DIO1)
    AD_EX_DIO2 = pyinterface.IdentiferElement('AD_EX_DIO2', libgpg3100.AD_EX_DIO2)
    AD_EX_DIO3 = pyinterface.IdentiferElement('AD_EX_DIO3', libgpg3100.AD_EX_DIO3)
    AD_EX_DIO4 = pyinterface.IdentiferElement('AD_EX_DIO4', libgpg3100.AD_EX_DIO4)
    AD_EX_DIO5 = pyinterface.IdentiferElement('AD_EX_DIO5', libgpg3100.AD_EX_DIO5)
    AD_EX_DIO6 = pyinterface.IdentiferElement('AD_EX_DIO6', libgpg3100.AD_EX_DIO6)
    AD_EX_DIO7 = pyinterface.IdentiferElement('AD_EX_DIO7', libgpg3100.AD_EX_DIO7)
    AD_EX_DIO8 = pyinterface.IdentiferElement('AD_EX_DIO8', libgpg3100.AD_EX_DIO8)
    pass

class Temperature(pyinterface.Identifer):
    AD_GET_CURRENT_TEMPERATURE = pyinterface.IdentiferElement('AD_GET_CURRENT_TEMPERATURE', libgpg3100.AD_GET_CURRENT_TEMPERATURE)
    AD_LOAD_TEMPERATURE = pyinterface.IdentiferElement('AD_LOAD_TEMPERATURE', libgpg3100.AD_LOAD_TEMPERATURE)
    AD_LOAD_FACTORY_SETTING_TEMPERATURE = pyinterface.IdentiferElement('AD_LOAD_FACTORY_SETTING_TEMPERATURE', libgpg3100.AD_LOAD_FACTORY_SETTING_TEMPERATURE)
    AD_SAVE_TEMPERATURE_USER = pyinterface.IdentiferElement('AD_SAVE_TEMPERATURE_USER', libgpg3100.AD_SAVE_TEMPERATURE_USER)
    pass

class SamplingNum(object):
    @classmethod
    def verify(cls, value):
        value = int(value)
        if value < 1:
            errmsg = '%s.%s: value should be >1 (value=%d)'%(cls.__module__, cls.__name__, value)
            raise ValueError, errmsg
        return value

class SamplingEventNum(object):
    @classmethod
    def verify(cls, value):
        value = int(value)
        if value < 0:
            errmsg = '%s.%s: value should be >0 (value=%d)'%(cls.__module__, cls.__name__, value)
            raise ValueError, errmsg
        return value

class SamplingFreq(object):
    @classmethod
    def verify(cls, value):
        value = float(value)
        if value < 0:
            errmsg = '%s.%s: value should be 0.0 or >0.01 (value=%d)'%(cls.__module__, cls.__name__, value)
            raise ValueError, errmsg
        return value

class TriggerDelay(object):
    @classmethod
    def verify(cls, value):
        value = int(value)
        if (value < -1073741824)or((value > 1073741824)):
            errmsg = '%s.%s: value should be |t|<1073741824 (t=%d)'%(cls.__module__, cls.__name__, value)
            raise ValueError, errmsg
        return value


# Error Wrapper
# =============

class ErrorGPG3100(pyinterface.ErrorCode):
    AD_ERROR_SUCCESS = pyinterface.IdentiferElement('AD_ERROR_SUCCESS', libgpg3100.AD_ERROR_SUCCESS)
    AD_ERROR_NOT_DEVICE = pyinterface.IdentiferElement('AD_ERROR_NOT_DEVICE', libgpg3100.AD_ERROR_NOT_DEVICE)
    AD_ERROR_NOT_OPEN = pyinterface.IdentiferElement('AD_ERROR_NOT_OPEN', libgpg3100.AD_ERROR_NOT_OPEN)
    AD_ERROR_INVALID_HANDLE = pyinterface.IdentiferElement('AD_ERROR_INVALID_HANDLE', libgpg3100.AD_ERROR_INVALID_HANDLE)
    AD_ERROR_INVALID_DEVICENO = pyinterface.IdentiferElement('AD_ERROR_INVALID_DEVICENO', libgpg3100.AD_ERROR_INVALID_DEVICENO)
    AD_ERROR_ALREADY_OPEN = pyinterface.IdentiferElement('AD_ERROR_ALREADY_OPEN', libgpg3100.AD_ERROR_ALREADY_OPEN)
    AD_ERROR_INVALID_DEVICE_NUMBER = pyinterface.IdentiferElement('AD_ERROR_INVALID_DEVICE_NUMBER', libgpg3100.AD_ERROR_INVALID_DEVICE_NUMBER)
    AD_ERROR_NOT_SUPPORTED = pyinterface.IdentiferElement('AD_ERROR_NOT_SUPPORTED', libgpg3100.AD_ERROR_NOT_SUPPORTED)
    AD_ERROR_NOW_SAMPLING = pyinterface.IdentiferElement('AD_ERROR_NOW_SAMPLING', libgpg3100.AD_ERROR_NOW_SAMPLING)
    AD_ERROR_STOP_SAMPLING = pyinterface.IdentiferElement('AD_ERROR_STOP_SAMPLING', libgpg3100.AD_ERROR_STOP_SAMPLING)
    AD_ERROR_START_SAMPLING = pyinterface.IdentiferElement('AD_ERROR_START_SAMPLING', libgpg3100.AD_ERROR_START_SAMPLING)
    AD_ERROR_SAMPLING_TIMEOUT = pyinterface.IdentiferElement('AD_ERROR_SAMPLING_TIMEOUT', libgpg3100.AD_ERROR_SAMPLING_TIMEOUT)
    AD_ERROR_INVALID_PARAMETER = pyinterface.IdentiferElement('AD_ERROR_INVALID_PARAMETER', libgpg3100.AD_ERROR_INVALID_PARAMETER)
    AD_ERROR_NULL_POINTER = pyinterface.IdentiferElement('AD_ERROR_NULL_POINTER', libgpg3100.AD_ERROR_NULL_POINTER)
    AD_ERROR_GET_DATA = pyinterface.IdentiferElement('AD_ERROR_GET_DATA', libgpg3100.AD_ERROR_GET_DATA)
    AD_ERROR_USED_DA = pyinterface.IdentiferElement('AD_ERROR_USED_DA', libgpg3100.AD_ERROR_USED_DA)
    AD_ERROR_FILE_OPEN = pyinterface.IdentiferElement('AD_ERROR_FILE_OPEN', libgpg3100.AD_ERROR_FILE_OPEN)
    AD_ERROR_FILE_CLOSE = pyinterface.IdentiferElement('AD_ERROR_FILE_CLOSE', libgpg3100.AD_ERROR_FILE_CLOSE)
    AD_ERROR_FILE_READ = pyinterface.IdentiferElement('AD_ERROR_FILE_READ', libgpg3100.AD_ERROR_FILE_READ)
    AD_ERROR_FILE_WRITE = pyinterface.IdentiferElement('AD_ERROR_FILE_WRITE', libgpg3100.AD_ERROR_FILE_WRITE)
    AD_ERROR_INVALID_DATA_FORMAT = pyinterface.IdentiferElement('AD_ERROR_INVALID_DATA_FORMAT', libgpg3100.AD_ERROR_INVALID_DATA_FORMAT)
    AD_ERROR_INVALID_AVERAGE_OR_SMOOTHING = pyinterface.IdentiferElement('AD_ERROR_INVALID_AVERAGE_OR_SMOOTHING', libgpg3100.AD_ERROR_INVALID_AVERAGE_OR_SMOOTHING)
    AD_ERROR_INVALID_SOURCE_DATA = pyinterface.IdentiferElement('AD_ERROR_INVALID_SOURCE_DATA', libgpg3100.AD_ERROR_INVALID_SOURCE_DATA)
    AD_ERROR_NOT_ALLOCATE_MEMORY = pyinterface.IdentiferElement('AD_ERROR_NOT_ALLOCATE_MEMORY', libgpg3100.AD_ERROR_NOT_ALLOCATE_MEMORY)
    AD_ERROR_CALIBRATION = pyinterface.IdentiferElement('AD_ERROR_CALIBRATION', libgpg3100.AD_ERROR_CALIBRATION)
    AD_ERROR_NOT_THREAD = pyinterface.IdentiferElement('AD_ERROR_NOT_THREAD', libgpg3100.AD_ERROR_NOT_THREAD)
    AD_ERROR_NOT_DEVICE_NODE = pyinterface.IdentiferElement('AD_ERROR_NOT_DEVICE_NODE', libgpg3100.AD_ERROR_NOT_DEVICE_NODE)
    
    AD_ERROR_USBIO_FAILED = pyinterface.IdentiferElement('AD_ERROR_USBIO_FAILED', libgpg3100.AD_ERROR_USBIO_FAILED)
    AD_ERROR_USBIO_TIMEOUT = pyinterface.IdentiferElement('AD_ERROR_USBIO_TIMEOUT', libgpg3100.AD_ERROR_USBIO_TIMEOUT)
    AD_ERROR_USBLIB_LOAD_FAILED = pyinterface.IdentiferElement('AD_ERROR_USBLIB_LOAD_FAILED', libgpg3100.AD_ERROR_USBLIB_LOAD_FAILED)
    
    _success = AD_ERROR_SUCCESS
    pass


# ==========================
# GPG-3100 Python Controller
# ==========================

class gpg3100(object):
    def __init__(self, ndev=1, remote=False):
        initialize = not remote
        self.ctrl = gpg3100_controller(ndev, initialize)
        self._board_name = self.ctrl._latest_device_info.ulBoardType
        self._board_id = self.ctrl._latest_device_info.ulBoardID
        self._sampling_mode = self.ctrl._latest_device_info.ulSamplingMode
        self._ch_count_s = self.ctrl._latest_device_info.ulChCountS
        self._ch_count_d = self.ctrl._latest_device_info.ulChCountD
        self._resolution = self.ctrl._latest_device_info.ulResolution
        self._input_range = SamplingRange(self.ctrl._latest_device_info.ulRange)
        self._isolation = self.ctrl._latest_device_info.ulIsolation
        self._ch_di = self.ctrl._latest_device_info.ulDI
        self._ch_do = self.ctrl._latest_device_info.ulDO
        
        self._status_open = True
        self.use_singleend()
        self._status_di = [0] * self._ch_di
        self._status_do = [0] * self._ch_do
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
        elif output=='continue':
            self.ctrl.close_ex(AfterCloseExOutput.DA_OUTPUT_MAINTAIN)
        else:
            raise ValueError, "output = 'terminate', 'conitnue'"
        self._status_open = False
        return
    
    def use_singleend(self):
        self.change_sampling_method('AD_INPUT_SINGLE')
        return
        
    def use_differential(self):
        self.change_sampling_method('AD_INPUT_DIFF')
        return
        
    def change_sampling_method(self, method):
        method = SamplingMethod.verify(method)
        if method=='AD_INPUT_SINGLE': ch_count = self._ch_count_s
        elif method=='AD_INPUT_DIFF': ch_count = self._ch_count_d
        self._single_diff = method
        self._ch_count = ch_count
        self._input_ch = range(ch_count)
        self._status_input_value = [0] * ch_count
        self._status_input_value_digit = [0] * ch_count
        self._status_input_range = [0] * ch_count
        self.set_range(self._input_range.available[0])
        return
        
    def set_range(self, ad_range, ch=None):
        self._set_input(self._status_input_range, ad_range, ch, self._input_range.verify)
        return

    def _set_input(self, config, value, ch, checker=None):
        value_is_list = (type(value) in [list, tuple])
        ch_is_list = (type(ch) in [list, tuple])
        if (not value_is_list) & (not ch_is_list):
            if ch is None:
                value = [value] * len(self._input_ch)
                ch = range(len(self._input_ch))
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
            config[c] = v
            continue
        return
    
    def input(self):
        dd = self.ctrl.input_ad(self._input_ch, self._status_input_range, self._single_diff)
        da = self._analogize(dd, self._status_input_range)
        self._status_input_value = da
        self._status_input_value_digit = dd
        return da
    
    def _analogize(self, data, drange):
        return [r.analogize(d, 2**self._resolution) for d,r in zip(data, drange)]
    
    def input_series(self, freq, num):
        self.ctrl.set_sampling_config(self._input_ch, self._status_input_range, num,
                                      freq, None, self._sampling_mode, self._single_diff)
        self.ctrl.start_sampling()
        dd = self.ctrl.get_sampling_data(self._ch_count * num)
        dd = numpy.array(dd).reshape(num, self._ch_count)
        da = self._analogize(dd.T, self._status_input_range)
        return numpy.array(da).T
    
    def input_sync(self, freq, num, master_slave):
        """
        master_slave : 'AD_MASTER_MODE' or 'AD_SLAVE_MODE'
        """
        self.ctrl.set_sampling_config(self._input_ch, self._status_input_range, num,
                                      freq, None, self._sampling_mode, self._single_diff)
        self.ctrl.sync_sampling(master_slave)
        return
    
    def get_status(self):
        return self.ctrl.get_status()
    
    def get_sampling_data(self, num):
        dd = self.ctrl.get_sampling_data(self._ch_count * num)
        dd = numpy.array(dd).reshape(num, self._ch_count)
        da = self._analogize(dd.T, self._status_input_range)
        return numpy.array(da).T
    
    def read_board_name(self):
        return self._board_name
    
    def read_board_id(self):
        return self._board_id
        
    def read_sampling_mode(self):
        return self._sampling_mode
        
    def read_single_diff(self):
        return self._single_diff
    
    def read_ch_count(self):
        return self._ch_count

    def read_resolution(self):
        return self._resolution

    def read_input_range(self):
        return self._input_range

    def read_isolation(self):
        return self._isolation

    def read_ch_di(self):
        return self._ch_di

    def read_ch_do(self):
        return self._ch_do

    def read_status_open(self):
        return self._status_open

    def read_input_ch(self):
        return self._input_ch

    def read_status_input_value(self):
        return self._status_input_value

    def read_status_input_value_digit(self):
        return self._status_input_value_digit

    def read_status_input_range(self):
        return self._status_input_range

    def read_status_di(self):
        return self._status_di

    def read_status_do(self):
        return self._status_do

    def print_status(self):
        msg = 'open: %s\n'%(self._status_open)
        msg += 'ch: %s\n'%(self._input_ch)
        val = '[' + ', '.join(['%.3f'%v for v in self._status_input_value]) + ']'
        msg += 'output(value): %s\n'%(val)
        msg += 'output(bit): %s\n'%(self._status_input_value_digit)
        msg += 'output(range): %s\n'%([str(r) for r in self._status_input_range])
        msg += 'DI: %s, DO: %s\n'%(self._status_di, self._status_do)
        print(msg)
        return msg


class gpg3100_controller(object):
    ndev = int()
    
    def __init__(self, ndev=1, initialize=True):
        self.ndev = ndev
        if initialize: self.initialize()
        return
    
    def _log(self, msg):
        print('Interface GPG3100(%d): %s'%(self.ndev, msg))
        return
        
    def _error_check(self, error_no):
        ErrorGPG3100.check(error_no)
        return
        
    def initialize(self):
        self.open()
        self.get_device_info()
        self.get_sampling_config()
        return
        
    def open(self):
        self._log('open')
        ret = libgpg3100.AdOpen(self.ndev)
        self._error_check(ret)
        return
    
    def close(self):
        self._log('close')
        ret = libgpg3100.AdClose(self.ndev)
        self._error_check(ret)
        return
    
    def get_device_info(self):
        self._log('get_device_info')
        info = libgpg3100.ADBOARDSPEC()
        ret = libgpg3100.AdGetDeviceInfo(self.ndev, info)
        self._error_check(ret)
        self._latest_device_info = info
        resolution = info.ulResolution
        if resolution==8: self._data_size = ctypes.c_ubyte
        if resolution==10: self._data_size = ctypes.c_ushort
        if resolution==12: self._data_size = ctypes.c_ushort
        if resolution==16: self._data_size = ctypes.c_ushort
        if resolution==24: self._data_size = ctypes.c_ulong
        print(info)
        return info
    
    def get_common_pci_info(self):
        self._log('get_common_pci_info')
        device_id = ctypes.c_ulong(0)
        vendor_id = ctypes.c_ulong(0)
        class_code = ctypes.c_ulong(0)
        revision_id = ctypes.c_ulong(0)
        base_address0 = ctypes.c_ulong(0)
        base_address1 = ctypes.c_ulong(0)
        base_address2 = ctypes.c_ulong(0)
        base_address3 = ctypes.c_ulong(0)
        base_address4 = ctypes.c_ulong(0)
        base_address5 = ctypes.c_ulong(0)
        subsystem_id = ctypes.c_ulong(0)
        subsystem_vendor_id = ctypes.c_ulong(0)
        interrupt_line = ctypes.c_ulong(0)
        board_id = ctypes.c_ulong(0)
        ret = libgpg3100.AdCommonGetPciDeviceInfo(self.ndev, device_id, vendor_id, class_code,
                                                  revision_id, base_address0, base_address1, 
                                                  base_address2, base_address3, base_address4,
                                                  base_address5, subsystem_id, subsystem_vendor_id,
                                                  interrupt_line, board_id)
        self._error_check(ret)
        device_id = device_id.value
        vendor_id = vendor_id.value
        class_code = class_code.value
        revision_id = revision_id.value
        base_address0 = base_address0.value
        base_address1 = base_address1.value
        base_address2 = base_address2.value
        base_address3 = base_address3.value
        base_address4 = base_address4.value
        base_address5 = base_address5.value
        subsystem_id = subsystem_id.value
        subsystem_vendor_id = subsystem_vendor_id.value
        interrupt_line = interrupt_line.value
        board_id = board_id.value
        print('DeviceID : %d'%(device_id))
        print('VendorID : %d'%(vendor_id))
        print('ClassCode : %d'%(class_code))
        print('RevisionID : %d'%(revision_id))
        print('BaseAddress0 : %d'%(base_address0))
        print('BaseAddress1 : %d'%(base_address1))
        print('BaseAddress2 : %d'%(base_address2))
        print('BaseAddress3 : %d'%(base_address3))
        print('BaseAddress4 : %d'%(base_address4))
        print('BaseAddress5 : %d'%(base_address5))
        print('SubsystemID : %d'%(subsystem_id))
        print('SubsystemVendorID : %d'%(subsystem_vendor_id))
        print('InterruptLine : %d'%(interrupt_line))
        print('BoardID : %d'%(board_id))
        return (device_id, vendor_id, class_code, revision_id, base_address0, base_address1,
                base_address2, base_address3, base_address4, base_address5, subsystem_id,
                subsystem_vendor_id, interrupt_line, board_id)
    
    def set_board_config(self, callback_func, user_data):
        self._log('set_board_config')
        ret = libgpg3100.AdSetBoardConfig(self.ndev,  None, callbacl_func, user_data)
        self._error_check(ret)
        return
    
    def get_board_config(self):
        self._log('get_board_config')
        factor = ctypes.c_ulong(0)
        ret = libgpg3100.AdGetBoardConfig(self.ndev, factor)
        self._error_check(ret)
        factor = SamplingEvent.get_element(factor.value)
        print('SmplEventFactor = %d (%s)'%(factor, factor))
        return factor
    
    def set_sampling_config(self, chs=None, ranges=None, sample_num=None, freq=None, event_num=None,
                            mode=None, single_diff=None, trig_mode=None, trig_point=None, 
                            trig_delay=None, trig_ch=None, trig_level1=None, trig_level2=None,
                            clock_edge=None, trig_pulse=None, trig_edge=None, trig_di=None, fast=None):
        self._log('set_sampling_config')
        config = self._create_sampling_config(chs, ranges, sample_num, freq, event_num, 
                                              mode, single_diff, trig_mode, trig_point,
                                              trig_delay, trig_ch, trig_level1, trig_level2,
                                              clock_edge, trig_pulse, trig_edge, trig_di, fast)
        ret = libgpg3100.AdSetSamplingConfig(self.ndev, config)
        self._error_check(ret)
        self._latest_sampling_config = config
        return
    
    def _create_sampling_config(self, chs, ranges, sample_num, freq, event_num, 
                                mode, single_diff, trig_mode, trig_point,
                                trig_delay, trig_ch, trig_level1, trig_level2,
                                clock_edge, trig_pulse, trig_edge, trig_di, fast):
        
        if mode is None: mode = self._latest_sampling_config.ulSamplingMode
        if single_diff is None: single_diff = self._latest_sampling_config.ulSingleDiff
        if sample_num is None: sample_num = self._latest_sampling_config.ulSmplNum
        if event_num is None: event_num = self._latest_sampling_config.ulSmplEventNum
        if freq is None: freq = self._latest_sampling_config.fSmplFreq
        if trig_point is None: trig_point = self._latest_sampling_config.ulTrigPoint
        if trig_mode is None: trig_mode = self._latest_sampling_config.ulTrigMode
        if trig_delay is None: trig_delay = self._latest_sampling_config.lTrigDelay
        if trig_ch is None: trig_ch = self._latest_sampling_config.ulTrigCh
        if trig_level1 is None: trig_level1 = self._latest_sampling_config.fTriglevel1
        if trig_level2 is None: trig_level2 = self._latest_sampling_config.fTriglevel2
        if clock_edge is None: clock_edge = self._latest_sampling_config.ulEClkEdge
        if trig_pulse is None: trig_pulse = self._latest_sampling_config.ulATrgPulse
        if trig_edge is None: trig_edge = self._latest_sampling_config.ulTrigEdge
        if trig_di is None: trig_di = self._latest_sampling_config.ulTrigDI
        if fast is None: fast = self._latest_sampling_config.ulFastMode
        
        ch_config, ch_no = self._create_sampling_ch_config(chs, ranges)
        
        config = libgpg3100.ADSMPLREQ()
        config.ulChCount = ch_no
        config.SmplChReq = ch_config
        config.ulSamplingMode = SamplingFormula.verify(mode)
        config.ulSingleDiff = SamplingMethod.verify(single_diff)
        config.ulSmplNum = SamplingNum.verify(sample_num)
        config.ulSmplEventNum = SamplingEventNum.verify(event_num)
        config.fSmplFreq = SamplingFreq.verify(freq)
        config.ulTrigPoint = TriggerPoint.verify(trig_point)
        config.ulTrigMode = TriggerMode.verify(trig_mode)
        config.lTrigDelay = TriggerDelay.verify(trig_delay)
        config.ulTrigCh = int(trig_ch)
        config.fTriglevel1 = float(trig_level1)
        config.fTriglevel2 = float(trig_level2)
        config.ulEClkEdge = TriggerEdge.verify(clock_edge)
        config.ulATrgPulse = AnalogTriggerEdge.verify(trig_pulse)
        config.ulTrigEdge = TriggerEdge.verify(trig_edge)
        config.ulTrigDI = trig_di
        config.ulFastMode = FastMode.verify(fast)
        print(config)
        return config
    
    def _create_sampling_ch_config(self, chs, ranges):
        if chs is None: 
            chs = [req.ulChNo-1 for req in self._latest_sampling_config.SmplChReq if req.ulChNo!=0]
            pass
        
        if ranges is None: 
            ranges = [req.ulRange for req in self._latest_sampling_config.SmplChReq if req.ulChNo!=0]
            pass
        
        if not type(chs) in [list, tuple]:
            chs = [chs,]
            ranges = [ranges,]
            pass
        
        ch_config = (libgpg3100.ADSMPLCHREQ * 256)()
        for i, (ch, rang) in enumerate(zip(chs, ranges)):
            ch_config[i].ulChNo = ch + 1
            ch_config[i].ulRange = SamplingRange.verify(rang)
            continue
        
        return ch_config, len(chs)
        
    def get_sampling_config(self):
        self._log('get_sampling_config')
        config = libgpg3100.ADSMPLREQ()
        ret = libgpg3100.AdGetSamplingConfig(self.ndev, config)
        self._error_check(ret)
        self._latest_sampling_config = config
        print(config)
        return config
    
    def get_sampling_data(self, size=256):
        self._log('get_sampling_data')
        data = (self._data_size * size)(0)
        size = ctypes.c_ulong(size)
        ret = libgpg3100.AdGetSamplingData(self.ndev, data, size)
        self._error_check(ret)
        data = list(data)
        return data

    def read_sampling_buffer(self, size=256, offset=-1):
        self._log('read_sampling_buffer')
        data = (self._data_size * size)(0)
        size = ctypes.c_ulong(size)
        offset = ctypes.c_long(offset)
        ret = libgpg3100.AdReadSamplingBuffer(self.ndev, offset, size, data)
        self._error_check(ret)
        data = list(data)
        return data

    def clear_sampling_data(self):
        self._log('clear_sampling_data')
        ret = libgpg3100.AdClearSamplingData(self.ndev)
        self._error_check(ret)
        return 

    def start_sampling(self, sync='FLAG_SYNC'):
        """
        sync : 'FLAG_SYNC' or 'FLAG_ASYNC'
        """
        self._log('start_sampling')
        sync = SamplingMode.verify(sync)
        ret = libgpg3100.AdStartSampling(self.ndev, sync)
        self._error_check(ret)
        return 

    def start_file_sampling(self, save_path, filetype='FLAG_CSV'):
        """
        save_path : 
        filetype : 'FLAG_CSV' or 'FLAG_BIN'
        """
        self._log('start_file_sampling')
        path = ctypes.c_char_p(save_path)
        flag = FileType.verify(filetype)
        ret = libgpg3100.AdStartFileSampling(self.ndev, path, flag)
        self._error_check(ret)
        return 

    def sync_sampling(self, mode):
        """
        mode : 'AD_MASTER_MODE' or 'AD_SLAVE_MODE'
        """
        self._log('sync_sampling')
        mode = SyncSampling.verify(mode)
        ret = libgpg3100.AdSyncSampling(self.ndev, mode)
        self._error_check(ret)
        return 

    def trigger_sampling(self, ch, repeat=1024):
        self._log('trigger_sampling')
        ret = libgpg3100.AdTriggerSampling(self.ndev, ch, 0, 0, 0, 0, num)
        self._error_check(ret)
        return 

    def mem_trigger_sampling(self, num=1024, repeat=1):
        self._log('mem_trigger_sampling')
        ret = libgpg3100.AdMemTriggerSampling(self.ndev, 0, None, num, repeat, 0, 0.0, 0, 0)
        self._error_check(ret)
        return 

    def stop_sampling(self):
        self._log('stop_sampling')
        ret = libgpg3100.AdStopSampling(self.ndev)
        self._error_check(ret)
        return 

    def get_status(self):
        #self._log('get_status')
        status = ctypes.c_ulong(0)
        count = ctypes.c_ulong(0)
        rest = ctypes.c_ulong(0)
        ret = libgpg3100.AdGetStatus(self.ndev, status, count, rest)
        self._error_check(ret)
        status = SamplingStatus.verify(status.value)
        count = count.value
        rest = rest.value
        #print('status : %d (%s)'%(status, status))
        #print('count : %d'%(count))
        #print('rest : %d'%(rest))
        return status, count, rest

    def input_ad(self, chs=None, ranges=None, singlediff='AD_INPUT_SINGLE'):
        """
        ch : 
        singlediff : 'AD_INPUT_SINGLE' or 'AD_INPUT_DIFF'
        """
        self._log('input_ad')
        singlediff = SamplingMethod.verify(singlediff)
        ch_config, ch_no = self._create_sampling_ch_config(chs, ranges)
        data = (self._data_size * ch_no)(0)
        ret = libgpg3100.AdInputAD(self.ndev, ch_no, singlediff, ch_config, data)
        self._error_check(ret)
        data = list(data)
        return data
        
    def set_input_ad_ex(self, chs=None, ranges=None, singlediff='AD_INPUT_SINGLE'):
        self._log('set_input_ad_ex')
        singlediff = SamplingMethod.verify(singlediff)
        ch_config, ch_no = self._create_sampling_ch_config(chs, ranges)
        self._input_ad_ex_chno = ch_no
        ret = libgpg3100.AdSetInputADEx(self.ndev, ch_no, singlediff, ch_config)
        self._error_check(ret)
        return 

    def input_ad_ex(self):
        self._log('input_ad_ex')
        data = (self._data_size * self._input_ad_ex_chno)(0)
        ret = libgpg3100.AdInputADEx(self.ndev, data)
        self._error_check(ret)
        data = list(data)
        return data
        
    def input_di(self):
        self._log('input_di')
        data = ctypes.c_ulong(0)
        ret = libgpg3100.AdInputDI(self.ndev, data)
        self._error_check(ret)
        data = data.value
        return data

    def output_do(self, data):
        self._log('output_do')
        ret = libgpg3100.AdOutputDO(self.ndev, data)
        self._error_check(ret)
        return 
    
    def set_range_event(self, event_mask, stop_mode):
        self._log('set_range_event')
        ret = libgpg3100.AdSetRangeEvent(self.ndev, event_mask, stop_mode)
        self._error_check(ret)
        return 
    
    def reset_range_event(self):
        self._log('reset_range_event')
        ret = libgpg3100.AdResetRangeEvent(self.ndev)
        self._error_check(ret)
        return 

    def get_range_event_status(self):
        self._log('get_range_event_status')
        ch = ctypes.c_ulong(0)
        status = ctypes.c_ulong(0)
        ret = libgpg3100.AdGetRangeEventStatus(self.ndev, ch, status)
        self._error_check(ret)
        ch = ch.value
        status = RangeStatus.verify(status)
        print('ch : %d'%(ch))
        print('status : %d (%s)'%(status, status))
        return ch, status

    def data_conv(self):
        #self._log('data_conv')
        #ret = libgpg3100.AdDataConv(self.ndev)
        #self._error_check(ret)
        return 

    def read_file(self):
        #self._log('read_file')
        #ret = libgpg3100.AdReadFile(self.ndev)
        #self._error_check(ret)
        return 

    def fn_conv(self):
        #self._log('fn_conv')
        #ret = libgpg3100.fnConv(self.ndev)
        #self._error_check(ret)
        return 

    def callback_proc(self):
        #self._log('callback_proc')
        #ret = libgpg3100.CallbackProc(self.ndev)
        #self._error_check(ret)
        return 

    def fifo_get_sampling_data(self):
        #self._log('fifo_get_sampling_data')
        #ret = libgpg3100.AdFifoGetSamplingData(self.ndev)
        #self._error_check(ret)
        return 








