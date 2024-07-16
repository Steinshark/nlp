
import ctypes
import numpy

import pyinterface
import libgpg7400
pIE = pyinterface.IdentiferElement


# Identifer Wrapper
# =================

class AxisConfig(pyinterface.BitIdentifer):
    size = 16
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('X', 'OFF', 'ON')
    bits[1].set_params('Y', 'OFF', 'ON')
    bits[2].set_params('Z', 'OFF', 'ON')
    bits[3].set_params('U', 'OFF', 'ON')
    pass

class OpenFlag(pyinterface.Identifer):
    MTR_FLAG_NORMAL = pIE('MTR_FLAG_NORMAL', libgpg7400.MTR_FLAG_NORMAL)
    MTR_FLAG_OVERLAPPED = pIE('MTR_FLAG_OVERLAPPED', libgpg7400.MTR_FLAG_OVERLAPPED)
    MTR_ILOCK_RESET_OFF = pIE('MTR_ILOCK_RESET_OFF', libgpg7400.MTR_ILOCK_RESET_OFF)
    MTR_EXT_RESET_OFF = pIE('MTR_EXT_RESET_OFF', libgpg7400.MTR_EXT_RESET_OFF)
    MTR_FLAG_SHARE = pIE('MTR_FLAG_SHARE', libgpg7400.MTR_FLAG_SHARE)
    pass

class ResetMode(pyinterface.Identifer):
    MTR_RESET_CTL = pIE('MTR_RESET_CTL', libgpg7400.MTR_RESET_CTL)
    MTR_RESET_MOTION = pIE('MTR_RESET_MOTION', libgpg7400.MTR_RESET_MOTION)
    pass

class PulseOutConfigSection(pyinterface.Identifer):
    MTR_METHOD = pIE('MTR_METHOD', libgpg7400.MTR_METHOD)
    MTR_IDLING = pIE('MTR_IDLING', libgpg7400.MTR_IDLING)
    MTR_FINISH_FLAG = pIE('MTR_FINISH_FLAG', libgpg7400.MTR_FINISH_FLAG)
    MTR_SYNC_OUT = pIE('MTR_SYNC_OUT', libgpg7400.MTR_SYNC_OUT)
    pass

class PulseOutMethod(pyinterface.BitIdentifer):
    size = 16
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('PULSE', 'OFF', 'ON')
    bits[1].set_params('OUT', 'OFF', 'ON')
    bits[2].set_params('DIR', 'OFF', 'ON')
    bits[3].set_params('WAIT', 'OFF', 'ON')
    bits[4].set_params('DUTY', 'OFF', 'ON')
    pass

class FinishFlag(pyinterface.Identifer):
    MTR_PULSE_OUT = pIE('MTR_PULSE_OUT', libgpg7400.MTR_PULSE_OUT)
    MTR_INP = pIE('MTR_INP', libgpg7400.MTR_INP)
    MTR_PULSE_OFF = pIE('MTR_PULSE_OFF', libgpg7400.MTR_PULSE_OFF)
    pass

class SyncOutMode(pyinterface.Identifer):
    MTR_SYNC_OFF = pIE('MTR_SYNC_OFF', libgpg7400.MTR_SYNC_OFF)
    MTR_COMP1 = pIE('MTR_COMP1', libgpg7400.MTR_COMP1)
    MTR_COMP2 = pIE('MTR_COMP2', libgpg7400.MTR_COMP2)
    MTR_COMP3 = pIE('MTR_COMP3', libgpg7400.MTR_COMP3)
    MTR_COMP4 = pIE('MTR_COMP4', libgpg7400.MTR_COMP4)
    MTR_COMP5 = pIE('MTR_COMP5', libgpg7400.MTR_COMP5)
    MTR_ACC_START = pIE('MTR_ACC_START', libgpg7400.MTR_ACC_START)
    MTR_ACC_FINISH = pIE('MTR_ACC_FINISH', libgpg7400.MTR_ACC_FINISH)
    MTR_DEC_START = pIE('MTR_DEC_START', libgpg7400.MTR_DEC_START)
    MTR_DEC_FINISH = pIE('MTR_DEC_FINISH', libgpg7400.MTR_DEC_FINISH)
    pass

class LimitConfigSection(pyinterface.Identifer):
    MTR_MASK = pIE('MTR_MASK', libgpg7400.MTR_MASK)
    MTR_LOGIC = pIE('MTR_LOGIC', libgpg7400.MTR_LOGIC)
    MTR_SD_FUNC = pIE('MTR_SD_FUNC', libgpg7400.MTR_SD_FUNC)
    MTR_SD_ACTIVE = pIE('MTR_SD_ACTIVE', libgpg7400.MTR_SD_ACTIVE)
    MTR_ORG_FUNC = pIE('MTR_ORG_FUNC', libgpg7400.MTR_ORG_FUNC)
    MTR_ORG_ACTIVE = pIE('MTR_ORG_ACTIVE', libgpg7400.MTR_ORG_ACTIVE)
    MTR_ORG_EZ_COUNT = pIE('MTR_ORG_EZ_COUNT', libgpg7400.MTR_ORG_EZ_COUNT)
    MTR_ALM_FUNC = pIE('MTR_ALM_FUNC', libgpg7400.MTR_ALM_FUNC)
    MTR_SIGNAL_FILTER = pIE('MTR_SIGNAL_FILTER', libgpg7400.MTR_SIGNAL_FILTER)
    MTR_EL_FUNC = pIE('MTR_EL_FUNC', libgpg7400.MTR_EL_FUNC)
    MTR_EZ_ACTIVE = pIE('MTR_EZ_ACTIVE', libgpg7400.MTR_EZ_ACTIVE)
    MTR_LTC_FUNC = pIE('MTR_LTC_FUNC', libgpg7400.MTR_LTC_FUNC)
    MTR_CLR_FUNC = pIE('MTR_CLR_FUNC', libgpg7400.MTR_CLR_FUNC)
    MTR_PCS_FUNC = pIE('MTR_PCS_FUNC', libgpg7400.MTR_PCS_FUNC)
    MTR_PCS_ACTIVE = pIE('MTR_PCS_ACTIVE', libgpg7400.MTR_PCS_ACTIVE)
    pass

class LimitConfigLogic(pyinterface.BitIdentifer):
    size = 16
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('SD', 'OFF', 'ON')
    bits[2].set_params('EL', 'OFF', 'ON')
    bits[5].set_params('ORG', 'OFF', 'ON')
    bits[6].set_params('ALM', 'OFF', 'ON')
    bits[8].set_params('INP', 'OFF', 'ON')
    bits[9].set_params('PCS', 'OFF', 'ON')
    pass

class SDFunc(pyinterface.Identifer):
    MTR_CHANGE_SD_SPEED = pIE('MTR_CHANGE_SD_SPEED', libgpg7400.MTR_CHANGE_SD_SPEED)
    MTR_DEC_STOP_SIGNAL = pIE('MTR_DEC_STOP_SIGNAL', libgpg7400.MTR_DEC_STOP_SIGNAL)
    MTR_SD_OFF = pIE('MTR_SD_OFF', libgpg7400.MTR_SD_OFF)
    pass

class SDActive(pyinterface.Identifer):
    MTR_SIGNAL_LEVEL = pIE('MTR_SIGNAL_LEVEL', libgpg7400.MTR_SIGNAL_LEVEL)
    MTR_SIGNAL_LATCH = pIE('MTR_SIGNAL_LATCH', libgpg7400.MTR_SIGNAL_LATCH)
    pass

class ORGFunc(pyinterface.Identifer):
    MTR_ORG_STOP = pIE('MTR_ORG_STOP', libgpg7400.MTR_ORG_STOP)
    MTR_ORG_DEC_EZ_STOP = pIE('MTR_ORG_DEC_EZ_STOP', libgpg7400.MTR_ORG_DEC_EZ_STOP)
    MTR_ORG_EZ_STOP = pIE('MTR_ORG_EZ_STOP', libgpg7400.MTR_ORG_EZ_STOP)
    MTR_ORG_REVERSAL = pIE('MTR_ORG_REVERSAL', libgpg7400.MTR_ORG_REVERSAL)
    MTR_ORG_REV_EZ_STOP = pIE('MTR_ORG_REV_EZ_STOP', libgpg7400.MTR_ORG_REV_EZ_STOP)
    MTR_ORG_STOP_ZERO = pIE('MTR_ORG_STOP_ZERO', libgpg7400.MTR_ORG_STOP_ZERO)
    MTR_ORG_EZ_STOP_ZERO = pIE('MTR_ORG_EZ_STOP_ZERO', libgpg7400.MTR_ORG_EZ_STOP_ZERO)
    MTR_ORG_REV_EZ_ZERO = pIE('MTR_ORG_REV_EZ_ZERO', libgpg7400.MTR_ORG_REV_EZ_ZERO)
    pass

class ALMFunc(pyinterface.Identifer):
    MTR_ALM_STOP = pIE('MTR_ALM_STOP', libgpg7400.MTR_ALM_STOP)
    MTR_ALM_DEC_STOP = pIE('MTR_ALM_DEC_STOP', libgpg7400.MTR_ALM_DEC_STOP)
    pass

class SignalFilter(pyinterface.Identifer):
    MTR_OFF = pIE('MTR_OFF', libgpg7400.MTR_OFF)
    MTR_ON = pIE('MTR_ON', libgpg7400.MTR_ON)
    pass

class ELFunc(pyinterface.Identifer):
    MTR_EL_STOP = pIE('MTR_EL_STOP', libgpg7400.MTR_EL_STOP)
    MTR_EL_DEC_STOP = pIE('MTR_EL_DEC_STOP', libgpg7400.MTR_EL_DEC_STOP)
    pass

class EZActive(pyinterface.Identifer):
    MTR_DOWN_EDGE = pIE('MTR_DOWN_EDGE', libgpg7400.MTR_DOWN_EDGE)
    MTR_UP_EDGE = pIE('MTR_UP_EDGE', libgpg7400.MTR_UP_EDGE)
    pass

class LTCFunc(pyinterface.Identifer):
    MTR_DOWN_EDGE = pIE('MTR_DOWN_EDGE', libgpg7400.MTR_DOWN_EDGE)
    MTR_UP_EDGE = pIE('MTR_UP_EDGE', libgpg7400.MTR_UP_EDGE)
    pass

class CLRFunc(pyinterface.Identifer):
    MTR_DOWN_EDGE = pIE('MTR_DOWN_EDGE', libgpg7400.MTR_DOWN_EDGE)
    MTR_UP_EDGE = pIE('MTR_UP_EDGE', libgpg7400.MTR_UP_EDGE)
    MTR_LOW_LEVEL = pIE('MTR_LOW_LEVEL', libgpg7400.MTR_LOW_LEVEL)
    MTR_HIGH_LEVEL = pIE('MTR_HIGH_LEVEL', libgpg7400.MTR_HIGH_LEVEL)
    pass

class PCSFunc(pyinterface.Identifer):
    MTR_OFF = pIE('MTR_OFF', libgpg7400.MTR_OFF)
    MTR_ON = pIE('MTR_ON', libgpg7400.MTR_ON)
    MTR_EXT_START = pIE('MTR_EXT_START', libgpg7400.MTR_EXT_START)
    pass

class CounterConfigSection(pyinterface.Identifer):
    MTR_ENCODER_MODE = pIE('MTR_ENCODER_MODE', libgpg7400.MTR_ENCODER_MODE)
    MTR_ENCODER_CLEAR = pIE('MTR_ENCODER_CLEAR', libgpg7400.MTR_ENCODER_CLEAR)
    MTR_COUNTER_CLEAR_ORG = pIE('MTR_COUNTER_CLEAR_ORG', libgpg7400.MTR_COUNTER_CLEAR_ORG)
    MTR_COUNTER_CLEAR_CLR = pIE('MTR_COUNTER_CLEAR_CLR', libgpg7400.MTR_COUNTER_CLEAR_CLR)
    MTR_LATCH_MODE = pIE('MTR_LATCH_MODE', libgpg7400.MTR_LATCH_MODE)
    MTR_DECLINO_MODE = pIE('MTR_DECLINO_MODE', libgpg7400.MTR_DECLINO_MODE)
    MTR_SOFT_LATCH = pIE('MTR_SOFT_LATCH', libgpg7400.MTR_SOFT_LATCH)
    pass

class EncoderMode(pyinterface.Identifer):
    MTR_SINGLE = pIE('MTR_SINGLE', libgpg7400.MTR_SINGLE)
    MTR_DOUBLE = pIE('MTR_DOUBLE', libgpg7400.MTR_DOUBLE)
    MTR_QUAD = pIE('MTR_QUAD', libgpg7400.MTR_QUAD)
    MTR_UP_DOWN = pIE('MTR_UP_DOWN', libgpg7400.MTR_UP_DOWN)
    pass

class CounterClearORG(pyinterface.BitIdentifer):
    size = 16
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('CU1R', 'OFF', 'ON')
    bits[1].set_params('CU2R', 'OFF', 'ON')
    bits[2].set_params('CU3R', 'OFF', 'ON')
    pass

class CounterClearCLR(pyinterface.BitIdentifer):
    size = 16
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('CU1C', 'OFF', 'ON')
    bits[1].set_params('CU2C', 'OFF', 'ON')
    bits[2].set_params('CU3C', 'OFF', 'ON')
    pass

class LatchMode(pyinterface.Identifer):
    MTR_OFF = pIE('MTR_OFF', libgpg7400.MTR_OFF)
    MTR_ORG = pIE('MTR_ORG', libgpg7400.MTR_ORG)
    MTR_LTC = pIE('MTR_LTC', libgpg7400.MTR_LTC)
    MTR_COMP4 = pIE('MTR_COMP4', libgpg7400.MTR_COMP4)
    MTR_COMP5 = pIE('MTR_COMP5', libgpg7400.MTR_COMP5)
    pass

class DeclinoMode(pyinterface.Identifer):
    MTR_DECLINO = pIE('MTR_DECLINO', libgpg7400.MTR_DECLINO)
    MTR_SPEED = pIE('MTR_SPEED', libgpg7400.MTR_SPEED)
    pass

class Comparator(pyinterface.Identifer):
    MTR_COMP1 = pIE('MTR_COMP1', libgpg7400.MTR_COMP1)
    MTR_COMP2 = pIE('MTR_COMP2', libgpg7400.MTR_COMP2)
    MTR_COMP3 = pIE('MTR_COMP3', libgpg7400.MTR_COMP3)
    MTR_COMP4 = pIE('MTR_COMP4', libgpg7400.MTR_COMP4)
    MTR_COMP5 = pIE('MTR_COMP5', libgpg7400.MTR_COMP5)
    pass

class SyncSection(pyinterface.Identifer):
    MTR_START_MODE = pIE('MTR_START_MODE', libgpg7400.MTR_START_MODE)
    MTR_EXT_STOP = pIE('MTR_EXT_STOP', libgpg7400.MTR_EXT_STOP)
    MTR_START_LINE = pIE('MTR_START_LINE', libgpg7400.MTR_START_LINE)
    MTR_STOP_LINE = pIE('MTR_STOP_LINE', libgpg7400.MTR_STOP_LINE)
    pass

class StartMode(pyinterface.Identifer):
    MTR_NO = pIE('MTR_NO', libgpg7400.MTR_NO)
    MTR_X = pIE('MTR_X', libgpg7400.MTR_X)
    MTR_Y = pIE('MTR_Y', libgpg7400.MTR_Y)
    MTR_Z = pIE('MTR_Z', libgpg7400.MTR_Z)
    MTR_U = pIE('MTR_U', libgpg7400.MTR_U)
    MTR_SYNC_X = pIE('MTR_SYNC_X', libgpg7400.MTR_SYNC_X)
    MTR_SYNC_Y = pIE('MTR_SYNC_Y', libgpg7400.MTR_SYNC_Y)
    MTR_SYNC_Z = pIE('MTR_SYNC_Z', libgpg7400.MTR_SYNC_Z)
    MTR_SYNC_U = pIE('MTR_SYNC_U', libgpg7400.MTR_SYNC_U)
    MTR_SYNC_EXT = pIE('MTR_SYNC_EXT', libgpg7400.MTR_SYNC_EXT)
    pass

class ExtStop(pyinterface.Identifer):
    MTR_OFF = pIE('MTR_OFF', libgpg7400.MTR_OFF)
    MTR_CSTP_STOP = pIE('MTR_CSTP_STOP', libgpg7400.MTR_CSTP_STOP)
    MTR_CSTP_DEC_STOP = pIE('MTR_CSTP_DEC_STOP', libgpg7400.MTR_CSTP_DEC_STOP)
    pass

class StartLine(pyinterface.Identifer):
    SYN0 = pIE('SYN0', 0x00)
    SYN1 = pIE('SYN1', 0x01)
    SYN2 = pIE('SYN2', 0x02)
    SYN3 = pIE('SYN3', 0x03)
    SYN4 = pIE('SYN4', 0x04)
    SYN5 = pIE('SYN5', 0x05)
    SYN6 = pIE('SYN6', 0x06)
    SYN7 = pIE('SYN7', 0x07)
    pass

class StopLine(pyinterface.Identifer):
    SYN0 = pIE('SYN0', 0x00)
    SYN1 = pIE('SYN1', 0x01)
    SYN2 = pIE('SYN2', 0x02)
    SYN3 = pIE('SYN3', 0x03)
    SYN4 = pIE('SYN4', 0x04)
    SYN5 = pIE('SYN5', 0x05)
    SYN6 = pIE('SYN6', 0x06)
    SYN7 = pIE('SYN7', 0x07)
    pass

class ReviseSection(pyinterface.Identifer):
    MTR_PULSE = pIE('MTR_PULSE', libgpg7400.MTR_PULSE)
    MTR_REVISE_MODE = pIE('MTR_REVISE_MODE', libgpg7400.MTR_REVISE_MODE)
    MTR_COUNTER_MODE = pIE('MTR_COUNTER_MODE', libgpg7400.MTR_COUNTER_MODE)
    MTR_REST_RT = pIE('MTR_REST_RT', libgpg7400.MTR_REST_RT)
    MTR_REST_FT = pIE('MTR_REST_FT', libgpg7400.MTR_REST_FT)
    pass

class ReviseMode(pyinterface.Identifer):
    MTR_REVISE_OFF = pIE('MTR_REVISE_OFF', 0x00)
    MTR_BACK = pIE('MTR_BACK', libgpg7400.MTR_BACK)
    MTR_SLIP = pIE('MTR_SLIP', libgpg7400.MTR_SLIP)
    pass

class ReviseCounterMode(pyinterface.BitIdentifer):
    size = 16
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('CU1B', 'OFF', 'ON')
    bits[1].set_params('CU2B', 'OFF', 'ON')
    bits[2].set_params('CU3B', 'OFF', 'ON')
    pass

class ERCConfigSection(pyinterface.Identifer):
    MTR_AUTO = pIE('MTR_AUTO', libgpg7400.MTR_AUTO)
    MTR_LOGIC = pIE('MTR_LOGIC', libgpg7400.MTR_LOGIC)
    MTR_WIDTH = pIE('MTR_WIDTH', libgpg7400.MTR_WIDTH)
    MTR_OFF_TIMER = pIE('MTR_OFF_TIMER', libgpg7400.MTR_OFF_TIMER)
    MTR_SIGNAL_ON = pIE('MTR_SIGNAL_ON', libgpg7400.MTR_SIGNAL_ON)
    MTR_SIGNAL_OFF = pIE('MTR_SIGNAL_OFF', libgpg7400.MTR_SIGNAL_OFF)
    pass

class ERCAuto(pyinterface.BitIdentifer):
    size = 16
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('EROE', 'OFF', 'ON')
    bits[1].set_params('EROR', 'OFF', 'ON')
    pass

class ERCLogic(pyinterface.Identifer):
    MTR_ACTIVE_LOW = pIE('MTR_ACTIVE_LOW', libgpg7400.MTR_ACTIVE_LOW)
    MTR_ACTIVE_HIGH = pIE('MTR_ACTIVE_HIGH', libgpg7400.MTR_ACTIVE_HIGH)
    pass

class ERCWidth(pyinterface.Identifer):
    MTR_12MICRO = pIE('MTR_12MICRO', libgpg7400.MTR_12MICRO)
    MTR_102MICRO = pIE('MTR_102MICRO', libgpg7400.MTR_102MICRO)
    MTR_409MICRO = pIE('MTR_409MICRO', libgpg7400.MTR_409MICRO)
    MTR_1600MICRO = pIE('MTR_1600MICRO', libgpg7400.MTR_1600MICRO)
    MTR_13M = pIE('MTR_13M', libgpg7400.MTR_13M)
    MTR_52M = pIE('MTR_52M', libgpg7400.MTR_52M)
    MTR_104M = pIE('MTR_104M', libgpg7400.MTR_104M)
    MTR_LEVEL = pIE('MTR_LEVEL', libgpg7400.MTR_LEVEL)
    pass

class ERCOffTimer(pyinterface.Identifer):
    MTR_ZERO = pIE('MTR_ZERO', libgpg7400.MTR_ZERO)
    MTR_12MICRO = pIE('MTR_12MICRO', libgpg7400.MTR_12MICRO)
    MTR_1600MICRO = pIE('MTR_1600MICRO', libgpg7400.MTR_1600MICRO)
    MTR_104M = pIE('MTR_104M', libgpg7400.MTR_104M)
    pass

class MotionSection(pyinterface.Identifer):
    MTR_JOG = pIE('MTR_JOG', libgpg7400.MTR_JOG)
    MTR_ORG = pIE('MTR_ORG', libgpg7400.MTR_ORG)
    MTR_PTP = pIE('MTR_PTP', libgpg7400.MTR_PTP)
    MTR_TIMER = pIE('MTR_TIMER', libgpg7400.MTR_TIMER)
    MTR_SINGLE_STEP = pIE('MTR_SINGLE_STEP', libgpg7400.MTR_SINGLE_STEP)
    MTR_ORG_SEARCH = pIE('MTR_ORG_SEARCH', libgpg7400.MTR_ORG_SEARCH)
    MTR_ORG_EXIT = pIE('MTR_ORG_EXIT', libgpg7400.MTR_ORG_EXIT)
    MTR_ORG_ZERO = pIE('MTR_ORG_ZERO', libgpg7400.MTR_ORG_ZERO)
    MTR_PTP_REPEAT = pIE('MTR_PTP_REPEAT', libgpg7400.MTR_PTP_REPEAT)
    MTR_REPEAT_CLEAR = pIE('MTR_REPEAT_CLEAR', libgpg7400.MTR_REPEAT_CLEAR)
    MTR_START_MODE_OFF = pIE('MTR_START_MODE_OFF', libgpg7400.MTR_START_MODE_OFF)
    pass

class MotionLineSection(pyinterface.Identifer):
    MTR_LINE_NORMAL = pIE('MTR_LINE_NORMAL', libgpg7400.MTR_LINE_NORMAL)
    MTR_LINE_REPEAT = pIE('MTR_LINE_REPEAT', libgpg7400.MTR_LINE_REPEAT)
    MTR_REPEAT_CLEAR = pIE('MTR_REPEAT_CLEAR', libgpg7400.MTR_REPEAT_CLEAR)
    MTR_START_MODE_OFF = pIE('MTR_START_MODE_OFF', libgpg7400.MTR_START_MODE_OFF)
    pass

class MotionArcSection(pyinterface.Identifer):
    MTR_ARC_NORMAL = pIE('MTR_ARC_NORMAL', libgpg7400.MTR_ARC_NORMAL)
    MTR_ARC_REPEAT = pIE('MTR_ARC_REPEAT', libgpg7400.MTR_ARC_REPEAT)
    MTR_REPEAT_CLEAR = pIE('MTR_REPEAT_CLEAR', libgpg7400.MTR_REPEAT_CLEAR)
    MTR_START_MODE_OFF = pIE('MTR_START_MODE_OFF', libgpg7400.MTR_START_MODE_OFF)
    pass

class StartMotion(pyinterface.Identifer):
    MTR_ACC = pIE('MTR_ACC', libgpg7400.MTR_ACC)
    MTR_CONST = pIE('MTR_CONST', libgpg7400.MTR_CONST)
    MTR_CONST_DEC = pIE('MTR_CONST_DEC', libgpg7400.MTR_CONST_DEC)
    pass

class MotionMode(pyinterface.Identifer):
    MTR_JOG = pIE('MTR_JOG', libgpg7400.MTR_JOG)
    MTR_ORG = pIE('MTR_ORG', libgpg7400.MTR_ORG)
    MTR_PTP = pIE('MTR_PTP', libgpg7400.MTR_PTP)
    MTR_TIMER = pIE('MTR_TIMER', libgpg7400.MTR_TIMER)
    MTR_SINGLE_STEP = pIE('MTR_SINGLE_STEP', libgpg7400.MTR_SINGLE_STEP)
    MTR_ORG_SEARCH = pIE('MTR_ORG_SEARCH', libgpg7400.MTR_ORG_SEARCH)
    MTR_ORG_EXIT = pIE('MTR_ORG_EXIT', libgpg7400.MTR_ORG_EXIT)
    MTR_ORG_ZERO = pIE('MTR_ORG_ZERO', libgpg7400.MTR_ORG_ZERO)
    MTR_LINE = pIE('MTR_LINE', libgpg7400.MTR_LINE)
    MTR_SYNC_LINE = pIE('MTR_SYNC_LINE', libgpg7400.MTR_SYNC_LINE)
    MTR_ARC = pIE('MTR_ARC', libgpg7400.MTR_ARC)
    MTR_CP = pIE('MTR_CP', libgpg7400.MTR_CP)
    MTR_LIMIT = pIE('MTR_LIMIT', libgpg7400.MTR_LIMIT)
    MTR_ABSOLUTE = pIE('MTR_ABSOLUTE', libgpg7400.MTR_ABSOLUTE)
    pass

class StopMotion(pyinterface.Identifer):
    MTR_DEC_STOP = pIE('MTR_DEC_STOP', libgpg7400.MTR_DEC_STOP)
    MTR_IMMEDIATE_STOP = pIE('MTR_IMMEDIATE_STOP', libgpg7400.MTR_IMMEDIATE_STOP)
    pass

class RepeatMoveMode(pyinterface.Identifer):
    MTR_PTP = pIE('MTR_PTP', libgpg7400.MTR_PTP)
    MTR_IP = pIE('MTR_IP', libgpg7400.MTR_IP)
    MTR_IP_SPEED = pIE('MTR_IP_SPEED', libgpg7400.MTR_IP_SPEED)
    pass

class OutputSync(pyinterface.Identifer):
    MTR_EXT_START = pIE('MTR_EXT_START', libgpg7400.MTR_EXT_START)
    MTR_EXT_STOP = pIE('MTR_EXT_STOP', libgpg7400.MTR_EXT_STOP)
    MTR_SYNC_NORMAL = pIE('MTR_SYNC_NORMAL', libgpg7400.MTR_SYNC_NORMAL)
    MTR_SYNC_SLAVE = pIE('MTR_SYNC_SLAVE', libgpg7400.MTR_SYNC_SLAVE)
    MTR_SYNC_MASTER = pIE('MTR_SYNC_MASTER', libgpg7400.MTR_SYNC_MASTER)
    pass

class ChangeSpeed(pyinterface.Identifer):
    MTR_IMMEDIATE_CHANGE = pIE('MTR_IMMEDIATE_CHANGE', libgpg7400.MTR_IMMEDIATE_CHANGE)
    MTR_ACCDEC_CHANGE = pIE('MTR_ACCDEC_CHANGE', libgpg7400.MTR_ACCDEC_CHANGE)
    MTR_LOW_SPEED = pIE('MTR_LOW_SPEED', libgpg7400.MTR_LOW_SPEED)
    MTR_DEC_LOW_SPEED = pIE('MTR_DEC_LOW_SPEED', libgpg7400.MTR_DEC_LOW_SPEED)
    pass

class StatusSection(pyinterface.Identifer):
    MTR_BUSY = pIE('MTR_BUSY', libgpg7400.MTR_BUSY)
    MTR_FINISH_STATUS = pIE('MTR_FINISH_STATUS', libgpg7400.MTR_FINISH_STATUS)
    MTR_LIMIT_STATUS = pIE('MTR_LIMIT_STATUS', libgpg7400.MTR_LIMIT_STATUS)
    MTR_INTERLOCK_STATUS = pIE('MTR_INTERLOCK_STATUS', libgpg7400.MTR_INTERLOCK_STATUS)
    MTR_PRIREG_STATUS = pIE('MTR_PRIREG_STATUS', libgpg7400.MTR_PRIREG_STATUS)
    MTR_SYNC_STATUS = pIE('MTR_SYNC_STATUS', libgpg7400.MTR_SYNC_STATUS)
    MTR_PTP_REPEAT_NUM = pIE('MTR_PTP_REPEAT_NUM', libgpg7400.MTR_PTP_REPEAT_NUM)
    MTR_IP_REPEAT_NUM = pIE('MTR_IP_REPEAT_NUM', libgpg7400.MTR_IP_REPEAT_NUM)
    pass

class StatusBusy(pyinterface.BitIdentifer):
    size = 32
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('BUSY', 'NO', 'YES')
    bits[1].set_params('ACC', 'NO', 'YES')
    bits[2].set_params('DEC', 'NO', 'YES')
    bits[3].set_params('WAIT', 'NO', 'YES')
    bits[4].set_params('SYNC', 'NO', 'YES')
    bits[5].set_params('STOP', 'OFF', 'ON')
    bits[6].set_params('INP', 'NO', 'YES')
    bits[7].set_params('CSTA', 'NO', 'YES')
    bits[8].set_params('BACK', 'NO', 'YES')
    bits[10].set_params('CMD', 'NO', 'YES')
    bits[12].set_params('ERC', 'NO', 'YES')
    bits[13].set_params('DIR', 'NO', 'YES')
    bits[14].set_params('OTH', 'NO', 'YES')
    pass

class StatusFinish(pyinterface.BitIdentifer):
    size = 32
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('FNS', 'NO', 'YES')
    bits[3].set_params('CMP1', 'NO', 'YES')
    bits[4].set_params('CMP2', 'NO', 'YES')
    bits[5].set_params('CMP3', 'NO', 'YES')
    bits[6].set_params('CMP4', 'NO', 'YES')
    bits[7].set_params('CMP5', 'NO', 'YES')
    bits[8].set_params('SD', 'NO', 'YES')
    bits[10].set_params('+EL', 'NO', 'YES')
    bits[11].set_params('-EL', 'NO', 'YES')
    bits[12].set_params('CSTP', 'NO', 'YES')
    bits[13].set_params('EA/EB', 'NO', 'YES')
    bits[14].set_params('ALM', 'NO', 'YES')
    bits[15].set_params('ERR', 'NO', 'YES')
    bits[16].set_params('OVER', 'NO', 'YES')
    pass

class StatusLimit(pyinterface.BitIdentifer):
    size = 32
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('SD', 'NOINPUT', 'INPUT')
    bits[2].set_params('+EL', 'NOINPUT', 'INPUT')
    bits[3].set_params('-EL', 'NOINPUT', 'INPUT')
    bits[5].set_params('ORG', 'NOINPUT', 'INPUT')
    bits[6].set_params('ALM', 'NOINPUT', 'INPUT')
    bits[8].set_params('INP', 'NOINPUT', 'INPUT')
    bits[9].set_params('CLR', 'NOINPUT', 'INPUT')
    bits[10].set_params('LTC', 'NOINPUT', 'INPUT')
    bits[11].set_params('CTSA', 'NOINPUT', 'INPUT')
    bits[12].set_params('CSTP', 'NOINPUT', 'INPUT')
    bits[13].set_params('PCS', 'NOINPUT', 'INPUT')
    bits[14].set_params('ERC', 'NOOUTPUT', 'OUTPUT')
    bits[15].set_params('EZ', 'NOINPUT', 'INPUT')
    pass

class StatusInterlock(pyinterface.Identifer):
    MTR_ILOCK_OFF = pIE('MTR_ILOCK_OFF', libgpg7400.MTR_ILOCK_OFF)
    MTR_ILOCK_ON = pIE('MTR_ILOCK_ON', libgpg7400.MTR_ILOCK_ON)
    pass

class StatusSync(pyinterface.BitIdentifer):
    size = 32
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('IPLx', 'NONE', 'SET')
    bits[1].set_params('IPLy', 'NONE', 'SET')
    bits[2].set_params('IPLz', 'NONE', 'SET')
    bits[3].set_params('IPLu', 'NONE', 'SET')
    bits[4].set_params('IPEx', 'NONE', 'SET')
    bits[5].set_params('IPEy', 'NONE', 'SET')
    bits[6].set_params('IPEz', 'NONE', 'SET')
    bits[7].set_params('IPEu', 'NONE', 'SET')
    bits[8].set_params('IPSx', 'NONE', 'SET')
    bits[9].set_params('IPSy', 'NONE', 'SET')
    bits[10].set_params('IPSz', 'NONE', 'SET')
    bits[11].set_params('IPSu', 'NONE', 'SET')
    bits[12].set_params('IPFx', 'NONE', 'SET')
    bits[13].set_params('IPFy', 'NONE', 'SET')
    bits[14].set_params('IPFz', 'NONE', 'SET')
    bits[15].set_params('IPFu', 'NONE', 'SET')
    bits[16].set_params('IPL', 'NOT', 'APPLY')
    bits[17].set_params('IPE', 'NOT', 'APPLY')
    bits[18].set_params('IPCW', 'NOT', 'APPLY')
    bits[19].set_params('IPCC', 'NOT', 'APPLY')
    bits[20].set_params('SDM0', 'OFF', 'ON')
    bits[21].set_params('SDM1', 'OFF', 'ON')
    bits[22].set_params('SED0', 'OFF', 'ON')
    bits[23].set_params('SED1', 'OFF', 'ON')
    pass

class CounterType(pyinterface.Identifer):
    MTR_ENCODER = pIE('MTR_ENCODER', libgpg7400.MTR_ENCODER)
    MTR_COUNTER = pIE('MTR_COUNTER', libgpg7400.MTR_COUNTER)
    MTR_REMAINS = pIE('MTR_REMAINS', libgpg7400.MTR_REMAINS)
    MTR_DECLINO = pIE('MTR_DECLINO', libgpg7400.MTR_DECLINO)
    MTR_ABSOLUTE = pIE('MTR_ABSOLUTE', libgpg7400.MTR_ABSOLUTE)
    MTR_LATCH = pIE('MTR_LATCH', libgpg7400.MTR_LATCH)
    pass

class SampleStatus(pyinterface.Identifer):
    MTR_REPEAT = pIE('MTR_REPEAT', libgpg7400.MTR_REPEAT)
    MTR_SAMPLE_FINISHED = pIE('MTR_SAMPLE_FINISHED', libgpg7400.MTR_SAMPLE_FINISHED)
    MTR_NOW_SAMPLING = pIE('MTR_NOW_SAMPLING', libgpg7400.MTR_NOW_SAMPLING)
    MTR_WAITING_BUSY = pIE('MTR_WAITING_BUSY', libgpg7400.MTR_WAITING_BUSY)
    MTR_FULL_BUFFER = pIE('MTR_FULL_BUFFER', libgpg7400.MTR_FULL_BUFFER)
    MTR_EMPTY_DATA = pIE('MTR_EMPTY_DATA', libgpg7400.MTR_EMPTY_DATA)
    pass

class CMDBufferSection(pyinterface.Identifer):
    MTR_CMD_MOVE = pIE('MTR_CMD_MOVE', libgpg7400.MTR_CMD_MOVE)
    MTR_CMD_CLOCK = pIE('MTR_CMD_CLOCK', libgpg7400.MTR_CMD_CLOCK)
    MTR_CMD_LOW_SPEED = pIE('MTR_CMD_LOW_SPEED', libgpg7400.MTR_CMD_LOW_SPEED)
    MTR_CMD_SPEED = pIE('MTR_CMD_SPEED', libgpg7400.MTR_CMD_SPEED)
    MTR_CMD_ACC = pIE('MTR_CMD_ACC', libgpg7400.MTR_CMD_ACC)
    MTR_CMD_DEC = pIE('MTR_CMD_DEC', libgpg7400.MTR_CMD_DEC)
    MTR_CMD_ACC_SPEED = pIE('MTR_CMD_ACC_SPEED', libgpg7400.MTR_CMD_ACC_SPEED)
    MTR_CMD_DEC_SPEED = pIE('MTR_CMD_DEC_SPEED', libgpg7400.MTR_CMD_DEC_SPEED)
    MTR_CMD_STEP = pIE('MTR_CMD_STEP', libgpg7400.MTR_CMD_STEP)
    MTR_CMD_CENTER = pIE('MTR_CMD_CENTER', libgpg7400.MTR_CMD_CENTER)
    MTR_CMD_START_MODE = pIE('MTR_CMD_START_MODE', libgpg7400.MTR_CMD_START_MODE)
    pass

class CMDMode(pyinterface.Identifer):
    MTR_CMD_JOG_P = pIE('MTR_CMD_JOG_P', libgpg7400.MTR_CMD_JOG_P)
    MTR_CMD_JOG_M = pIE('MTR_CMD_JOG_M', libgpg7400.MTR_CMD_JOG_M)
    MTR_CMD_ORG_P = pIE('MTR_CMD_ORG_P', libgpg7400.MTR_CMD_ORG_P)
    MTR_CMD_ORG_M = pIE('MTR_CMD_ORG_M', libgpg7400.MTR_CMD_ORG_M)
    MTR_CMD_ORG_EXIT_P = pIE('MTR_CMD_ORG_EXIT_P', libgpg7400.MTR_CMD_ORG_EXIT_P)
    MTR_CMD_ORG_EXIT_M = pIE('MTR_CMD_ORG_EXIT_M', libgpg7400.MTR_CMD_ORG_EXIT_M)
    MTR_CMD_ORG_SEARCH_P = pIE('MTR_CMD_ORG_SEARCH_P', libgpg7400.MTR_CMD_ORG_SEARCH_P)
    MTR_CMD_ORG_SEARCH_M = pIE('MTR_CMD_ORG_SEARCH_M', libgpg7400.MTR_CMD_ORG_SEARCH_M)
    MTR_CMD_PTP = pIE('MTR_CMD_PTP', libgpg7400.MTR_CMD_PTP)
    MTR_CMD_ORG_ZERO = pIE('MTR_CMD_ORG_ZERO', libgpg7400.MTR_CMD_ORG_ZERO)
    MTR_CMD_SINGLE_STEP_P = pIE('MTR_CMD_SINGLE_STEP_P', libgpg7400.MTR_CMD_SINGLE_STEP_P)
    MTR_CMD_SINGLE_STEP_M = pIE('MTR_CMD_SINGLE_STEP_M', libgpg7400.MTR_CMD_SINGLE_STEP_M)
    MTR_CMD_TIMER = pIE('MTR_CMD_TIMER', libgpg7400.MTR_CMD_TIMER)
    MTR_CMD_LINE = pIE('MTR_CMD_LINE', libgpg7400.MTR_CMD_LINE)
    MTR_CMD_ARC_CW = pIE('MTR_CMD_ARC_CW', libgpg7400.MTR_CMD_ARC_CW)
    MTR_CMD_ARC_CCW = pIE('MTR_CMD_ARC_CCW', libgpg7400.MTR_CMD_ARC_CCW)
    MTR_CMD_ACC_SIN = pIE('MTR_CMD_ACC_SIN', libgpg7400.MTR_CMD_ACC_SIN)
    MTR_CMD_FH_OFF = pIE('MTR_CMD_FH_OFF', libgpg7400.MTR_CMD_FH_OFF)
    MTR_CMD_SP_COMPOSE = pIE('MTR_CMD_SP_COMPOSE', libgpg7400.MTR_CMD_SP_COMPOSE)
    pass

class CMDStartMode(pyinterface.Identifer):
    MTR_CMD_CONST = pIE('MTR_CMD_CONST', libgpg7400.MTR_CMD_CONST)
    MTR_CMD_ACC_DEC = pIE('MTR_CMD_ACC_DEC', libgpg7400.MTR_CMD_ACC_DEC)
    MTR_CMD_CONST_DEC = pIE('MTR_CMD_CONST_DEC', libgpg7400.MTR_CMD_CONST_DEC)
    pass

class StartCMDBuffer(pyinterface.Identifer):
    MTR_CMD_AUTO_START = pIE('MTR_CMD_AUTO_START', libgpg7400.MTR_CMD_AUTO_START)
    MTR_CMD_STEP_START = pIE('MTR_CMD_STEP_START', libgpg7400.MTR_CMD_STEP_START)
    pass

# for structures
# - - - - - - - -
class CompConfig(pyinterface.Identifer):
    MTR_NO = pIE('MTR_NO', libgpg7400.MTR_NO)
    MTR_EQ = pIE('MTR_EQ', libgpg7400.MTR_EQ)
    MTR_EQ_UP = pIE('MTR_EQ_UP', libgpg7400.MTR_EQ_UP)
    MTR_EQ_DOWN = pIE('MTR_EQ_DOWN', libgpg7400.MTR_EQ_DOWN)
    MTR_LT = pIE('MTR_LT', libgpg7400.MTR_LT)
    MTR_GT = pIE('MTR_GT', libgpg7400.MTR_GT)
    MTR_SOFT_LIMIT = pIE('MTR_SOFT_LIMIT', libgpg7400.MTR_SOFT_LIMIT)    
    pass

class CompMotion(pyinterface.Identifer):
    MTR_NO = pIE('MTR_NO', libgpg7400.MTR_NO)
    MTR_STOP = pIE('MTR_STOP', libgpg7400.MTR_STOP)
    MTR_DEC = pIE('MTR_DEC', libgpg7400.MTR_DEC)
    MTR_CHG_REG = pIE('MTR_CHG_REG', libgpg7400.MTR_CHG_REG)
    pass

class CompCntType(pyinterface.Identifer):
    MTR_CMP_COUNTER = pIE('MTR_CMP_COUNTER', libgpg7400.MTR_CMP_COUNTER)
    MTR_CMP_ENCODER = pIE('MTR_CMP_ENCODER', libgpg7400.MTR_CMP_ENCODER)
    MTR_CMP_DECLINO = pIE('MTR_CMP_DECLINO', libgpg7400.MTR_CMP_DECLINO)
    MTR_CMP_SPEED = pIE('MTR_CMP_SPEED', libgpg7400.MTR_CMP_SPEED)
    pass

class MotionAccMode(pyinterface.Identifer):
    MTR_ACC_NORMAL = pIE('MTR_ACC_NORMAL', libgpg7400.MTR_ACC_NORMAL)
    MTR_ACC_SIN = pIE('MTR_ACC_SIN', libgpg7400.MTR_ACC_SIN)
    MTR_FH = pIE('MTR_FH', libgpg7400.MTR_FH)
    pass

class LineMode(pyinterface.Identifer):
    MTR_LINE = pIE('MTR_LINE', libgpg7400.MTR_LINE)
    MTR_LINE_JOG = pIE('MTR_LINE_JOG', libgpg7400.MTR_LINE_JOG)
    pass

class LineAccMode(pyinterface.Identifer):
    MTR_ACC_NORMAL = pIE('MTR_ACC_NORMAL', libgpg7400.MTR_ACC_NORMAL)
    MTR_ACC_SIN = pIE('MTR_ACC_SIN', libgpg7400.MTR_ACC_SIN)
    MTR_SP_COMPOSE = pIE('MTR_SP_COMPOSE', libgpg7400.MTR_SP_COMPOSE)
    pass

class CounterClearCLR(pyinterface.BitIdentifer):
    size = 16
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('CU1C', 'OFF', 'ON')
    bits[1].set_params('CU2C', 'OFF', 'ON')
    bits[2].set_params('CU3C', 'OFF', 'ON')
    pass

class ARCMode(pyinterface.Identifer):
    MTR_ARC_CW = pIE('MTR_ARC_CW', libgpg7400.MTR_ARC_CW)
    MTR_ARC_CCW = pIE('MTR_ARC_CCW', libgpg7400.MTR_ARC_CCW)
    MTR_SP_COMPOSE = pIE('MTR_SP_COMPOSE', libgpg7400.MTR_SP_COMPOSE)
    pass

class EventPulseOut(pyinterface.BitIdentifer):
    size = 16
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('STOP', 'OFF', 'ON')
    bits[1].set_params('ACC-START', 'OFF', 'ON')
    bits[2].set_params('ACC-FINISH', 'OFF', 'ON')
    bits[3].set_params('DEC-START', 'OFF', 'ON')
    bits[4].set_params('DEC-FINISH', 'OFF', 'ON')
    bits[5].set_params('NEXT', 'OFF', 'ON')
    bits[6].set_params('WRITE', 'OFF', 'ON')
    pass

class EventComparator(pyinterface.BitIdentifer):
    size = 16
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('CMP1', 'OFF', 'ON')
    bits[1].set_params('CMP2', 'OFF', 'ON')
    bits[2].set_params('CMP3', 'OFF', 'ON')
    bits[3].set_params('CMP4', 'OFF', 'ON')
    bits[4].set_params('CMP5', 'OFF', 'ON')
    pass

class EventSignal(pyinterface.BitIdentifer):
    size = 16
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('SD', 'OFF', 'ON')
    bits[1].set_params('ORG', 'OFF', 'ON')
    bits[2].set_params('CLR', 'OFF', 'ON')
    bits[3].set_params('LTC', 'OFF', 'ON')
    bits[4].set_params('ILOCK', 'OFF', 'ON')
    bits[5].set_params('EXT', 'OFF', 'ON')
    bits[6].set_params('CSTA', 'OFF', 'ON')
    pass

class SampleFreqMode(pyinterface.Identifer):
    MTR_ENCODER = pIE('MTR_ENCODER', libgpg7400.MTR_ENCODER)
    MTR_COUNTER = pIE('MTR_COUNTER', libgpg7400.MTR_COUNTER)
    MTR_DECLINO = pIE('MTR_DECLINO', libgpg7400.MTR_DECLINO)
    MTR_BUSY_SAMP = pIE('MTR_BUSY_SAMP', libgpg7400.MTR_BUSY_SAMP)
    MTR_REPEAT = pIE('MTR_REPEAT', libgpg7400.MTR_REPEAT)
    pass    


# Error Wrapper
# =============
class ErrorGPG7400(pyinterface.ErrorCode):
    MTR_ERROR_SUCCESS = pIE('MTR_ERROR_SUCCESS', libgpg7400.MTR_ERROR_SUCCESS)
    MTR_ERROR_NOT_DEVICE = pIE('MTR_ERROR_NOT_DEVICE', libgpg7400.MTR_ERROR_NOT_DEVICE)
    MTR_ERROR_NOT_OPEN = pIE('MTR_ERROR_NOT_OPEN', libgpg7400.MTR_ERROR_NOT_OPEN)
    MTR_ERROR_INVALID_DEVICE_NUMBER = pIE('MTR_ERROR_INVALID_DEVICE_NUMBER', libgpg7400.MTR_ERROR_INVALID_DEVICE_NUMBER)
    MTR_ERROR_ALREADY_OPEN = pIE('MTR_ERROR_ALREADY_OPEN', libgpg7400.MTR_ERROR_ALREADY_OPEN)
    MTR_ERROR_NOT_SUPPORTED = pIE('MTR_ERROR_NOT_SUPPORTED', libgpg7400.MTR_ERROR_NOT_SUPPORTED)
    MTR_ERROR_NOW_MOVING = pIE('MTR_ERROR_NOW_MOVING', libgpg7400.MTR_ERROR_NOW_MOVING)
    MTR_ERROR_NOW_STOPPED = pIE('MTR_ERROR_NOW_STOPPED', libgpg7400.MTR_ERROR_NOW_STOPPED)
    MTR_ERROR_NOW_SAMPLING = pIE('MTR_ERROR_NOW_SAMPLING', libgpg7400.MTR_ERROR_NOW_SAMPLING)
    MTR_ERROR_NOW_STOP_SAMPLING = pIE('MTR_ERROR_NOW_STOP_SAMPLING', libgpg7400.MTR_ERROR_NOW_STOP_SAMPLING)
    MTR_ERROR_NOW_BUSY_CMD_BUFF = pIE('MTR_ERROR_NOW_BUSY_CMD_BUFF', libgpg7400.MTR_ERROR_NOW_BUSY_CMD_BUFF)
    MTR_ERROR_NOW_STOP_CMD_BUFF = pIE('MTR_ERROR_NOW_STOP_CMD_BUFF', libgpg7400.MTR_ERROR_NOW_STOP_CMD_BUFF)
    MTR_ERROR_EEPROM_BUSY = pIE('MTR_ERROR_EEPROM_BUSY', libgpg7400.MTR_ERROR_EEPROM_BUSY)
    MTR_ERROR_WRITE_FAILED = pIE('MTR_ERROR_WRITE_FAILED', libgpg7400.MTR_ERROR_WRITE_FAILED)
    MTR_ERROR_READ_FAILED = pIE('MTR_ERROR_READ_FAILED', libgpg7400.MTR_ERROR_READ_FAILED)
    MTR_ERROR_INVALID_DEVICE = pIE('MTR_ERROR_INVALID_DEVICE', libgpg7400.MTR_ERROR_INVALID_DEVICE)
    MTR_ERROR_INVALID_AXIS = pIE('MTR_ERROR_INVALID_AXIS', libgpg7400.MTR_ERROR_INVALID_AXIS)
    MTR_ERROR_INVALID_SPEED = pIE('MTR_ERROR_INVALID_SPEED', libgpg7400.MTR_ERROR_INVALID_SPEED)
    MTR_ERROR_INVALID_ACCDEC = pIE('MTR_ERROR_INVALID_ACCDEC', libgpg7400.MTR_ERROR_INVALID_ACCDEC)
    MTR_ERROR_INVALID_PULSE = pIE('MTR_ERROR_INVALID_PULSE', libgpg7400.MTR_ERROR_INVALID_PULSE)
    MTR_ERROR_INVALID_PARAMETER = pIE('MTR_ERROR_INVALID_PARAMETER', libgpg7400.MTR_ERROR_INVALID_PARAMETER)
    MTR_ERROR_INVALID_INDEX = pIE('MTR_ERROR_INVALID_INDEX', libgpg7400.MTR_ERROR_INVALID_INDEX)
    MTR_ERROR_REPEAT_LINE_ARC = pIE('MTR_ERROR_REPEAT_LINE_ARC', libgpg7400.MTR_ERROR_REPEAT_LINE_ARC)
    MTR_ERROR_NOW_INTERLOCKED = pIE('MTR_ERROR_NOW_INTERLOCKED', libgpg7400.MTR_ERROR_NOW_INTERLOCKED)
    MTR_ERROR_IMPOSSIBLE = pIE('MTR_ERROR_IMPOSSIBLE', libgpg7400.MTR_ERROR_IMPOSSIBLE)
    MTR_ERROR_WRITE_FAILED_EEPROM = pIE('MTR_ERROR_WRITE_FAILED_EEPROM', libgpg7400.MTR_ERROR_WRITE_FAILED_EEPROM)
    MTR_ERROR_READ_FAILED_EEPROM = pIE('MTR_ERROR_READ_FAILED_EEPROM', libgpg7400.MTR_ERROR_READ_FAILED_EEPROM)
    MTR_ERROR_NOT_ALLOCATE_MEMORY = pIE('MTR_ERROR_NOT_ALLOCATE_MEMORY', libgpg7400.MTR_ERROR_NOT_ALLOCATE_MEMORY)
    MTR_ERROR_NOW_WAIT_STA = pIE('MTR_ERROR_NOW_WAIT_STA', libgpg7400.MTR_ERROR_NOW_WAIT_STA)
    MTR_ERROR_EMPTY_DATA = pIE('MTR_ERROR_EMPTY_DATA', libgpg7400.MTR_ERROR_EMPTY_DATA)
    MTR_ERROR_FULL_PRIREG = pIE('MTR_ERROR_FULL_PRIREG', libgpg7400.MTR_ERROR_FULL_PRIREG)
    MTR_ERROR_FAILED_CREATE_THREAD = pIE('MTR_ERROR_FAILED_CREATE_THREAD', libgpg7400.MTR_ERROR_FAILED_CREATE_THREAD)
    MTN_ERROR_NOT_ALLOCATE_MEMORY = pIE('MTN_ERROR_NOT_ALLOCATE_MEMORY', libgpg7400.MTN_ERROR_NOT_ALLOCATE_MEMORY)
    
    _success = MTR_ERROR_SUCCESS
    pass



# ==========================
# GPG-7400 Python Controller
# ==========================

class gpg7400_controller(object):
    ndev = int()
    
    def __init__(self, ndev=1, initialize=True):
        self.ndev = ndev
        if initialize: self.initialize()
        return
    
    def _log(self, msg):
        print('Interface GPG7400(%d): %s'%(self.ndev, msg))
        return
        
    def _error_check(self, error_no):
        ErrorGPG7400.check(error_no)
        return
        
    def initialize(self):
        self.open()
        #self.get_device_info()
        return
        
    def open(self, open_flag='MTR_FLAG_NORMAL'):
        """
        1. MtnOpen
        """
        self._log('open')
        open_flag = OpenFlag.verify(open_flag)
        ret = libgpg7400.MtnOpen(self.ndev, open_flag)
        self._error_check(ret)
        return
    
    def close(self):
        """
        2. MtnClose
        """
        self._log('close')
        ret = libgpg7400.MtnClose(self.ndev)
        self._error_check(ret)
        return
    
    def reset(self, axis='XYZU', mode='MTR_RESET_CTL'):
        """
        3. MtnReset
        """
        self._log('reset')
        axis = AxisConfig(axis)
        mode = ResetMode.verify(mode)
        ret = libgpg7400.MtnReset(self.ndev, axis, mode)
        self._error_check(ret)
        return
    
    def set_pulse_out(self, axis='XYZU', mode='MTR_METHOD', config=''):
        """
        4. MtnSetPulseOut
        """
        self._log('set_pulse_out')
        axis = AxisConfig(axis)
        mode = PulseOutConfigSection.verify(mode)
        if mode=='MTR_METHOD': config = PulseOutMethod(config)
        elif mode=='MTR_IDLING': config = int(config)
        elif mode=='MTR_FINISH_FLAG': config = FinishFlag.verify(config)
        elif mode=='MTR_SYNC_OUT': config = SyncOutMode.verify(config)
        ret = libgpg7400.MtnSetPulseOut(self.ndev, axis, mode, config)
        self._error_check(ret)
        return
    
    def set_limit_config(self, axis='XYZU', mode='MTR_METHOD', config=''):
        """
        5. MtnSetLimitConfig
        """
        self._log('set_limit_config')
        axis = AxisConfig(axis)
        mode = LimitConfigSection.verify(mode)
        if mode=='MTR_LOGIC': config = LimitConfigLogic(config)
        elif mode=='MTR_SD_FUNC': config = SDFunc.verify(config)
        elif mode=='MTR_SD_ACTIVE': config = SDActive.verify(config)
        elif mode=='MTR_ORG_FUNC': config = ORGFunc.verify(config)
        elif mode=='MTR_ORG_EZ_COUNT': config = int(config)
        elif mode=='MTR_ALM_FUNC': config = ALMFunc.verify(config)
        elif mode=='MTR_SIGNAL_FILTER': config = SignalFilter.verify(config)
        elif mode=='MTR_EL_FUNC': config = ELFunc.verify(config)
        elif mode=='MTR_EZ_ACTIVE': config = EZActive.verify(config)
        elif mode=='MTR_LTC_FUNC': config = LTCFunc.verify(config)
        elif mode=='MTR_CLR_FUNC': config = CLRFunc.verify(config)
        elif mode=='MTR_PCS_FUNC': config = PCSFunc.verify(config)
        elif mode=='MTR_PCS_ACTIVE': config = None
        ret = libgpg7400.MtnSetLimitConfig(self.ndev, axis, mode, config)
        self._error_check(ret)
        return
    
    def set_counter_config(self, axis='XYZU', mode='MTR_ENCODER_MODE', config='MTR_SINGLE'):
        """
        6. MtnSetCounterConfig
        """
        self._log('set_counter_config')
        axis = AxisConfig(axis)
        mode = CounterConfigSection.verify(mode)
        if mode=='MTR_ENCODER_MODE': config = EncoderMode.verify(config)
        elif mode=='MTR_COUNTER_CLEAR_ORG': config = CounterClearORG(config)
        elif mode=='MTR_COUNTER_CLEAR_CLR': config = CounterClearORG(config)
        elif mode=='MTR_LATCH_MODE': config = LatchMode.verify(config)
        elif mode=='MTR_DECLINO_MODE': config = DeclinoMode.verify(config)
        elif mode=='MTR_SOFT_LATCH': config = None
        ret = libgpg7400.MtnSetCounterConfig(self.ndev, axis, mode, config)
        self._error_check(ret)
        return
    
    def set_comparator(self, axis='XYZU', comp_no='MTR_COMP1', config=None, motion=None,
                       counter=None, cont_type=None):
        """
        7. MtnSetComparator
        """
        self._log('set_comparator')
        axis = AxisConfig(axis)
        comp_no = Comparator.verify(comp_no)
        comp = (libgpg7400.MTNCOMP * 4)()
        for i, ax in enumerate(axis.get_ind_on()):
            comp[ax].wConfig = CompConfig.verify(config[i])
            comp[ax].wMotion = CompMotion.verify(motion[i])
            comp[ax].lCounter = int(counter[i])
            comp[ax].wCntType = CompCntType.verify(cont_type[i])
            continue
        ret = libgpg7400.MtnSetConparator(self.ndev, axis, comp_no, comp)
        self._error_check(ret)
        return
    
    def set_sync(self, axis='XYZU', mode='MTR_START_MODE', config='MTR_NO'):
        """
        8. MtnSetSync
        """
        self._log('set_sync')
        axis = AxisConfig(axis)
        mode = SyncSection.verify(mode)
        if mode=='MTR_START_MODE': config = StartMode.verify(config)
        elif mode=='MTR_EXT_STOP': config = ExtStop.verify(config)
        elif mode=='MTR_START_LINE': config = StartLine.verify(config)
        elif mode=='MTR_STOP_LINE': config = StopLine.verify(config)
        ret = libgpg7400.MtnSetSync(self.ndev, axis, mode, comfig)
        self._error_check(ret)
        return
    
    def set_revise(self, axis='XYZU', mode='MTR_PULSE', config=1):
        """
        9. MtnSetRevise
        """
        self._log('set_revise')
        axis = AxisConfig(axis)
        mode = ReviseSection.verify(mode)
        if mode=='MTR_PULSE': config = int(config)
        elif mode=='MTR_REVISE_MODE': config = ReviseMode.verify(config)
        elif mode=='MTR_COUNTER_MODE': config = ReviseCounterMode(config)
        elif mode=='MTR_REST_RT': config = int(config)
        elif mode=='MTR_REST_FT': config = int(config)
        ret = libgpg7400.MtnSetRevise(self.ndev, axis, mode, config)
        self._error_check(ret)
        return
    
    def set_erc_config(self, axis='XYZU', mode='MTR_AUTO', config=0):
        """
        10. MtnSetERCConfig
        """
        self._log('set_erc_config')
        axis = AxisConfig(axis)
        mode = ERCConfigSection.verify(mode)
        if mode=='MTR_AUTO': config = ERCAuto(config)
        elif mode=='MTR_LOGIC': config = ERCLogic.verify(config)
        elif mode=='MTR_WIDTH': config = ERCWidth.verify(config)
        elif mode=='MTR_OFF_TIMER': config = ERCOffTimer.verify(config)
        elif mode=='MTR_SIGNAL_ON': config = None
        elif mode=='MTR_SIGNAL_OFF': config = None
        ret = libgpg7400.MtnSetERCConfig(self.ndev, axis, mode, config)
        self._error_check(ret)
        return
    
    def set_motion(self, axis='XYZU', mode='MTR_JOG', 
                   clock=[], accmode=[], lowspeed=[], speed=[], acc=[], dec=[],
                   accspeed=[], decspeed=[], step=[]):
        """
        11. MtnSetMotion
        """
        self._log('set_motion')
        axis = AxisConfig(axis)
        mode = MotionSection.verify(mode)
        motion = (libgpg7400.MTNMOTION * 4)()
        for i, ax in enumerate(axis.get_ind_on()):
            motion[ax].wClock = int(clock[i])
            motion[ax].wAccMode = MotionAccMode.verify(accmode[i])
            motion[ax].fLowSpeed = float(lowspeed[i])
            motion[ax].fSpeed = float(speed[i])
            motion[ax].ulAcc = int(acc[i])
            motion[ax].ulDec = int(dec[i])
            motion[ax].fSAccSpeed = float(accspeed[i])
            motion[ax].fSDecSpeed = float(decspeed[i])
            motion[ax].lStep = int(step[i])
            continue
        ret = libgpg7400.MtnSetMotion(self.ndev, axis, mode, motion)
        self._error_check(ret)
        return
    
    def set_motion_line(self, mode='MTR_LINE_NORMAL', 
                        axis='XYZU', clock=4095, linemode='MTR_LINE',
                        accmode='MTR_ACC_NORMAL', lowspeed=1.0, speed=1.0, acc=1,
                        dec=1, accspeed=0.0, decspeed=0.0, step=[0,0,0,0]):
        """
        12. MtnSetMotionLine
        """
        self._log('set_motion_line')
        mode = MotionLineSection.verify(mode)
        line = libgpg7400.MTNLINE()
        line.wAxis = AxisConfig(axis)
        line.wClock = int(clock)
        line.wMode = LineMode(linemode)
        line.wAccMode = LineAccMode(accmode)
        line.fLowSpeed = float(lowspeed)
        line.fSpeed = float(speed)
        line.ulAcc = int(acc)
        line.ulDec = int(dec)
        line.fSaccSpeed = float(accspeed)
        line.fSDecSpeed = float(decspeed)
        for i, ax in enumerate(axis.get_ind_on()):
            line.lStep[ax] = int(step[i])
            continue
        ret = libgpg7400.MtnSetMotionLine(self.ndev, mode, line)
        self._error_check(ret)
        return
    
    def set_sync_line(self, maxstep=1000, axis='XYZU', clock=4095, linemode='MTR_LINE',
                      accmode='MTR_ACC_NORMAL', lowspeed=1.0, speed=1.0, acc=1,
                      dec=1, accspeed=0.0, decspeed=0.0, step=[0,0,0,0]):
        """
        13. MtnSetSyncLine
        """
        self._log('set_sync_line')
        maxstep = int(maxstep)
        line = libgpg7400.MTNLINE()
        line.wAxis = AxisConfig(axis)
        line.wClock = int(clock)
        line.wMode = LineMode(linemode)
        line.wAccMode = LineAccMode(accmode)
        line.fLowSpeed = float(lowspeed)
        line.fSpeed = float(speed)
        line.ulAcc = int(acc)
        line.ulDec = int(dec)
        line.fSaccSpeed = float(accspeed)
        line.fSDecSpeed = float(decspeed)
        for i, ax in enumerate(axis.get_ind_on()):
            line.lStep[ax] = int(step[i])
            continue
        ret = libgpg7400.MtnSetSyncLine(self.ndev, maxstep, line)
        self._error_check(ret)
        return
    
    def set_motion_arc(self, mode='MTR_ARC_NORMAL', axis='XYZU', clock=4095,
                       arcmode='MTR_ARC_CW', speed=1.0,
                       cetnerx=0, centery=0, endx=0, endy=0):
        """
        14. MtnSetMotionArc
        """
        self._log('set_motion_arc')
        mode = MotionArcSection(mode)
        arc = libgpg7400.MTNARC()
        arc.wAxis = AxisConfig(axis)
        arc.wClock = int(clock)
        arc.wMode = ARCMode(arcmode)
        arc.fSpeed = float(speed)
        arc.lCenterX = int(centerx)
        arc.lCenterY = int(centery)
        arc.lEndX = int(endx)
        arc.lEndY = int(endy)
        ret = libgpg7400.MtnSetMotionArc(self.ndev, mode, arc)
        self._error_check(ret)
        return
    
    def set_motion_cp(self, axis='X', num=2,
                      clock=[], accmode=[], lowspeed=[], speed=[], acc=[], dec=[],
                      accspeed=[], decspeed=[], step=[]):
        """
        15. MtnSetMotionArc
        """
        self._log('set_motion_cp')
        axis = AxisConfig(axis)
        motion = (libgpg7400.MTNMOTION * num)()
        for i in range(num):
            motion[i].wClock = int(clock[i])
            motion[i].wAccMode = MotionAccMode.verify(accmode[i])
            motion[i].fLowSpeed = float(lowspeed[i])
            motion[i].fSpeed = float(speed[i])
            motion[i].ulAcc = int(acc[i])
            motion[i].ulDec = int(dec[i])
            motion[i].fSAccSpeed = float(accspeed[i])
            motion[i].fSDecSpeed = float(decspeed[i])
            motion[i].lStep = int(step[i])
            continue
        ret = libgpg7400.MtnSetMotionCp(self.ndev, axis, num, motion)
        self._error_check(ret)
        return

    
# .................................
# The following is not implemented.
