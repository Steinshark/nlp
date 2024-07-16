
import sys
import time
import ctypes
import numpy

import pyinterface
import libgpg7204 as lib
pIE = pyinterface.IdentiferElement


# Identifer Wrapper
# =================

class BoardID(pyinterface.Identifer):
    id742020 = pIE('id742020', 742020)
    id7204 = pIE('id7204', 7204)
    pass

# 5: MtrSetBaseClock
# ------------------
class BaseClockMode(pyinterface.Identifer):
    MTR_CLOCK_1M = pIE('MTR_CLOCK_1M', lib.MTR_CLOCK_1M)
    MTR_CLOCK_1_4M = pIE('MTR_CLOCK_1_4M', lib.MTR_CLOCK_1_4M)
    MTR_CLOCK_1_16M = pIE('MTR_CLOCK_1_16M', lib.MTR_CLOCK_1_16M)
    pass

# 6: MtrSetPulseOut
# -----------------
class PulseOutMode(pyinterface.Identifer):
    MTR_METHOD = pIE('MTR_METHOD', lib.MTR_METHOD)
    MTR_FINISH_FLAG = pIE('MTR_FINISH_FLAG', lib.MTR_FINISH_FLAG)
    pass

class PulseOutMethod(pyinterface.BitIdentifer):
    size = 32
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('PULSE', 'CW/CCW', 'OUT/DIR')
    bits[1].set_params('LOGIC', 'NEGATIVE', 'POSITIVE')
    bits[2].set_params('DIR', 'NEGATIVE', 'POSITIVE')
    pass

class PulseOutFinishFlag(pyinterface.Identifer):
    MTR_PULSE_OUT = pIE('MTR_PULSE_OUT', lib.MTR_PULSE_OUT)
    MTR_INP = pIE('MTR_INP', lib.MTR_INP)
    pass

# 7: MtrLimitConfig
# -----------------
class LimitConfigMode(pyinterface.Identifer):
    MTR_MASK = pIE('MTR_MASK', lib.MTR_MASK)
    MTR_LOGIC = pIE('MTR_LOGIC', lib.MTR_LOGIC)
    MTR_SIGNAL_FILTER = pIE('MTR_SIGNAL_FILTER', lib.MTR_SIGNAL_FILTER)
    MTR_DI = pIE('MTR_DI', lib.MTR_DI)
    MTR_LIMIT = pIE('MTR_LIMIT', lib.MTR_LIMIT)
    MTR_SYNC_EXT = pIE('MTR_SYNC_EXT', lib.MTR_SYNC_EXT)
    pass

class LimitConfigMask(pyinterface.BitIdentifer):
    size = 32
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('+SD', 'ENABLED', 'DISABLED')
    bits[1].set_params('-SD', 'ENABLED', 'DISABLED')
    bits[2].set_params('+EL', 'ENABLED', 'DISABLED')
    bits[3].set_params('-EL', 'ENABLED', 'DISABLED')
    bits[5].set_params('ORG', 'ENABLED', 'DISABLED')
    bits[6].set_params('ALM', 'ENABLED', 'DISABLED')
    pass

class LimitConfigLogic(pyinterface.BitIdentifer):
    size = 32
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('+SD', 'NEGATIVE', 'POSITIVE')
    bits[1].set_params('-SD', 'NEGATIVE', 'POSITIVE')
    bits[2].set_params('+EL', 'NEGATIVE', 'POSITIVE')
    bits[3].set_params('-EL', 'NEGATIVE', 'POSITIVE')
    bits[5].set_params('ORG', 'NEGATIVE', 'POSITIVE')
    bits[6].set_params('ALM', 'NEGATIVE', 'POSITIVE')
    bits[8].set_params('INP', 'NEGATIVE', 'POSITIVE')
    pass

class LimitConfigSignalFilter(pyinterface.Identifer):
    MTR_1MICRO = pIE('MTR_1MICRO', lib.MTR_1MICRO)
    MTR_10MICRO = pIE('MTR_10MICRO', lib.MTR_10MICRO)
    MTR_100MICRO = pIE('MTR_100MICRO', lib.MTR_100MICRO)
    pass

# 9: MtrSetSync
# -------------
class SyncType(pyinterface.Identifer):
    MTR_RESET_SYNC_START = pIE('MTR_RESET_SYNC_START', lib.MTR_RESET_SYNC_START)
    MTR_SET_SYNC_START = pIE('MTR_SET_SYNC_START', lib.MTR_SET_SYNC_START)
    pass

# 10: MtrSetMotion
# ----------------
class MotionMode(pyinterface.Identifer):
    MTR_JOG = pIE('MTR_JOG', lib.MTR_JOG)
    MTR_ORG = pIE('MTR_ORG', lib.MTR_ORG)
    MTR_PTP = pIE('MTR_PTP', lib.MTR_PTP)
    pass

# 17: MtrStartMotion
# ------------------
class StartMotionMode(pyinterface.Identifer):
    MTR_JOG = pIE('MTR_JOG', lib.MTR_JOG)
    MTR_ORG = pIE('MTR_ORG', lib.MTR_ORG)
    MTR_PTP = pIE('MTR_PTP', lib.MTR_PTP)
    MTR_CONST = pIE('MTR_CONST', lib.MTR_CONST)
    pass

# 18: MtrSingleStep
# -----------------
class SingleStepDir(pyinterface.Identifer):
    MTR_CW = pIE('MTR_CW', lib.MTR_CW)
    MTR_CCW = pIE('MTR_CCW', lib.MTR_CCW)
    pass

# 20: MtrStopMotion
# -----------------
class StopMotionMode(pyinterface.Identifer):
    MTR_DEC_STOP = pIE('MTR_DEC_STOP', lib.MTR_DEC_STOP)
    MTR_IMMEDIATE_STOP = pIE('MTR_IMMEDIATE_STOP', lib.MTR_IMMEDIATE_STOP)
    pass

# 21: MtrChangeSpeed
# ------------------
class ChangeSpeedMode(pyinterface.Identifer):
    MTR_IMMEDIATE_CHANGE = pIE('MTR_IMMEDIATE_CHANGE', lib.MTR_IMMEDIATE_CHANGE)
    MTR_ACCDEC_CHANGE = pIE('MTR_ACCDEC_CHANGE', lib.MTR_ACCDEC_CHANGE)
    pass

# 22: MtrGetStatus
# ----------------
class StatusMode(pyinterface.Identifer):
    MTR_BUSY = pIE('MTR_BUSY', lib.MTR_BUSY)
    MTR_FINISH_STATUS = pIE('MTR_FINISH_STATUS', lib.MTR_FINISH_STATUS)
    MTR_LIMIT_STATUS = pIE('MTR_LIMIT_STATUS', lib.MTR_LIMIT_STATUS)
    MTR_INTERLOCK_STATUS = pIE('MTR_INTERLOCK_STATUS', lib.MTR_INTERLOCK_STATUS)
    MTR_ERROR_CODE = pIE('MTR_ERROR_CODE', lib.MTR_ERROR_CODE)
    pass

class StatusBusy(pyinterface.BitIdentifer):
    size = 32
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('BUSY', 'NO', 'YES')
    bits[1].set_params('ACC', 'NO', 'YES')
    bits[2].set_params('DEC', 'NO', 'YES')
    bits[3].set_params('WAIT', 'OFF', 'ON')
    bits[5].set_params('STOP', 'OFF', 'ON')
    bits[6].set_params('INP', 'OFF', 'ON')
    pass

class StatusFinish(pyinterface.BitIdentifer):
    size = 32
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('FNS', 'NO', 'YES')
    bits[1].set_params('CMD', 'NO', 'YES')
    bits[8].set_params('+SD', 'NO', 'YES')
    bits[9].set_params('-SD', 'NO', 'YES')
    bits[10].set_params('+EL', 'NO', 'YES')
    bits[11].set_params('-EL', 'NO', 'YES')
    bits[13].set_params('ORG', 'NO', 'YES')
    bits[14].set_params('ALM', 'NO', 'YES')
    pass

class StatusLimit(pyinterface.BitIdentifer):
    size = 32
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('+SD', 'NO', 'YES')
    bits[1].set_params('-SD', 'NO', 'YES')
    bits[2].set_params('+EL', 'NO', 'YES')
    bits[3].set_params('-EL', 'NO', 'YES')
    bits[5].set_params('ORG', 'NO', 'YES')
    bits[6].set_params('ALM', 'NO', 'YES')
    bits[8].set_params('INP', 'NO', 'YES')
    pass

class StatusInterlock(pyinterface.Identifer):
    MTR_ILOCK_OFF = pIE('MTR_ILOCK_OFF', lib.MTR_ILOCK_OFF)
    MTR_ILOCK_ON = pIE('MTR_ILOCK_ON', lib.MTR_ILOCK_ON)
    pass

class StatusError(pyinterface.Identifer):
    NO_ERROR = pIE('NO_ERROR', 0x00)
    NOT_DEFINED = pIE('NOT_DEFINED', 0x01)
    NOT_INITIALIZED= pIE('NOT_INITIALIZED', 0x02)
    CANT_START_EL_SD_ALM = pIE('CANT_START_EL_SD_ALM', 0x03)
    CANT_START_0_MOVE = pIE('CANT_START_0_MOVE', 0x04)
    RECV_STOP_DURING_STAYING = pIE('RECV_STOP_DURING_STAYING', 0x05)
    RECV_DATA_WO_CMD = pIE('RECV_DATA_WO_CMD', 0x06)
    RECV_CMD_INSTEAD_OF_DATA = pIE('RECV_CMD_INSTEAD_OF_DATA', 0x07)
    CANT_START_ORG_INPUT = pIE('CANT_START_ORG_INPUT', 0x08)
    RECV_CMD_INEXCUTABLE = pIE('RECV_CMD_INEXCUTABLE', 0x09)
    WRONG_INIT_VEL = pIE('WRONG_INIT_VEL', 0x0A)
    WRONG_INIT_ACC_PULS = pIE('WRONG_INIT_ACC_PULS', 0x0B)
    WRONG_INIT_FASTER_START_SPEED = pIE('WRONG_INIT_FASTER_START_SPEED', 0x0C)
    VEL_OUTRANGE = pIE('VEL_OUTRANGE', 0x0D)
    SD_OR_RECV_VEL_CHANGE_DURING_DEC = pIE('SD_OR_RECV_VEL_CHANGE_DURING_DEC', 0x0E)
    SD_OR_RECV_STOP_DEC_DURING_DEC = pIE('SD_OR_RECV_STOP_DEC_DURING_DEC', 0x0F)
    RECV_VEL_CHANGE_DURING_STAYING = pIE('RECV_VEL_CHANGE_DURING_STAYING', 0x10)
    WRONG_INIT_STEP = pIE('WRONG_INIT_STEP', 0x11)
    WRONG_INIT_VEL = pIE('WRONG_INIT_VEL', 0x12)
    WRONG_INIT_PULS = pIE('WRONG_INIT_PLUS', 0x13)
    pass

# 27: MtrOutputDO
# ---------------
class OutputDO(pyinterface.BitIdentifer):
    size = 32
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('OUT1', 'HIGH', 'LOW')
    bits[1].set_params('OUT2', 'HIGH', 'LOW')
    bits[2].set_params('OUT3', 'HIGH', 'LOW')
    bits[3].set_params('OUT4', 'HIGH', 'LOW')
    bits[4].set_params('OUT5', 'HIGH', 'LOW')
    bits[5].set_params('OUT6', 'HIGH', 'LOW')
    bits[6].set_params('OUT7', 'HIGH', 'LOW')
    bits[7].set_params('OUT8', 'HIGH', 'LOW')
    pass

# 28: MtrInputDI
# ---------------
class InputDI(pyinterface.BitIdentifer):
    size = 32
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('IN1', 'HIGH', 'LOW')
    bits[1].set_params('IN2', 'HIGH', 'LOW')
    bits[2].set_params('IN3', 'HIGH', 'LOW')
    bits[3].set_params('IN4', 'HIGH', 'LOW')
    bits[4].set_params('IN5', 'HIGH', 'LOW')
    bits[5].set_params('IN6', 'HIGH', 'LOW')
    bits[6].set_params('IN7', 'HIGH', 'LOW')
    bits[7].set_params('IN8', 'HIGH', 'LOW')
    pass

# 29: MtrSetEvent
# ---------------
class EventMask(pyinterface.BitIdentifer):
    size = 32
    bits = [pyinterface.BitIdentiferElement(i) for i in range(size)]
    del(i)
    bits[0].set_params('STOP', 'MASK', 'UNMASK')
    bits[8].set_params('IN1', 'MASK', 'UNMASK')
    bits[9].set_params('IN2', 'MASK', 'UNMASK')
    bits[10].set_params('IN3', 'MASK', 'UNMASK')
    bits[11].set_params('IN4', 'MASK', 'UNMASK')
    pass

# MTRMOTION.dwMode
# ----------------
class MotionAccMode(pyinterface.Identifer):
    MTR_ACC_NORMAL = pIE('MTR_ACC_NORMAL', lib.MTR_ACC_NORMAL)
    MTR_ACC_SIN = pIE('MTR_ACC_SIN', lib.MTR_ACC_SIN)
    MTR_ACC_ORIGINAL = pIE('MTR_ACC_ORIGINAL', lib.MTR_ACC_ORIGINAL)
    pass


# Error Wrapper
# =============
class ErrorGPG7204(pyinterface.ErrorCode):
    MTR_ERROR_SUCCESS = pIE('MTR_ERROR_SUCCESS', lib.MTR_ERROR_SUCCESS)
    MTR_ERROR_NOT_DEVICE = pIE('MTR_ERROR_NOT_DEVICE', lib.MTR_ERROR_NOT_DEVICE)
    MTR_ERROR_NOT_OPEN = pIE('MTR_ERROR_NOT_OPEN', lib.MTR_ERROR_NOT_OPEN)
    MTR_ERROR_INVALID_DEVICE_NUMBER = pIE('MTR_ERROR_INVALID_DEVICE_NUMBER', lib.MTR_ERROR_INVALID_DEVICE_NUMBER)
    MTR_ERROR_ALREADY_OPEN = pIE('MTR_ERROR_ALREADY_OPEN', lib.MTR_ERROR_ALREADY_OPEN)
    MTR_ERROR_NOT_SUPPORTED = pIE('MTR_ERROR_NOT_SUPPORTED', lib.MTR_ERROR_NOT_SUPPORTED)
    MTR_ERROR_NOW_MOVING = pIE('MTR_ERROR_NOW_MOVING', lib.MTR_ERROR_NOW_MOVING)
    MTR_ERROR_NOW_STOPPED = pIE('MTR_ERROR_NOW_STOPPED', lib.MTR_ERROR_NOW_STOPPED)
    MTR_ERROR_WRITE_FAILED = pIE('MTR_ERROR_WRITE_FAILED', lib.MTR_ERROR_WRITE_FAILED)
    MTR_ERROR_READ_FAILED = pIE('MTR_ERROR_READ_FAILED', lib.MTR_ERROR_READ_FAILED)
    MTR_ERROR_INVALID_AXIS = pIE('MTR_ERROR_INVALID_AXIS', lib.MTR_ERROR_INVALID_AXIS)
    MTR_ERROR_INVALID_SPEED = pIE('MTR_ERROR_INVALID_SPEED', lib.MTR_ERROR_INVALID_SPEED)
    MTR_ERROR_INVALID_ACCDEC = pIE('MTR_ERROR_INVALID_ACCDEC', lib.MTR_ERROR_INVALID_ACCDEC)
    MTR_ERROR_INVALID_PULSE = pIE('MTR_ERROR_INVALID_PULSE', lib.MTR_ERROR_INVALID_PULSE)
    MTR_ERROR_INVALID_PARAMETER = pIE('MTR_ERROR_INVALID_PARAMETER', lib.MTR_ERROR_INVALID_PARAMETER)
    MTR_ERROR_NOW_INTERLOCKED = pIE('MTR_ERROR_NOW_INTERLOCKED', lib.MTR_ERROR_NOW_INTERLOCKED)
    MTR_ERROR_IMPOSSIBLE = pIE('MTR_ERROR_IMPOSSIBLE', lib.MTR_ERROR_IMPOSSIBLE)
    MTR_ERROR_NOT_ALLOCATE_MEMORY = pIE('MTR_ERROR_NOT_ALLOCATE_MEMORY', lib.MTR_ERROR_NOT_ALLOCATE_MEMORY)
    
    _success = MTR_ERROR_SUCCESS
    pass



# ==========================
# GPG-7204 Python Controller
# ==========================

class gpg7204(object):
    def __init__(self, ndev=1, remote=False):
        initialize = not remote
        self.ctrl = gpg7204_controller(ndev, initialize=initialize)
        pass
            
    def start(self, speed, low_speed=5, acc=100, dec=100, sspeed=0):
        if speed >= 0: direc = 1
        else: direc = -1
        speed = abs(int(speed))
        self.ctrl.set_motion('MTR_JOG', 'MTR_ACC_NORMAL', low_speed=low_speed, speed=speed, acc=acc, 
                             dec=dec, sspeed=sspeed, step=direc)
        self.ctrl.start_motion('MTR_JOG')
        return
    
    def stop(self):
        self.ctrl.stop_motion()
        return
        
    def change_speed(self, new_speed, mode='MTR_ACCDEC_CHANGE'):
        new_speed = abs(new_speed)
        self.ctrl.change_speed(mode, new_speed)
        return
    
    def move(self, speed, count, low_speed=5, acc=100, dec=100, sspeed=0):
        if speed >= 0: direc = 1
        else: direc = -1
        speed = abs(int(speed))
        count *= direc
        self.ctrl.set_motion('MTR_PTP', 'MTR_ACC_NORMAL', low_speed=low_speed, speed=speed, acc=acc, 
                             dec=dec, sspeed=sspeed, step=count)
        self.ctrl.start_motion('MTR_PTP')
        return
        
    def move_with_lock(self, speed, count, low_speed=5, acc=100, dec=100, sspeed=0):
        initial_position = self.ctrl.get_counter()
        self.move(speed, count, low_speed=5, acc=100, dec=100, sspeed=0)
        while True:
            self.ctrl.print_log = False
            current_speed = self.ctrl.get_speed()
            if current_speed == 0: break
            position = abs(initial_position - self.ctrl.get_counter())
            sys.stdout.write('\rmoving... %5d/%d %d'%(position, abs(count), current_speed))
            sys.stdout.flush()
            time.sleep(0.1)
            continue
        self.ctrl.print_log = True
        print('')
        return
        
    def move_org(self, speed=1000, low_speed=5, acc=100, dec=100, sspeed=0):
        initial_position = self.ctrl.get_counter()
        speed = abs(int(speed))
        step = initial_position * -1
        self.move_with_lock(speed, step, low_speed=low_speed, acc=acc, dec=dec, sspeed=sspeed)
        return
    
    def set_org(self):
        self.ctrl.clear_counter()
        return
        
    def get_position(self):
        ret = self.ctrl.get_counter()
        return ret
        
    def di_check(self):
        ret = self.ctrl.input_di()
        return ret

    def do_output(self, ch, output_time=100):
        self.ctrl.output_do('OUT%d'%ch)
        if output_time==0: return
        time.sleep(output_time/1000.)
        self.ctrl.output_do(0)
        return


class gpg7204_controller(object):
    ndev = int()
    boardid = ''
    print_log = True
    
    def __init__(self, ndev=1, boardid=742020, initialize=True):
        """
        boardid = 7204 or 742020
        """
        self.ndev = ndev
        self.boardid = BoardID.verify(boardid)
        if initialize: self.initialize()
        return
    
    def _log(self, msg):
        if self.print_log:
            print('Interface GPG7204(%d): %s'%(self.ndev, msg))
            pass
        return
        
    def _error_check(self, error_no):
        ErrorGPG7204.check(error_no)
        return
        
    def initialize(self):
        self.open()
        return
        
    def open(self):
        """
        1. MtrOpen
        """
        self._log('open')
        ret = lib.MtrOpen(self.ndev)
        self._error_check(ret)
        return
    
    def close(self):
        """
        2. MtrClose
        """
        self._log('close')
        ret = lib.MtrClose(self.ndev)
        self._error_check(ret)
        return
    
    def reset(self):
        """
        3. MtrReset
        """
        self._log('reset')
        ret = lib.MtrReset(self.ndev)
        self._error_check(ret)
        return
        
    def off_inter_lock(self):
        """
        4. MtrInterLock
        """
        self._log('off_inter_lock')
        ret = lib.MtrOffInterLock(self.ndev)
        self._error_check(ret)
        return

    def set_base_clock(self, clock):
        """
        5. MtrBaseClock
        ---------------
        clock : 
        """
        self._log('set_base_clock')
        if self.boardid==7204:
            clock = BaseClockMode.verify(clock)
        else:
            clock = int(clock)
            pass
        ret = lib.MtrSetBaseClock(self.ndev, clock)
        self._error_check(ret)
        return

    def set_pulse_out(self, mode='MTR_METHOD', config=0):
        """
        6. MtrSetPulseOut
        -----------------
        mode : 'MTR_METHOD' | 'MTR_FINISH_FLAG'
        config : 
            mode = 'MTR_METHOD' ::  PULSE, OUT, DIR
            mode = 'MTR_FINISH_FLAG' :: MTR_PULSE_OUT | MTR_INP
        """
        self._log('set_pulse_out')
        mode = PulseOutMode.verify(mode)
        if mode=='MTR_METHOD': config = PulseOutMethod(config)
        elif mode=='MTR_FINISH_FLAG': config = PulseOutFinishFlag.verify(config)
        ret = lib.MtrSetPulseOut(self.ndev, mode, config)
        self._error_check(ret)
        return

    def set_limit_config(self, mode='MTR_MASK', config=0):
        """
        7. MtrSetLimitConfig
        --------------------
        """
        self._log('set_limit_config')
        mode = LimitConfigMode.verify(mode)
        if mode=='MTR_MASK': config = LimitConfigMask(config)
        elif mode=='MTR_LOGIC': config = LimitConfigLogic(config)
        elif mode=='MTR_SIGNAL_FILTER': config = LimitConfigSignalFilter.verify(config)
        ret = lib.MtrSetLimitConfig(self.ndev, mode, config)
        self._error_check(ret)
        return

    def set_acc_curve(self, speed, acc):
        """
        8. MtrSetAccCurve
        -----------------
        """
        self._log('set_acc_curve')
        size = len(speed)
        orig_acc = (lib.MTRORIGINALACC * size)()
        for i in range(size):
            orig_acc[i].dwSpeed = speed[i]
            orig_acc[i].dwAcc = acc[i]
            continue
        ret = lib.MtrSetAccCurve(self.ndev, size, orig_acc)
        self._error_check(ret)
        return

    def set_sync(self, sync):
        """
        9. MtrSetSync
        -------------
        """
        self._log('set_sync')
        sync = SyncType.verify(sync)
        ret = lib.MtrSetSync(self.ndev, sync)
        self._error_check(ret)
        return

    def set_motion(self, mode='MTR_JOG', motion_mode='MTR_ACC_NORMAL',
                   low_speed=10, speed=100, acc=50, dec=50, sspeed=50, step=0):
        """
        10. MtrSetMotion
        ----------------
        """
        self._log('set_motion')
        mode = MotionMode.verify(mode)
        print(mode)
        motion = lib.MTRMOTION()
        motion.dwMode = MotionAccMode.verify(motion_mode)
        motion.dwLowSpeed = low_speed
        motion.dwSpeed = speed
        motion.dwAcc = acc
        motion.dwDec = dec
        motion.dwSSpeed = sspeed
        motion.nStep = int(step)
        motion.nReserved = 0
        print(motion)
        ret = lib.MtrSetMotion(self.ndev, mode, motion)
        self._error_check(ret)
        return

    def get_base_clock(self):
        """
        11. MtrGetBaseClock
        -------------------
        """
        self._log('get_base_clock')
        clock = ctypes.c_ulong(0)
        ret = lib.MtrGetBaseClock(self.ndev, clock)
        self._error_check(ret)
        if self.boardid==7204:
            clock = BaseClockMode.verify(clock.value)
        else:
            clock = int(clock.value)
            pass
        return clock

    def get_pulse_out(self, mode='MTR_METHOD'):
        """
        12. MtrGetPulseOut
        -----------------
        mode : 'MTR_METHOD' | 'MTR_FINISH_FLAG'
        """
        self._log('get_pulse_out')
        mode = PulseOutMode.verify(mode)
        config = ctypes.c_ulong(0)
        ret = lib.MtrGetPulseOut(self.ndev, mode, config)
        self._error_check(ret)
        if mode=='MTR_METHOD': config = PulseOutMethod(config.value)
        elif mode=='MTR_FINISH_FLAG': config = PulseOutFinishFlag.verify(config.value)
        return config

    def get_limit_config(self, mode='MTR_MASK'):
        """
        13. MtrGetLimitConfig
        --------------------
        """
        self._log('get_limit_config')
        mode = LimitConfigMode.verify(mode)
        config = ctypes.c_ulong(0)
        ret = lib.MtrGetLimitConfig(self.ndev, mode, config)
        self._error_check(ret)
        if mode=='MTR_MASK': config = LimitConfigMask(config.value)
        elif mode=='MTR_LOGIC': config = LimitConfigLogic(config.value)
        elif mode=='MTR_SIGNAL_FILTER': config = LimitConfigSignalFilter.verify(config.value)
        return config

    def get_acc_curve(self):
        """
        14. MtrGetAccCurve
        -----------------
        """
        self._log('get_acc_curve')
        size = ctypes.c_ulong(0)
        orig_acc = (lib.MTRORIGINALACC * 96)()
        ret = lib.MtrGetAccCurve(self.ndev, size, orig_acc)
        self._error_check(ret)
        return orig_acc[:size.value]

    def get_sync(self):
        """
        15. MtrGetSync
        -------------
        """
        self._log('get_sync')
        sync = ctypes.c_ulong(0)
        ret = lib.MtrGetSync(self.ndev, sync)
        self._error_check(ret)
        sync = SyncType.verify(sync.value)
        return sync

    def get_motion(self, mode='MTR_JOG'):
        """
        16. MtrGetMotion
        ----------------
        """
        self._log('get_motion')
        mode = MotionMode.verify(mode)
        motion = lib.MTRMOTION()
        ret = lib.MtrGetMotion(self.ndev, mode, motion)
        print(motion)
        self._error_check(ret)
        return motion

    def start_motion(self, mode='MTR_JOG'):
        """
        17. MtrStartMotion
        ------------------
        """
        self._log('start_motion')
        mode = StartMotionMode.verify(mode)
        ret = lib.MtrStartMotion(self.ndev, mode)
        self._error_check(ret)
        return 

    def single_step(self, direc='MTR_CW'):
        """
        18. MtrSingleStep
        -----------------
        """
        self._log('single_step')
        direc = SingleStepDir.verify(direc)
        ret = lib.MtrSingleStep(self.ndev, direc)
        self._error_check(ret)
        return 

    def start_sync(self):
        """
        19. MtrStartSync
        ----------------
        """
        self._log('start_sync')
        ret = lib.MtrStartSync(self.ndev)
        self._error_check(ret)
        return 

    def stop_motion(self, mode='MTR_DEC_STOP'):
        """
        20. MtrStopMotion
        -----------------
        """
        self._log('stop_motion')
        mode = StopMotionMode.verify(mode)
        ret = lib.MtrStopMotion(self.ndev, mode)
        self._error_check(ret)
        return 

    def change_speed(self, mode='MTR_ACCDEC_CHANGE', speed=None):
        """
        21. MtrChangeSpeed
        ------------------
        """
        self._log('change_speed')
        mode = ChangeSpeedMode.verify(mode)
        ret = lib.MtrChangeSpeed(self.ndev, mode, speed)
        self._error_check(ret)
        return 

    def get_status(self, mode='MTR_BUSY'):
        """
        22. MtrGetStatus
        ----------------
        """
        self._log('get_status')
        mode = StatusMode.verify(mode)
        status = ctypes.c_ulong(0)
        ret = lib.MtrGetStatus(self.ndev, mode, status)
        self._error_check(ret)
        if mode=='MTR_BUSY': status = StatusBusy(status.value)
        elif mode=='MTR_FINISH_STATUS': status = StatusFinish(status.value)
        elif mode=='MTR_LIMIT_STATUS': status = StatusLimit(status.value)
        elif mode=='MTR_INTERLOCK_STATUS': status = StatusInterlock.verify(status.value)
        elif mode=='MTR_ERROR_CODE': status = StatusError.verify(srtatus.value)        
        return status

    def get_speed(self):
        """
        23. MtrReadSpeed
        ----------------
        """
        self._log('read_speed')
        speed = ctypes.c_ulong(0)
        ret = lib.MtrReadSpeed(self.ndev, speed)
        self._error_check(ret)
        return speed.value

    def get_counter(self):
        """
        24. MtrReadCounter
        ------------------
        """
        self._log('read_counter')
        pos = ctypes.c_long(0)
        ret = lib.MtrReadCounter(self.ndev, lib.MTR_COUNTER, pos)
        self._error_check(ret)
        return pos.value

    def set_counter(self, pos):
        """
        25. MtrWriteCounter
        -------------------
        """
        self._log('write_counter')
        ret = lib.MtrWriteCounter(self.ndev, lib.MTR_COUNTER, pos)
        self._error_check(ret)
        return 

    def clear_counter(self):
        """
        26. MtrClearCounter
        -------------------
        """
        self._log('clear_counter')
        ret = lib.MtrClearCounter(self.ndev, lib.MTR_COUNTER)
        self._error_check(ret)
        return 

    def output_do(self, out):
        """
        27. MtrOutputDO
        ---------------
        """
        self._log('output_do')
        out = OutputDO(out)
        ret = lib.MtrOutputDO(self.ndev, out)
        self._error_check(ret)
        return 

    def input_di(self):
        """
        28. MtrInputDI
        ---------------
        """
        self._log('input_di')
        din = ctypes.c_ulong(0)
        ret = lib.MtrInputDI(self.ndev, din)
        self._error_check(ret)
        din = InputDI(din.value)
        return din


