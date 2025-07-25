# utils/volume_control.py
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize system volume interface
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get current range (min, max in dB)
minVol, maxVol, _ = volume.GetVolumeRange()


def set_volume(direction):
    current = volume.GetMasterVolumeLevel()
    step = 3.0  # in dB
    if direction == "up":
        new_vol = min(current + step, maxVol)
    else:
        new_vol = max(current - step, minVol)
    volume.SetMasterVolumeLevel(new_vol, None)

def get_volume_level():
    current = volume.GetMasterVolumeLevel()
    return (current - minVol) / (maxVol - minVol)
