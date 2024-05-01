import ctypes
from ctypes import wintypes
################################
#0: Orientamento standard (landscape)
#1: Ruotato di 90 gradi in senso orario (portrait)
#2: Ruotato di 180 gradi (landscape invertito)
#3: Ruotato di 270 gradi in senso orario (portrait invertito)
##################################

# Carica la user32.dll
user32 = ctypes.WinDLL('user32')

# Ottieni le informazioni del display corrente
def get_screen_orientation():
    # Struttura DEVMODE: https://docs.microsoft.com/en-us/windows/win32/api/wingdi/ns-wingdi-devmodea
    class DEVMODE(ctypes.Structure):
        _fields_ = [
            ("dmDeviceName", wintypes.BYTE * 32),
            ("dmSpecVersion", wintypes.WORD),
            ("dmDriverVersion", wintypes.WORD),
            ("dmSize", wintypes.WORD),
            ("dmDriverExtra", wintypes.WORD),
            ("dmFields", wintypes.DWORD),
            ("dmOrientation", ctypes.c_short),  # Modificato qui
            # Altri campi omessi per brevità
        ]

    # Creazione di un'istanza DEVMODE
    dm = DEVMODE()
    dm.dmSize = ctypes.sizeof(DEVMODE)

    # EnumDisplaySettings: https://docs.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-enumdisplaysettingsa
    if user32.EnumDisplaySettingsA(None, 0, ctypes.byref(dm)):
        return dm.dmOrientation
    else:
        return None

# Esempio di uso
orientation = get_screen_orientation()
print("L'orientamento dello schermo è:", orientation)
