"""
Nordic Thingy:52 BLE UUID Constants

This module defines the 128-bit vendor-specific (custom) Universally Unique Identifiers 
(UUIDs) for the GATT architecture of the Nordic Thingy:52 sensor. These addresses allow 
the Bluetooth Low Energy (BLE) client to map and interact with the device's hardware.

Understanding the Address Hierarchy:
- Services (Macro-groups): Typically end in '00' (e.g., ef680400-...) and represent 
  broad functional modules or departments (e.g., Motion Service, Environment Service).
- Characteristics (Data Channels): End in unique sequential bytes (e.g., ef680406-...) 
  and act as specific endpoints or mailboxes. They are used to read sensor metrics, 
  write configuration packets, or subscribe to real-time data notifications.

For detailed specifications of the firmware architecture, refer to the official documentation:
https://nordicsemiconductor.github.io/Nordic-Thingy52-FW/documentation/firmware_architecture.html
"""

# ==============================================================================
# Thingy Configuration Service (TCS)
# ==============================================================================
TCS_UUID              = 'ef680100-9b35-4933-9b10-52ffa9740042'

# ==============================================================================
# Thingy Environment Service (TES)
# ==============================================================================
TES_UUID              = 'ef680200-9b35-4933-9b10-52ffa9740042'
TES_TEMP_UUID         = 'ef680201-9b35-4933-9b10-52ffa9740042'
TES_PRESS_UUID        = 'ef680202-9b35-4933-9b10-52ffa9740042'
TES_HUMID_UUID        = 'ef680203-9b35-4933-9b10-52ffa9740042'
TES_GAS_UUID          = 'ef680204-9b35-4933-9b10-52ffa9740042'
TES_COLOR_UUID        = 'ef680205-9b35-4933-9b10-52ffa9740042'
TES_CONF_UUID         = 'ef680206-9b35-4933-9b10-52ffa9740042'

# ==============================================================================
# User Interface Service (UIS)
# ==============================================================================
UIS_UUID              = 'ef680300-9b35-4933-9b10-52ffa9740042'
UIS_LED_UUID          = 'ef680301-9b35-4933-9b10-52ffa9740042'
UIS_BTN_UUID          = 'ef680302-9b35-4933-9b10-52ffa9740042'
UIS_PIN_UUID          = 'ef680303-9b35-4933-9b10-52ffa9740042'

# ==============================================================================
# Thingy Motion Service (TMS)
# ==============================================================================
TMS_UUID              = 'ef680400-9b35-4933-9b10-52ffa9740042'
TMS_CONF_UUID         = 'ef680401-9b35-4933-9b10-52ffa9740042'
TMS_TAP_UUID          = 'ef680402-9b35-4933-9b10-52ffa9740042'
TMS_ORIENTATION_UUID  = 'ef680403-9b35-4933-9b10-52ffa9740042'
TMS_QUATERNION_UUID   = 'ef680404-9b35-4933-9b10-52ffa9740042'
TMS_STEP_COUNTER_UUID = 'ef680405-9b35-4933-9b10-52ffa9740042'
TMS_RAW_DATA_UUID     = 'ef680406-9b35-4933-9b10-52ffa9740042'  # Crucial for IMU data streaming
TMS_EULER_UUID        = 'ef680407-9b35-4933-9b10-52ffa9740042'
TMS_ROTATION_UUID     = 'ef680408-9b35-4933-9b10-52ffa9740042'
TMS_HEADING_UUID      = 'ef680409-9b35-4933-9b10-52ffa9740042'
TMS_GRAVITY_UUID      = 'ef68040a-9b35-4933-9b10-52ffa9740042'

# ==============================================================================
# Thingy Sound Service (TSS)
# ==============================================================================
TSS_UUID              = 'ef680500-9b35-4933-9b10-52ffa9740042'
TSS_CONF_UUID         = 'ef680501-9b35-4933-9b10-52ffa9740042'
TSS_SPEAKER_DATA_UUID = 'ef680502-9b35-4933-9b10-52ffa9740042'
TSS_SPEAKER_STAT_UUID = 'ef680503-9b35-4933-9b10-52ffa9740042'
TSS_MIC_UUID          = 'ef680504-9b35-4933-9b10-52ffa9740042'