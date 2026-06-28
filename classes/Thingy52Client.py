"""
Thingy:52 BLE Client and Real-Time Inference Interface

This module defines the Thingy52Client class, which inherits from BleakClient. 
It encapsulates the connection management, hardware configuration, data streaming, 
fixed-point byte unpacking, and live neural network prediction using ONNX Runtime.
"""

import os
import asyncio
from datetime import datetime
import struct
import numpy as np
import onnxruntime as ort
from bleak import BleakClient, BLEDevice

from utils.UUIDs import TMS_RAW_DATA_UUID, TMS_CONF_UUID
from utils.utility import motion_characteristics, change_status, get_uuid


class Thingy52Client(BleakClient):

    """
    Custom BLE Client tailored for streaming IMU data from the Nordic Thingy:52 
    and running real-time human activity recognition models.
    """

    def __init__(self, device: BLEDevice):
        # Initialize the base BleakClient using the resolved hardware identifier
        super().__init__(get_uuid(device)) # class constructor of BleakClient need UUID to be passed, not the MAC address

        self.mac_address = device.address
        self.choice = ["record","train","predict"]

        # Recording information
        self.recording_name = None
        self.file = None

        # Machine Learning inference configuration
        # Loads the trained CNN model via ONNX runtime for cross-platform efficiency
        self.model = ort.InferenceSession('training/CNN_180.onnx') # load the ONNX model for inference
        self.classes = ["balling", "shooting"]
        self.buffer_size = 180 # Corresponds to a 3-second temporal window at 60Hz
        self.data_buffer = []


    async def connect(self, **kwargs) -> bool:

        """
        Establish a BLE connection with the target Thingy:52 device.

        Returns:
            bool: True if connection succeeds, False otherwise.
        """

        print(f"Connecting to {self.mac_address}")
        await super().connect(**kwargs)

        try:
            print(f"Connected to {self.mac_address}")
            # Visual feedback: Turn the sensor LED green upon successful handshake
            await change_status(self, "connected")
            return True

        except Exception as e:
            print(f"Failed to connect to {self.mac_address}: {e}")
            return False
        

    def save_to(self, file_name: str) -> None:

        """
        Generate a standardized CSV filename based on the device MAC address.
        """

        self.recording_name = f"{self.mac_address.replace(':', '-')}_{file_name}.csv"


    async def disconnect(self) -> bool:

        """
        Safely terminate the BLE connection and ensure data files are closed.
        """

        print(f"\nDisconnecting from {self.mac_address}")
        if self.file and not self.file.closed:
            self.file.close()
        return await super().disconnect()


    def raw_data_callback(self, sender, data: bytearray) -> None:

        """
        GATT Notification Callback. Unpacks real-time binary IMU payloads, 
        converts fixed-point data, and feeds the sliding window into the CNN model.
        """

        receive_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # Unpack Accelerometer raw data: 16-bit signed integer ('h') in Little-Endian format.
        # Firmware specification: 6Q10 fixed-point format -> Divide by 2^10 (1024) to convert to G forces.
        acc_x = (struct.unpack('<h', data[0:2])[0] * 1.0) / 2 ** 10
        acc_y = (struct.unpack('<h', data[2:4])[0] * 1.0) / 2 ** 10
        acc_z = (struct.unpack('<h', data[4:6])[0] * 1.0) / 2 ** 10

        # Unpack Gyroscope raw data: 16-bit signed integer ('h') in Little-Endian format.
        # Firmware specification: 11Q5 fixed-point format -> Divide by 2^5 (32) to convert to deg/s.
        gyro_x = (struct.unpack('<h', data[6:8])[0] * 1.0) / 2 ** 5
        gyro_y = (struct.unpack('<h', data[8:10])[0] * 1.0) / 2 ** 5
        gyro_z = (struct.unpack('<h', data[10:12])[0] * 1.0) / 2 ** 5

        # Optional: Persist the raw time-series data stream to local disk
        # self.file.write(f"{receive_time},{acc_x},{acc_y},{acc_z},{gyro_x},{gyro_y},{gyro_z}\n")

        # Append the current frame metrics to the sliding window buffer
        self.data_buffer.append([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])

        # Evaluate the buffer window once it reaches capacity
        if len(self.data_buffer) == self.buffer_size:
            # Reshape input array to match the CNN expectations: (Batch, Time-steps, Features)
            input_data = np.array(self.data_buffer, dtype=np.float32).reshape(1, self.buffer_size, 6)
            input_node_name = self.model.get_inputs()[0].name

            # Execute ONNX forward pass (single run optimization)
            predictions = self.model.run(None, {input_node_name: input_data})[0]

            # Parse prediction scores using argmax to select the most likely class index
            cls_index = np.argmax(predictions, axis=1)[0]
            cls_probability = predictions[0][cls_index]
            
            # Print dynamically updating terminal prediction outputs
            print(f"\r{self.mac_address} | {receive_time} - Prediction: {self.classes[cls_index]} ({cls_probability * 100:.2f}%)", end="", flush=True)
            
            # Flush the data buffer to prepare for the subsequent window lifecycle
            self.data_buffer.clear()


    async def receive_inertial_data(self, sampling_frequency: int = 60) -> None:

        """
        Configure the Motion Processing Unit (MPU) variables and enable BLE notifications.
        """

        # Pack and transmit configuration bytes to adjust sensor data processing rate
        payload = motion_characteristics(motion_processing_unit_freq=sampling_frequency)
        await self.write_gatt_char(TMS_CONF_UUID, payload)

        # Ensure directory context exists and open the file stream in append mode
        directory = "training/data"
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, self.recording_name)
        self.file = open(file_path, "a+")

        # Activate real-time notifications on the raw data characteristic endpoint
        await self.start_notify(TMS_RAW_DATA_UUID, self.raw_data_callback)

        # Visual feedback: Turn the sensor LED red to indicate an active recording/streaming session
        await change_status(self, "recording")

        try:
            # Keep the event loop alive to capture async notifications continuously
            while True:
                await asyncio.sleep(0.1)

        except asyncio.CancelledError:
            # Clean up handles properly upon receiving cancellation requests (e.g., keyboard interrupts)
            await self.stop_notify(TMS_RAW_DATA_UUID)
            await self.disconnect()
            print("Stopped notification and safely disconnected device.")