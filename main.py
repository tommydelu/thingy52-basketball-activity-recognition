"""
Main Execution Entry Point

This script handles the user interface configuration, discovers the designated
Nordic Thingy:52 hardware, establishes the asynchronous BLE connection, and initiates
the real-time inertial data streaming pipeline.
"""

import asyncio
from classes.Thingy52Client import Thingy52Client
from utils.utility import scan, find


async def main(recording_label: str):

    my_thingy_addresses = ["F1:58:6C:E2:D8:44"] # The Thingy:52 device's MAC address (replace with your device's address if different)
    
    print("Initializing Bluetooth environment scan...")
    discovered_devices = await scan()
    
    my_devices = find(discovered_devices, my_thingy_addresses)
    
    # Error Guard: Ensure the device is powered on and within range before proceeding
    if not my_devices:
        print(f"\n[ERROR] Target Thingy:52 ({my_thingy_addresses[0]}) not found.")
        print("Please verify the device is turned ON, batteries are charged, and within BLE range.")
        return

    thingy52 = Thingy52Client(my_devices[0])
    
    # Attempt to establish the BLE connection handshake
    connection_success = await thingy52.connect()
    if not connection_success:
        return

    # Set up the target filename using the pre-verified user input string
    thingy52.save_to(recording_label)
    
    # Open the streaming channels and begin model evaluation / CSV logging
    print("\nStarting live inertial data processing stream. Press Ctrl+C to terminate.")
    await thingy52.receive_inertial_data()


if __name__ == '__main__':

    user_label = str(input("Enter recording session label (e.g., 'session1', 'test_shoot'): ")).strip()
    
    if not user_label:
        user_label = "default_session"

    try:
        asyncio.run(main(user_label))
    except KeyboardInterrupt:
        print("\n\nApplication terminated manually by user. Exiting safely.")