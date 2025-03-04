import serial

# Establish serial communication
arduino = serial.Serial('COM9', 9600)  # Replace 'COM3' with your port.


# # Send '0' to Arduino to turn light off
def sendCommend(command):
    if command == 1:
        command = b'1'
    elif command == 2:
        command = b'2'
    else:
        command = b'0'
    arduino.write(command)
    # print('=>' * 10, command, '=>' * 10)
