import hexdump

# Put your tflite model binary from .cc file into the list
model = [0x20, 0x00, 0x00, ] #...

hexstring = ''.join('%02X' % b for b in model)

with open("name_of_your_model.tflite", "wb") as file:
    file.write(hexdump.dehex(hexstring))
