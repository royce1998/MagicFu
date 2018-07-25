Hi and welcome to MagicFu version 1.0!

MagicFu is a visual prediction program for the Raspberry Pi.
This README.txt details the hardware setup, software setup, and the usage.

============================================================================

HARDWARE

Required Hardware:
	Raspberry Pi 3 Model B+
	SD Card (32 GB minimum) & SD Card Reader
	USB (8 GB minimum) as Swap
	Ethernet Cable
	Fan & Heatsink
	Mouse & Keyboard for setting up Raspberry Pi
	Pi Camera

Hardware Set-Up:
	Install fan and heatsink on the Raspberry Pi to prevent overheating.
	Insert SD Card (32 GB+) and blank USB (8 GB+) into their respective slots.
	Connect Raspberry Pi to another computer via Ethernet.
	Connect the Pi Camera.
	Power on Raspberry Pi.

=============================================================================

SOFTWARE

Install Operating System Image:
	Download Raspbian Stretch Image https://downloads.raspberrypi.org/raspbian_latest
	Unzip get the Img file
	Writing the image to the SD card
		Download Etcher https://etcher.io/
		Connect an SD card reader with the SD card inside
		Open Butcher and follow the instructions to write the SD card.

Enable SSH:
	Connect the Raspberry pi with electricity, mouse, keyboard, monitor, fan and Ethernet cable, and waiting for system properly turn on. (for first time may need 3 minutes)
		>>>sudo systemctl enable ssh
		>>>sudo systemctl start ssh

Connect to Raspberry Pi via SSH on Ubuntu:
	Open terminal
		>>>ifconfig
	Find the ip address
	Then this Raspberry Pi is accessible via SSH with 
	>username:pi & password:raspberry

Set Up a SWAP:
	Insert your USB drive, and find the /dev/XXX path for the device.
		>>>sudo blkid
	The path may look like  /dev/sda1
	Once you've found your device, unmount it
		>>>sudo umount /dev/XXX
	Then format the device to be swap:
		>>>sudo mkswap /dev/XXX
	If the previous command outputted an alphanumeric UUID, copy that now. Otherwise, find the UUID by running blkid again. Copy the UUID associated with /dev/XXX
		>>>sudo blkid
	Now edit your /etc/fstab file to register your swap file.
		>>>sudo nano /etc/fstab
	On a separate line, enter the following information. Replace the X's with the UUID (without quotes)
		>>>UUID=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX none swap sw,pri=5 0 0
		(use CTRL+X to save & exit)
	Save /etc/fstab
		>>>sudo swapon -a

Python 3.5.3 , pip3:
	>>> sudo apt-get install python3 python3-pip

Tensorflow 1.8.0 Dependencies:
	>>>sudo apt-get install python-pip python-numpy swig python-dev
	>>>sudo pip install wheel
	>>>sudo apt-get install gcc-4.8 g++-4.8
	>>>sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 100
	>>>sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 100

Install Tensorflow via Wheel:
	>>>wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.8.0/tensorflow-1.8.0-cp35-none-linux_armv7l.whl
	>>>sudo pip3 install tensorflow-1.8.0-cp35-none-linux_armv7l.whl
	>>>sudo pip3 uninstall mock
	>>>sudo pip3 install mock

Keras Dependencies:
	>>>sudo apt-get install libhdf5-serial-dev
	>>>pip install h5py
	>>>pip install pillow imutils
	>>>pip install scipy --no-cache-dir

Install Keras:
	>>>pip install keras==2.1.5

PiCamera:
	>>> sudo apt-get install python3-picamera

==================================================================================

USAGE

Video Requirements:
	Must use an empty background, devoid of distractions and unnecessary objects.
	Objects must fully be within the frame, not cut off.
	Objects cannot overlap.
	Objects must be at least two meters apart from each other.
	Person must wear dinstinct-colored clothing.

How to Use magicfu.py:
	Move into the MagicFu-1.0 directory.
	>>> python3 magicfu.py
	If you want the camera and timings to show, use debugging mode. Note that debugging mode will have slower processing times due to camera image preview.
	>>> python3 magicfu.py --debug=on
	Note that debug mode may sometimes leave the camera on. If the camera is stuck, reboot the Raspberry Pi.
