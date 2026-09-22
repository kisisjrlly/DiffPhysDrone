#!/usr/bin/bash
# Exit on any error
set -e

# Variables
PASSWORD=
CURRENT_USER=$(logname)
USER_HOME="/home/$CURRENT_USER"
RACER_WS_DIR="$USER_HOME/star_exploration_ws/"
TMP_DIR="$USER_HOME/install_lkh"

# Function to install ROS and other dependencies
install_dependencies() {
	echo "[STAR Lab] Installing dependencies..."
	# echo $password | sudo -S apt-get update -y

	# Install ros dependencies
	echo "$PASSWORD" | sudo -S apt-get install -y ros-noetic-cv-bridge ros-noetic-tf2-ros ros-noetic-tf2-geometry-msgs ros-noetic-tf2-eigen ros-noetic-tf2

	# Install other dependencies
	echo "$PASSWORD" | sudo -S apt-get install -y libglfw3-dev libglew-dev libzmqpp-dev libdw-dev
}

# Function to install LKH
install_lkh() {
	echo "[STAR Lab] Installing LKH to $TMP_DIR..."

	# Create a temporary directory and navigate to it
	mkdir -p "$TMP_DIR" && cd "$TMP_DIR"

	# Download and extract the package
	wget http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.6.tgz
	tar xvfz LKH-3.0.6.tgz

	# Navigate to the extracted directory
	cd LKH-3.0.6

	# Compile the software
	make

	echo "$PASSWORD" | sudo -S cp LKH /usr/local/bin

	# Clean up temporary directory
	rm -rf "$TMP_DIR"

	echo "[STAR Lab] LKH has been successfully installed!"
}

#Function to install NLopt
install_nlopt() {
	TMP_NLOPT_DIR=$USER_HOME/install_nlopt
	echo "[STAR Lab] Installing NLopt to $TMP_NLOPT_DIR..."

	# Create a temporary directory and navigate to it
	mkdir -p "$TMP_NLOPT_DIR" && cd "$TMP_NLOPT_DIR"

	# Download the package
	git clone git@github.com:stevengj/nlopt.git

	cd nlopt

	mkdir build && cd build

	# Compile the software
	cmake .. && make -j2

	echo "$PASSWORD" | sudo make install

	# Clean up temporary directory
	rm -rf "$TMP_NLOPT_DIR"

	echo "[STAR Lab] NLopt has been successfully installed!"
}

# Function to install racer
install_racer() {
	echo "[STAR Lab] Installing racer..."

	# Navigate to the workspace directory, create src directory if not exist
	echo "work tree dir $RACER_WS_DIR"
	[ ! -d "$RACER_WS_DIR" ] && mkdir -p "$RACER_WS_DIR"
	cd "$RACER_WS_DIR"
	[ ! -d "src" ] && mkdir -p src
	cd src
	[ -d "swarm_exploration_project" ] && rm swarm_exploration_project -rf

	# Clone the required git repository
	git clone git@github.com:SYSU-STAR/swarm_exploration_project.git --branch=project
	cd ..

	# Build the project
	catkin_make

	echo "[STAR Lab] RACER has been successfully installed!"
}

# Main execution
main() {
	echo "Please enter your password: "

	read -sr PASSWORD

	install_dependencies

	[ ! -f "/usr/local/bin/LKH" ] && install_lkh

	[ ! -f "/usr/local/lib/libnlopt.so" ] && install_nlopt

	echo "[SYSU-STAR] All dependencies installed successfully!"
}

main
