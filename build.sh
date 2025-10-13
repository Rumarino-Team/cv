#!/bin/bash
# Simple build script for ROS2 packages with virtual environment
# This script activates the venv automatically

cd ~/Projects/auv/ros2_ws

echo "🔧 Activating virtual environment..."
source ~/Projects/auv/.venv/bin/activate

echo "🔧 Sourcing ROS2..."
source /opt/ros/jazzy/setup.bash

echo "🏗️  Building packages..."
colcon build --symlink-install \
    --cmake-args \
    -DPython3_EXECUTABLE=$HOME/Projects/auv/.venv/bin/python3 \
    -DPython3_NumPy_INCLUDE_DIR=$HOME/Projects/auv/.venv/lib/python3.12/site-packages/numpy/core/include

if [ $? -eq 0 ]; then
    echo ""
    echo "🔧 Fixing shebangs to use virtual environment Python..."
    # Fix shebangs in all Python scripts in install/*/lib/*/
    find install/*/lib/*/ -type f -executable | while read script; do
        if head -n1 "$script" | grep -q "^#!/usr/bin/python"; then
            # Get the venv Python path
            VENV_PYTHON="$HOME/Projects/auv/.venv/bin/python3"
            # Replace the shebang
            sed -i "1s|^#!/usr/bin/python.*|#!$VENV_PYTHON|" "$script"
            echo "  Fixed: $script"
        fi
    done
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "To run your nodes:"
    echo "  source install/setup.bash"
    echo "  ros2 run hydrus_cv depth_publisher"
else
    echo "❌ Build failed!"
    exit 1
fi
