#!/bin/bash

# Delete everything except .sh, .cu, .cc, and .c files
find . -type f ! \( -name "*.sh" -o -name "*.cu" -o -name "*.cc" -o -name "*.c" \) -delete

# Optionally remove empty directories after files are deleted
find . -type d -empty -delete

echo "Cleaned all files except .sh, .cu, .cc, and .c"
