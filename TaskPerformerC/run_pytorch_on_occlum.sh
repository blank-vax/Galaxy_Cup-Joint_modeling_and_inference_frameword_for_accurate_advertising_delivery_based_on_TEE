#!/bin/bash
set -e

BLUE='\033[1;34m'
NC='\033[0m'

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null 2>&1 && pwd )"
python_dir="$script_dir/occlum_instance/image/opt/python-occlum"

occlum new occlum_instance
cd occlum_instance && rm -rf image
copy_bom -f ../pytorch.yaml --root image --include-dir /opt/occlum/etc/template

if [ ! -d $python_dir ];then
    echo "Error: cannot stat '$python_dir' directory"
    exit 1
fi

new_json="$(jq '.resource_limits.user_space_size = "20480MB" |
                .resource_limits.kernel_space_heap_size = "1024MB" |
                .resource_limits.kernel_space_stack_size = "128MB" |
                .resource_limits.max_num_of_threads = 64 |
                .process.default_stack_size = "4MB" |
                .process.default_heap_size = "256MB" |
                .process.default_mmap_size = "4000MB" |
                .env.default += ["PYTHONHOME=/opt/python-occlum"]' Occlum.json)" && \
echo "${new_json}" > Occlum.json
occlum build --sgx-mode SIM

# Run the python demo
# echo -e "${BLUE}occlum run /bin/python3 demo.py${NC}"
# occlum run /bin/python3 demo.py

# The test for federated learning codes
echo -e "${BLUE}occlum run /bin/python3 train.py${NC}"
occlum run /bin/python3 train.py

echo -e "${RED} training finished! "
