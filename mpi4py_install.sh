#!/bin/bash

# Usage: ./workaround_mpi4py.sh -e <environment_name>
while getopts ":e:" opt; do
  case $opt in
    e)
      ENV_NAME="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [[ -z "$ENV_NAME" ]]; then
  echo "Error: Environment name is required. Use -e <environment_name> flag."
  exit 1
fi

# Get conda base directory
CONDA_BASE=$(conda info --base)

# Backup original ld
cd "$CONDA_BASE"/envs/"$ENV_NAME"/compiler_compat || exit 1
mv ld ld.bak

# Create symlink to system ld
ln -s /usr/bin/ld ld

module add PE-gnu/4.0
# Retry mpi4py installation
conda run -n "$ENV_NAME" pip install mpi4py==3.1.6 --no-cache-dir

# Restore original ld
rm ld
mv ld.bak ld

echo "mpi4py workaround applied for environment '$ENV_NAME'."