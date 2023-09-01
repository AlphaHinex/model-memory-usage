check_dirs := src

# this target runs checks on all files
quality:
	black --required-version 23 --check $(check_dirs)
	ruff $(check_dirs)

# Format source code automatically and check is there are any problems left that need manual fixing
style:
	black --required-version 23 $(check_dirs)
	ruff $(check_dirs) --fix
