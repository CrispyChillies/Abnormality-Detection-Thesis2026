#!/bin/bash
# Script to delete all files ending with ':Zone.Identifier' in the repository

find . -type f -name '*:Zone.Identifier' -exec rm -v {} +

echo "All ':Zone.Identifier' files have been deleted."
