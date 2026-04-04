#!/bin/bash

echo "Fetching latest updates from donutsoft/ml2mqtt..."
git fetch upstream

echo ""
echo "Merging upstream changes into your custom branch..."
git merge upstream/main

echo ""
echo "If there are no merge conflicts, you should now push this to your custom repo!"
echo "Run: git push origin main"
