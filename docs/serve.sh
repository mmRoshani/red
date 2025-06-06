#!/bin/bash

# Install dependencies
echo "Installing dependencies..."
bundle install

# Serve the Jekyll site
echo "Starting Jekyll server..."
bundle exec jekyll serve --livereload --host 0.0.0.0 --port 4000 