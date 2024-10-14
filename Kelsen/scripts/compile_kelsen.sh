#!/bin/bash

# List the files in the order you want to concatenate them
FILES=(
    "../kelsen-files/BajaCaliforniaCivilCode/DeclaracionUnilateral/Strings/strings.kelsen"
    "../kelsen-files/BajaCaliforniaCivilCode/DeclaracionUnilateral/Subjects/subjects.kelsen"
    "../kelsen-files/BajaCaliforniaCivilCode/DeclaracionUnilateral/Assets/assets.kelsen"
    "../kelsen-files/BajaCaliforniaCivilCode/DeclaracionUnilateral/Clauses/clauses.kelsen"
)


# Output file name
OUTPUT="../kelsen-files/BajaCaliforniaCivilCode/DeclaracionUnilateral/combined.kelsen"

# Remove the output file if it already exists
rm -f $OUTPUT

# Concatenate all files into the output file
for file in "${FILES[@]}"; do
    if [[ -f "$file" ]]; then
        cat "$file" >> $OUTPUT
        echo "" >> $OUTPUT # Add a newline to separate the files
    else
        echo "File $file not found, skipping..."
    fi
done

# Compile the combined file
../build/kelsen $OUTPUT

