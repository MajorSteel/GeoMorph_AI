import json

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# GeoMorph AI - Example"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["print('Welcome to GeoMorph AI')"]
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("examples/train_example.ipynb", "w") as f:
    json.dump(notebook_content, f)

with open("examples/inference_example.ipynb", "w") as f:
    json.dump(notebook_content, f)

with open("examples/gis_export.ipynb", "w") as f:
    json.dump(notebook_content, f)

print("Created notebooks.")
