
# 2020Game Models
These are the scripts for use in the 2020 FRC Robotics Competition

* Color Detection
    * The Color Detection models (located in "color_detection")
        * ```simple_model.py```
            * Uses Dense Network with Categorical Crossentropy
    * CSV scripts
        * ```compile_csv.py```
            * Combines color csv files to larger compilations
        * csv_to_data.py
            * Converts csv files to .npz file with data and labels

* Edge Detection (located in "edge_detection")
    * ```Sobel Gradients```
        * edge_detection.py
            * Returns grayscale uint8 matrices

* Hexagon Localization (located in "hexagon_localization")
    * Hexagon Detection (located in "models")
        * ```hexagon_exist.py```
            * Looks at an image of arbitrary size and checks if it has a hexagon in it
    * Hexagon Bounding (located in "models")
        * ```hexagon_aabb.py```
            * Looks at an image of camera size (1920x1080?) and outputs bounding boxes of objects.

* Power Cell Localization (located in "power_cell_localization")
    * Power Cell Detection (located in "models")
        * ```power_cell_exist.py```
            * Looks at an image of arbitrary size and checks if it has a hexagon in it
    * Power Cell Bounding (located in "models")
        * ```power_cell_aabb.py```
            * Looks at an image of camera size (1920x1080?) and outputs bounding boxes of objects.