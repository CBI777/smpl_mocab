# Input key is basically a dictionary to keep track of whole "button pressed" process that occurs inside the project.
# By pressing the key, values inside the dictionary will change. This is done by directly manipulating globally shared
# dictionary, inputKey.

inputKey = {}
# "esc" is for keeping track of whether user pressed the exit button or not.
inputKey["esc"] = False
# "shape" is for keeping track of whether user pressed the shape estimation button or not.
inputKey["shape"] = False
# "obj" is for keeping track of whether user pressed the obj file extraction toggle button or not.
inputKey["obj"] = False

def buttons(key, dx, dy):

    # Taking care of key inputs if bbox+rendered image window is NOT shown.
    # 1. ESC : Shut down the program
    # 2. Spacebar : recalculate shape
    # 3. o button : toggle obj extraction
    if key == b'\x1b':  # exit on ESC
        inputKey["esc"] = True
    if key == b' ': # recalc shape on space
        print("Recalculating shape")
        inputKey["shape"] = True
    if key == b'o': # toggle obj extraction
        inputKey["obj"] = not inputKey["obj"]

    # The keys can always be alternated to other keys.
    # Use b'??' to let python know the "binary input" for certain keys.